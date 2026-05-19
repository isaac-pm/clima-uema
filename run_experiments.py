import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path

from src.models.lstm_ae import LSTMAutoencoder
from src.utils.dataset import NpySequenceDataset, DeviceDataLoader
from preprocessing.stations.gold_pipeline import FEATURES


def get_dataset_size(loader) -> int:
    if hasattr(loader, "dataloader"):
        return len(loader.dataloader.dataset)
    return len(loader.dataset)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    patience=5,
    feature_weights=None,
    checkpoint_path: Path | None = None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5
    )

    if feature_weights is not None:
        feature_weights = feature_weights.to(device)
        def weighted_mse_loss(pred, target):
            # Compute loss only over continuous features (indices 0-2).
            # This avoids dividing by the 4 zero-weighted cyclical dimensions,
            # which would otherwise dilute gradients by 7/3.
            cont = [0, 1, 2]
            diff = (pred[:, :, cont] - target[:, :, cont]) ** 2
            return torch.mean(diff * feature_weights[:3].view(1, 1, -1))
        criterion = weighted_mse_loss
    else:
        criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, target_X in train_loader:
            optimizer.zero_grad()
            reconstructed_X = model(batch_X)
            loss = criterion(reconstructed_X, target_X)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= get_dataset_size(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, target_X in val_loader:
                reconstructed_X = model(batch_X)
                loss = criterion(reconstructed_X, target_X)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= get_dataset_size(val_loader)

        scheduler.step(epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            if checkpoint_path is not None:
                torch.save(best_model_state, checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


CONTINUOUS_FEATURES = [
    "pressure_hPa",
    "precipitation_mm",
    "luminous_intensity_lux",
]
FEATURE_WEIGHTS = torch.tensor([1.5, 2.0, 1.0])
TRAINING_FEATURE_WEIGHTS = torch.tensor([1.5, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
SMOOTHING_WINDOW = 3
MIN_DETECTION_RATIO_PA = 0.1  # Fraction of segment windows that must be flagged to trigger PA


def compute_reconstruction_error(model, dataloader, feature_weights=None):
    model.eval()
    errors = []
    weights = feature_weights
    if weights is None:
        weights = FEATURE_WEIGHTS
    weights = weights.to(next(model.parameters()).device)
    with torch.no_grad():
        for batch_X, target_X in dataloader:
            reconstructed_X = model(batch_X)
            # MSE per sequence (continuous features only)
            cont_cols = [FEATURES.index(name) for name in CONTINUOUS_FEATURES]
            rec_cont = reconstructed_X[:, :, cont_cols]
            tgt_cont = target_X[:, :, cont_cols]
            mse = torch.mean(
                (tgt_cont - rec_cont) ** 2 * weights.view(1, 1, -1), dim=(1, 2)
            )
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


def calibrate_threshold(model, dataloader, percentile=95, feature_weights=None):
    errors = compute_reconstruction_error(model, dataloader, feature_weights)
    threshold = np.percentile(errors, percentile)
    return threshold


def calibrate_threshold_optimized(
    model, calib_loader, val_loader, y_val, feature_weights=None,
):
    """Select threshold by maximising F1 on a *labeled validation set*.

    WARNING: ``val_loader`` and ``y_val`` must be a held-out split that is
    entirely separate from both the training set and the final test set.
    Passing the test loader here is label leakage and will inflate all metrics.
    """
    errors = compute_reconstruction_error(model, calib_loader, feature_weights)
    thresholds = np.percentile(errors, np.arange(70, 96, 2))

    best_f1 = 0
    best_threshold = None
    val_errors = compute_reconstruction_error(model, val_loader, feature_weights)

    for thresh in thresholds:
        y_pred = (val_errors > thresh).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    if best_threshold is None:
        best_threshold = np.percentile(errors, 95)
    return best_threshold


def apply_point_adjustment(
    y_true: np.ndarray, y_pred: np.ndarray, min_detection_ratio: float = 0.1
) -> np.ndarray:
    """Adjust predictions within true anomaly segments (Point Adjustment protocol).

    A contiguous ground-truth anomaly segment is credited as detected if at least
    ``min_detection_ratio`` of its windows are flagged.  The threshold scales with
    segment length, preventing trivially small absolute counts for long alert blocks.
    Only predictions inside true-positive blocks are modified; false positives outside
    those blocks are unchanged, so FPR is unaffected.
    """
    y_pred_pa = y_pred.copy()
    in_anomaly = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        elif y_true[i] == 0 and in_anomaly:
            in_anomaly = False
            block_len = i - start
            min_hits = max(1, int(block_len * min_detection_ratio))
            if y_pred[start:i].sum() >= min_hits:
                y_pred_pa[start:i] = 1
    if in_anomaly:
        block_len = len(y_true) - start
        min_hits = max(1, int(block_len * min_detection_ratio))
        if y_pred[start:].sum() >= min_hits:
            y_pred_pa[start:] = 1
    return y_pred_pa


def score_from_smoothed_errors(
    smoothed: np.ndarray, threshold: float, y_true: np.ndarray
):
    y_pred = (smoothed > threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    y_pred_pa = apply_point_adjustment(y_true, y_pred, min_detection_ratio=MIN_DETECTION_RATIO_PA)
    precision_pa = precision_score(y_true, y_pred_pa, zero_division=0)
    recall_pa = recall_score(y_true, y_pred_pa, zero_division=0)
    f1_pa = f1_score(y_true, y_pred_pa, zero_division=0)
    tn_pa, fp_pa, fn_pa, tp_pa = confusion_matrix(y_true, y_pred_pa, labels=[0, 1]).ravel()
    fpr_pa = fp_pa / (fp_pa + tn_pa) if (fp_pa + tn_pa) > 0 else 0.0
    return precision, recall, f1, fpr, precision_pa, recall_pa, f1_pa, fpr_pa


def evaluate_model(
    model,
    test_loader,
    threshold,
    y_true,
    feature_weights=None,
    smoothing_window=3,
):
    errors = compute_reconstruction_error(model, test_loader, feature_weights)
    if smoothing_window and smoothing_window > 1:
        smoothed = (
            pd.Series(errors)
            .rolling(window=smoothing_window, min_periods=1)
            .mean()
            .values
        )
    else:
        smoothed = errors
    return score_from_smoothed_errors(smoothed, threshold, y_true)


def prepare_dataloaders(
    npy_path,
    apply_augmentation,
    device,
    batch_size=256,
    val_split=0.1,
    calib_path=None,
):
    # Determine split indices sequentially to prevent time-series data leakage
    # We load just the shape via mmap to avoid loading the whole array into memory twice
    full_length = len(np.load(npy_path, mmap_mode="r"))
    val_size = int(full_length * val_split)
    train_size = full_length - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, full_length))

    # Instantiate separate datasets: Train gets augmentation, Val remains pristine
    base_dataset = NpySequenceDataset(npy_path=npy_path, apply_augmentation=False)
    train_dataset = NpySequenceDataset(
        npy_path=npy_path, apply_augmentation=apply_augmentation
    )
    train_std = train_dataset.compute_feature_std_for_indices(train_indices)
    train_sigma = np.maximum(train_std * train_dataset.jitter_scale, 1e-8)
    train_dataset.noise_sigma = torch.tensor(train_sigma, dtype=torch.float32)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(base_dataset, val_indices)

    # We can still shuffle the train_loader batches, but the data pool is strictly separated from val
    train_loader = DeviceDataLoader(
        DataLoader(train_subset, batch_size=batch_size, shuffle=True), device
    )
    val_loader = DeviceDataLoader(
        DataLoader(val_subset, batch_size=batch_size, shuffle=False), device
    )

    if calib_path is not None and Path(calib_path).is_file():
        calib_dataset = NpySequenceDataset(
            npy_path=calib_path,
            apply_augmentation=False,
            return_target=True,
        )
        if len(calib_dataset) == 0:
            calib_dataset = base_dataset
    else:
        calib_dataset = base_dataset

    calib_loader = DeviceDataLoader(
        DataLoader(calib_dataset, batch_size=batch_size, shuffle=False), device
    )

    return train_loader, val_loader, calib_loader


def get_test_loader_and_y(station_prefix, device, batch_size=256):
    x_path = f"{station_prefix}_X_test.npy"
    y_path = f"{station_prefix}_y_test.npy"

    if not Path(x_path).is_file():
        print(f"Missing test file: {x_path}")
        return None, None
    if not Path(y_path).is_file():
        print(f"Missing test file: {y_path}")
        return None, None

    test_dataset = NpySequenceDataset(npy_path=x_path, apply_augmentation=False)
    test_loader = DeviceDataLoader(
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False), device
    )
    y_true = np.load(y_path)
    return test_loader, y_true


def get_calib_path(station_prefix) -> str:
    return f"{station_prefix}_X_calib.npy"


def get_prefix_from_train_path(train_path: Path) -> str:
    return str(train_path).replace("_X_train.npy", "")


def run_experiment_pipeline(
    pipeline_name,
    train_paths,
    test_prefixes,
    apply_augmentation,
    device,
    epochs=50,
    is_global=False,
    results_path: Path | None = None,
):
    results = []

    if is_global:
        # Train one global model
        print(f"--- Running {pipeline_name} (Global Model) ---")
        train_path = train_paths[0]
        global_prefix = get_prefix_from_train_path(Path(train_path))
        train_loader, val_loader, calib_loader = prepare_dataloaders(
            train_path,
            apply_augmentation,
            device,
            calib_path=get_calib_path(global_prefix),
        )

        model = LSTMAutoencoder().to(device)
        checkpoint_path = None
        if results_path is not None:
            checkpoint_path = results_path.with_suffix("").with_name(
                f"{pipeline_name.lower().replace(' ', '_')}_best_model.pt"
            )
        model = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            feature_weights=TRAINING_FEATURE_WEIGHTS,
            checkpoint_path=checkpoint_path,
        )

        # Evaluate on all local test sets
        for prefix in test_prefixes:
            station_name = Path(prefix).name
            test_loader, y_true = get_test_loader_and_y(prefix, device)
            if test_loader is None or y_true is None:
                continue

            calib_loader = prepare_dataloaders(
                train_path,
                apply_augmentation,
                device,
                calib_path=get_calib_path(global_prefix),
            )[2]

            threshold = calibrate_threshold(
                model, calib_loader, percentile=95, feature_weights=FEATURE_WEIGHTS
            )
            p, r, f1, fpr, p_pa, r_pa, f1_pa, fpr_pa = evaluate_model(
                model,
                test_loader,
                threshold,
                y_true,
                feature_weights=FEATURE_WEIGHTS,
                smoothing_window=SMOOTHING_WINDOW,
            )

            results.append(
                {
                    "Pipeline": pipeline_name,
                    "Station": station_name,
                    "Percentile": "95",
                    "Precision": p,
                    "Recall": r,
                    "F1_Score": f1,
                    "FPR": fpr,
                    "Precision_PA": p_pa,
                    "Recall_PA": r_pa,
                    "F1_Score_PA": f1_pa,
                    "FPR_PA": fpr_pa,
                }
            )
            print(
                f"[{station_name}] P {p:.4f} R {r:.4f} F1 {f1:.4f} FPR {fpr:.4f}"
                f" | PA: P {p_pa:.4f} R {r_pa:.4f} F1 {f1_pa:.4f} FPR {fpr_pa:.4f}"
            )

        if results_path is not None:
            pd.DataFrame(results).to_csv(results_path, index=False)

    else:
        # Train local model for each station
        print(f"--- Running {pipeline_name} (Local Models) ---")
        for t_path, prefix in zip(train_paths, test_prefixes):
            station_name = Path(prefix).name
            print(f"Training on {station_name}...")

            calib_path = get_calib_path(prefix)
            train_loader, val_loader, calib_loader = prepare_dataloaders(
                t_path, apply_augmentation, device, calib_path=calib_path
            )

            model = LSTMAutoencoder().to(device)
            checkpoint_path = None
            if results_path is not None:
                checkpoint_path = results_path.with_suffix("").with_name(
                    f"{pipeline_name.lower().replace(' ', '_')}_{station_name}_best_model.pt"
                )
            model = train_model(
                model,
                train_loader,
                val_loader,
                device,
                epochs=epochs,
                patience=10,
                feature_weights=TRAINING_FEATURE_WEIGHTS,
                checkpoint_path=checkpoint_path,
            )
            test_loader, y_true = get_test_loader_and_y(prefix, device)
            if test_loader is None or y_true is None:
                continue

            # Compute errors once; sweep percentiles without re-running inference
            val_errors = compute_reconstruction_error(
                model, val_loader, feature_weights=FEATURE_WEIGHTS
            )
            test_errors = compute_reconstruction_error(
                model, test_loader, feature_weights=FEATURE_WEIGHTS
            )
            smoothed = (
                pd.Series(test_errors)
                .rolling(window=SMOOTHING_WINDOW, min_periods=1)
                .mean()
                .values
            )

            for percentile in [85, 90, 95]:
                threshold = np.percentile(val_errors, percentile)
                p, r, f1, fpr, p_pa, r_pa, f1_pa, fpr_pa = score_from_smoothed_errors(
                    smoothed, threshold, y_true
                )
                results.append(
                    {
                        "Pipeline": pipeline_name,
                        "Station": station_name,
                        "Percentile": str(percentile),
                        "Precision": p,
                        "Recall": r,
                        "F1_Score": f1,
                        "FPR": fpr,
                        "Precision_PA": p_pa,
                        "Recall_PA": r_pa,
                        "F1_Score_PA": f1_pa,
                        "FPR_PA": fpr_pa,
                    }
                )
                print(
                    f"[{station_name}] P{percentile}"
                    f" P {p:.4f} R {r:.4f} F1 {f1:.4f} FPR {fpr:.4f}"
                    f" | PA: P {p_pa:.4f} R {r_pa:.4f} F1 {f1_pa:.4f} FPR {fpr_pa:.4f}"
                )

            if results_path is not None:
                pd.DataFrame(results).to_csv(results_path, index=False)

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path("data/stations/processed/gold")

    if not data_dir.exists():
        print(
            f"Data directory {data_dir} does not exist. Please run preprocessing first."
        )
        return

    IGNORED_STATIONS = ["recinto-guapiles"]  # Exclude stations with known issues

    # Find all local train files (excluding global)
    all_train_files = list(data_dir.glob("*_X_train.npy"))
    all_train_files = [
        f
        for f in all_train_files
        if not any(ignored in f.name for ignored in IGNORED_STATIONS)
    ]
    local_train_files = [f for f in all_train_files if "global" not in f.name]
    local_train_files.sort()

    # Deriving prefixes for test loading (strip _X_train.npy)
    local_prefixes = [str(f).replace("_X_train.npy", "") for f in local_train_files]

    global_train_file = data_dir / "global_X_train.npy"
    if not global_train_file.exists():
        print("Missing global_X_train.npy.")
        return

    epochs = 100
    all_results = []

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y_%m_%d_%H%M")

    # 1. Local Baseline (No Augmentation)
    results_local_base = run_experiment_pipeline(
        "Local Baseline",
        train_paths=local_train_files,
        test_prefixes=local_prefixes,
        apply_augmentation=False,
        device=device,
        epochs=epochs,
        is_global=False,
        results_path=results_dir / f"{timestamp}_local_baseline_metrics.csv",
    )
    all_results.extend(results_local_base)

    # 2. Global Baseline (No Augmentation)
    results_global_base = run_experiment_pipeline(
        "Global Baseline",
        train_paths=[global_train_file],
        test_prefixes=local_prefixes,
        apply_augmentation=False,
        device=device,
        epochs=epochs,
        is_global=True,
        results_path=results_dir / f"{timestamp}_global_baseline_metrics.csv",
    )
    all_results.extend(results_global_base)

    # 3. Local Augmented (Denoising)
    results_local_aug = run_experiment_pipeline(
        "Local Augmented",
        train_paths=local_train_files,
        test_prefixes=local_prefixes,
        apply_augmentation=True,
        device=device,
        epochs=epochs,
        is_global=False,
        results_path=results_dir / f"{timestamp}_local_augmented_metrics.csv",
    )
    all_results.extend(results_local_aug)

    # 4. Global Augmented (Denoising)
    results_global_aug = run_experiment_pipeline(
        "Global Augmented",
        train_paths=[global_train_file],
        test_prefixes=local_prefixes,
        apply_augmentation=True,
        device=device,
        epochs=epochs,
        is_global=True,
        results_path=results_dir / f"{timestamp}_global_augmented_metrics.csv",
    )
    all_results.extend(results_global_aug)

    # Save results
    df = pd.DataFrame(all_results)
    output_path = results_dir / f"{timestamp}_experiment_metrics.csv"
    df.to_csv(output_path, index=False)

    print(f"\nAll experiments complete! Metrics saved to {output_path}")

    # Display an aggregated summary
    summary = df.groupby(["Pipeline", "Percentile"])[
        ["Precision", "Recall", "F1_Score", "FPR",
         "Precision_PA", "Recall_PA", "F1_Score_PA", "FPR_PA"]
    ].mean()
    print("\n--- Aggregated Results (Mean across stations) ---")
    print(summary)


if __name__ == "__main__":
    main()
