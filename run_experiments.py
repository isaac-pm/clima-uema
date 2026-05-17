import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path

from src.models.lstm_ae import LSTMAutoencoder
from src.utils.dataset import NpySequenceDataset, DeviceDataLoader


def train_model(model, train_loader, val_loader, device, epochs=50, patience=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataloader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, target_X in val_loader:
                reconstructed_X = model(batch_X)
                loss = criterion(reconstructed_X, target_X)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataloader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def compute_reconstruction_error(model, dataloader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch_X, target_X in dataloader:
            reconstructed_X = model(batch_X)
            # MSE per sequence
            # batch_X shape: (batch_size, 144, 7)
            mse = torch.mean((target_X - reconstructed_X) ** 2, dim=(1, 2))
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


def calibrate_threshold(model, dataloader, device, percentile=95):
    errors = compute_reconstruction_error(model, dataloader, device)
    threshold = np.percentile(errors, percentile)
    return threshold


def evaluate_model(model, test_loader, threshold, device, y_true):
    errors = compute_reconstruction_error(model, test_loader, device)
    y_pred = (errors > threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return precision, recall, f1, fpr


def prepare_dataloaders(
    npy_path, apply_augmentation, device, batch_size=128, val_split=0.1
):
    # Determine split indices sequentially to prevent time-series data leakage
    # We load just the shape via mmap to avoid loading the whole array into memory twice
    full_length = len(np.load(npy_path, mmap_mode="r"))
    val_size = int(full_length * val_split)
    train_size = full_length - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, full_length))

    # Instantiate separate datasets: Train gets augmentation, Val remains pristine
    train_dataset = NpySequenceDataset(
        npy_path=npy_path, apply_augmentation=apply_augmentation
    )
    val_dataset = NpySequenceDataset(npy_path=npy_path, apply_augmentation=False)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # We can still shuffle the train_loader batches, but the data pool is strictly separated from val
    train_loader = DeviceDataLoader(
        DataLoader(train_subset, batch_size=batch_size, shuffle=True), device
    )
    val_loader = DeviceDataLoader(
        DataLoader(val_subset, batch_size=batch_size, shuffle=False), device
    )

    # Calibration uses the entire pristine dataset
    calib_dataset = NpySequenceDataset(npy_path=npy_path, apply_augmentation=False)
    calib_loader = DeviceDataLoader(
        DataLoader(calib_dataset, batch_size=batch_size, shuffle=False), device
    )

    return train_loader, val_loader, calib_loader


def get_test_loader_and_y(station_prefix, device, batch_size=128):
    x_path = f"{station_prefix}_X_test.npy"
    y_path = f"{station_prefix}_y_test.npy"

    test_dataset = NpySequenceDataset(npy_path=x_path, apply_augmentation=False)
    test_loader = DeviceDataLoader(
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False), device
    )
    y_true = np.load(y_path)
    return test_loader, y_true


def run_experiment_pipeline(
    pipeline_name,
    train_paths,
    test_prefixes,
    apply_augmentation,
    device,
    epochs=20,
    is_global=False,
):
    results = []

    if is_global:
        # Train one global model
        print(f"--- Running {pipeline_name} (Global Model) ---")
        train_path = train_paths[0]
        train_loader, val_loader, calib_loader = prepare_dataloaders(
            train_path, apply_augmentation, device
        )

        model = LSTMAutoencoder().to(device)
        model = train_model(model, train_loader, val_loader, device, epochs=epochs)
        threshold = calibrate_threshold(model, calib_loader, device, percentile=95)

        # Evaluate on all local test sets
        for prefix in test_prefixes:
            station_name = Path(prefix).name
            test_loader, y_true = get_test_loader_and_y(prefix, device)
            p, r, f1, fpr = evaluate_model(
                model, test_loader, threshold, device, y_true
            )

            results.append(
                {
                    "Pipeline": pipeline_name,
                    "Station": station_name,
                    "Precision": p,
                    "Recall": r,
                    "F1_Score": f1,
                    "FPR": fpr,
                }
            )
            print(
                f"[{station_name}] P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f}"
            )

    else:
        # Train local model for each station
        print(f"--- Running {pipeline_name} (Local Models) ---")
        for t_path, prefix in zip(train_paths, test_prefixes):
            station_name = Path(prefix).name
            print(f"Training on {station_name}...")

            train_loader, val_loader, calib_loader = prepare_dataloaders(
                t_path, apply_augmentation, device
            )

            model = LSTMAutoencoder().to(device)
            model = train_model(model, train_loader, val_loader, device, epochs=epochs)
            threshold = calibrate_threshold(model, calib_loader, device, percentile=95)

            test_loader, y_true = get_test_loader_and_y(prefix, device)
            p, r, f1, fpr = evaluate_model(
                model, test_loader, threshold, device, y_true
            )

            results.append(
                {
                    "Pipeline": pipeline_name,
                    "Station": station_name,
                    "Precision": p,
                    "Recall": r,
                    "F1_Score": f1,
                    "FPR": fpr,
                }
            )
            print(
                f"[{station_name}] P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f}"
            )

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

    epochs = 20  # You can adjust this for quicker testing or longer convergence
    all_results = []

    # 1. Local Baseline (No Augmentation)
    results_local_base = run_experiment_pipeline(
        "Local Baseline",
        train_paths=local_train_files,
        test_prefixes=local_prefixes,
        apply_augmentation=False,
        device=device,
        epochs=epochs,
        is_global=False,
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
    )
    all_results.extend(results_global_aug)

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(all_results)
    output_path = results_dir / "experiment_metrics.csv"
    df.to_csv(output_path, index=False)

    print(f"\nAll experiments complete! Metrics saved to {output_path}")

    # Display an aggregated summary
    summary = df.groupby("Pipeline")[["Precision", "Recall", "F1_Score", "FPR"]].mean()
    print("\n--- Aggregated Results (Mean across stations) ---")
    print(summary)


if __name__ == "__main__":
    main()
