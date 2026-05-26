import logging
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numba import njit
from torch.utils.data import DataLoader

from preprocessing.stations.gold_pipeline import FEATURES
from src.models.lstm_ae import LSTMAutoencoder
from src.utils.dataset import DeviceDataLoader, NpySequenceDataset

# ---------------------------------------------------------------------------
# Logging — unbuffered, line-flushed on every record
# ---------------------------------------------------------------------------


class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[_FlushHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation grid
# ---------------------------------------------------------------------------

ABLATION_SEEDS: List[int] = [
    42,
    306,
    461,
    808,
    770,
]  # Randomly generated, fixed for reproducibility.
BOTTLENECK_DIMS: List[int] = [8, 16, 32, 64]
LOG_EVERY_N_EPOCHS = 10

# ---------------------------------------------------------------------------
# Feature / weight constants
# ---------------------------------------------------------------------------

CONTINUOUS_FEATURES = [
    "pressure_hPa",
    "precipitation_mm",
    "luminous_intensity_lux",
]
FEATURE_WEIGHTS = torch.tensor([1.5, 2.0, 1.0])
TRAINING_FEATURE_WEIGHTS = torch.tensor([1.5, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
SMOOTHING_WINDOW = 3
MIN_DETECTION_RATIO_PA = 0.1

# ---------------------------------------------------------------------------
# Numba-accelerated hot paths
# ---------------------------------------------------------------------------


@njit(cache=True)
def _rolling_mean_nb(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = i - window + 1
        if start < 0:
            start = 0
        s = 0.0
        for j in range(start, i + 1):
            s += arr[j]
        out[i] = s / (i - start + 1)
    return out


@njit(cache=True)
def _apply_point_adjustment_nb(
    y_true: np.ndarray, y_pred: np.ndarray, min_detection_ratio: float
) -> np.ndarray:
    n = len(y_true)
    y_pred_pa = y_pred.copy()
    i = 0
    while i < n:
        if y_true[i] == 1:
            start = i
            while i < n and y_true[i] == 1:
                i += 1
            block_len = i - start
            min_hits = (
                1
                if int(block_len * min_detection_ratio) < 1
                else int(block_len * min_detection_ratio)
            )
            hits = 0
            for j in range(start, i):
                hits += y_pred[j]
            if hits >= min_hits:
                for j in range(start, i):
                    y_pred_pa[j] = 1
        else:
            i += 1
    return y_pred_pa


@njit(cache=True)
def _binary_metrics_nb(y_true: np.ndarray, y_pred: np.ndarray):
    tp = tn = fp = fn = 0
    for i in range(len(y_true)):
        t = y_true[i]
        p = y_pred[i]
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, recall, f1, fpr


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Set default seed so the module-level state is deterministic before ablation overrides it.
set_seed(42)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_dataset_size(loader) -> int:
    if hasattr(loader, "dataloader"):
        return len(loader.dataloader.dataset)
    return len(loader.dataset)


def prepare_dataloaders(
    npy_path,
    apply_augmentation,
    device,
    batch_size=256,
    val_split=0.1,
    calib_path=None,
):
    full_length = len(np.load(npy_path, mmap_mode="r"))
    val_size = int(full_length * val_split)
    train_size = full_length - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, full_length))

    base_dataset = NpySequenceDataset(npy_path=npy_path, apply_augmentation=False)
    train_dataset = NpySequenceDataset(
        npy_path=npy_path, apply_augmentation=apply_augmentation
    )
    train_std = train_dataset.compute_feature_std_for_indices(train_indices)
    train_sigma = np.maximum(train_std * train_dataset.jitter_scale, 1e-8)
    train_dataset.noise_sigma = torch.tensor(train_sigma, dtype=torch.float32)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(base_dataset, val_indices)

    train_loader = DeviceDataLoader(
        DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        ),
        device,
    )
    val_loader = DeviceDataLoader(
        DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        device,
    )

    if calib_path is not None and Path(calib_path).is_file():
        calib_dataset = NpySequenceDataset(
            npy_path=calib_path, apply_augmentation=False, return_target=True
        )
        if len(calib_dataset) == 0:
            calib_dataset = base_dataset
    else:
        calib_dataset = base_dataset

    calib_loader = DeviceDataLoader(
        DataLoader(
            calib_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        device,
    )
    return train_loader, val_loader, calib_loader


def get_test_loader_and_y(station_prefix, device, batch_size=256):
    x_path = f"{station_prefix}_X_test.npy"
    y_path = f"{station_prefix}_y_test.npy"

    if not Path(x_path).is_file():
        logger.warning("Missing test file: %s", x_path)
        return None, None
    if not Path(y_path).is_file():
        logger.warning("Missing test file: %s", y_path)
        return None, None

    test_dataset = NpySequenceDataset(npy_path=x_path, apply_augmentation=False)
    test_loader = DeviceDataLoader(
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        device,
    )
    y_true = np.load(y_path)
    return test_loader, y_true


def get_calib_path(station_prefix) -> str:
    return f"{station_prefix}_X_calib.npy"


def get_prefix_from_train_path(train_path: Path) -> str:
    return str(train_path).replace("_X_train.npy", "")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=100,
    patience=5,
    feature_weights=None,
    checkpoint_path: Optional[Path] = None,
    label: str = "",
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5
    )

    if feature_weights is not None:
        feature_weights = feature_weights.to(device)

        def weighted_mse_loss(pred, target):
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

        if (epoch + 1) % LOG_EVERY_N_EPOCHS == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "%s Epoch %3d/%d | train=%.6f val=%.6f lr=%.2e",
                label,
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                lr,
            )

        if epochs_no_improve >= patience:
            logger.info(
                "%s Early stop at epoch %d (best val=%.6f)",
                label,
                epoch + 1,
                best_val_loss,
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def compute_reconstruction_error(model, dataloader, feature_weights=None):
    model.eval()
    errors = []
    weights = FEATURE_WEIGHTS if feature_weights is None else feature_weights
    weights = weights.to(next(model.parameters()).device)
    cont_cols = [FEATURES.index(name) for name in CONTINUOUS_FEATURES]
    with torch.no_grad():
        for batch_X, target_X in dataloader:
            reconstructed_X = model(batch_X)
            rec_cont = reconstructed_X[:, :, cont_cols]
            tgt_cont = target_X[:, :, cont_cols]
            mse = torch.mean(
                (tgt_cont - rec_cont) ** 2 * weights.view(1, 1, -1), dim=(1, 2)
            )
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


def calibrate_threshold(model, dataloader, percentile=95, feature_weights=None):
    errors = compute_reconstruction_error(model, dataloader, feature_weights)
    return np.percentile(errors, percentile)


def score_from_smoothed_errors(
    smoothed: np.ndarray, threshold: float, y_true: np.ndarray
):
    y_pred = (smoothed > threshold).astype(np.int64)
    y_true_nb = y_true.astype(np.int64)
    precision, recall, f1, fpr = _binary_metrics_nb(y_true_nb, y_pred)
    y_pred_pa = _apply_point_adjustment_nb(y_true_nb, y_pred, MIN_DETECTION_RATIO_PA)
    precision_pa, recall_pa, f1_pa, fpr_pa = _binary_metrics_nb(y_true_nb, y_pred_pa)
    return precision, recall, f1, fpr, precision_pa, recall_pa, f1_pa, fpr_pa


def evaluate_model(
    model, test_loader, threshold, y_true, feature_weights=None, smoothing_window=3
):
    errors = compute_reconstruction_error(model, test_loader, feature_weights)
    errors_f64 = errors.astype(np.float64)
    smoothed = (
        _rolling_mean_nb(errors_f64, smoothing_window)
        if smoothing_window > 1
        else errors_f64
    )
    return score_from_smoothed_errors(smoothed, threshold, y_true)


# ---------------------------------------------------------------------------
# Core experiment pipeline  (latent_dim + seed params added)
# ---------------------------------------------------------------------------


def run_experiment_pipeline(
    pipeline_name,
    train_paths,
    test_prefixes,
    apply_augmentation,
    device,
    epochs=100,
    is_global=False,
    latent_dim=16,
    seed=42,
    results_path: Optional[Path] = None,
):
    results = []
    label_prefix = f"[{pipeline_name}|dim={latent_dim}|seed={seed}]"

    if is_global:
        logger.info("%s Training global model...", label_prefix)
        train_path = train_paths[0]
        global_prefix = get_prefix_from_train_path(Path(train_path))
        train_loader, val_loader, calib_loader = prepare_dataloaders(
            train_path,
            apply_augmentation,
            device,
            calib_path=get_calib_path(global_prefix),
        )

        model = LSTMAutoencoder(latent_dim=latent_dim).to(device)
        checkpoint_path = (
            results_path.parent
            / f"{pipeline_name.lower().replace(' ', '_')}_dim{latent_dim}_seed{seed}_best.pt"
            if results_path is not None
            else None
        )
        model = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            patience=5,
            feature_weights=TRAINING_FEATURE_WEIGHTS,
            checkpoint_path=checkpoint_path,
            label=label_prefix,
        )

        for prefix in test_prefixes:
            station_name = Path(prefix).name
            test_loader, y_true = get_test_loader_and_y(prefix, device)
            if test_loader is None or y_true is None:
                continue

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
                    "Seed": seed,
                    "BottleneckDim": latent_dim,
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
            logger.info(
                "%s [%s] P %.4f R %.4f F1 %.4f FPR %.4f | PA P %.4f R %.4f F1 %.4f FPR %.4f",
                label_prefix,
                station_name,
                p,
                r,
                f1,
                fpr,
                p_pa,
                r_pa,
                f1_pa,
                fpr_pa,
            )

    else:
        logger.info(
            "%s Training local models (%d stations)...", label_prefix, len(train_paths)
        )
        for t_path, prefix in zip(train_paths, test_prefixes):
            station_name = Path(prefix).name
            calib_path = get_calib_path(prefix)
            train_loader, val_loader, calib_loader = prepare_dataloaders(
                t_path, apply_augmentation, device, calib_path=calib_path
            )

            model = LSTMAutoencoder(latent_dim=latent_dim).to(device)
            checkpoint_path = (
                results_path.parent
                / f"{pipeline_name.lower().replace(' ', '_')}_{station_name}_dim{latent_dim}_seed{seed}_best.pt"
                if results_path is not None
                else None
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
                label=f"{label_prefix}[{station_name}]",
            )

            test_loader, y_true = get_test_loader_and_y(prefix, device)
            if test_loader is None or y_true is None:
                continue

            val_errors = compute_reconstruction_error(
                model, val_loader, feature_weights=FEATURE_WEIGHTS
            )
            test_errors = compute_reconstruction_error(
                model, test_loader, feature_weights=FEATURE_WEIGHTS
            )
            smoothed = _rolling_mean_nb(
                test_errors.astype(np.float64), SMOOTHING_WINDOW
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
                        "Seed": seed,
                        "BottleneckDim": latent_dim,
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
                logger.info(
                    "%s [%s] P%d  P %.4f R %.4f F1 %.4f FPR %.4f | PA P %.4f R %.4f F1 %.4f FPR %.4f",
                    label_prefix,
                    station_name,
                    percentile,
                    p,
                    r,
                    f1,
                    fpr,
                    p_pa,
                    r_pa,
                    f1_pa,
                    fpr_pa,
                )

    if results_path is not None:
        pd.DataFrame(results).to_csv(results_path, index=False)

    return results


# ---------------------------------------------------------------------------
# Single-seed wrapper (used both directly and by the parallel worker)
# ---------------------------------------------------------------------------


def run_single_seed(
    seed: int,
    latent_dim: int,
    pipeline_name: str,
    train_paths,
    test_prefixes,
    apply_augmentation: bool,
    is_global: bool,
    device: torch.device,
    epochs: int,
) -> list:
    set_seed(seed)
    return run_experiment_pipeline(
        pipeline_name=pipeline_name,
        train_paths=train_paths,
        test_prefixes=test_prefixes,
        apply_augmentation=apply_augmentation,
        device=device,
        epochs=epochs,
        is_global=is_global,
        latent_dim=latent_dim,
        seed=seed,
        results_path=None,
    )


# ---------------------------------------------------------------------------
# Multiprocessing worker (top-level: must be picklable)
# ---------------------------------------------------------------------------


def _parallel_worker(kwargs: dict) -> list:
    """Spawned per-seed worker. Sets CUDA_VISIBLE_DEVICES before torch init."""
    gpu_id = kwargs.pop("gpu_id", 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch as _torch  # noqa: PLC0415 — intentional re-import in worker

    _device = _torch.device("cuda:0" if _torch.cuda.is_available() else "cpu")
    return run_single_seed(device=_device, **kwargs)


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

PIPELINE_CONFIGS = [
    {
        "pipeline_name": "Local Baseline",
        "apply_augmentation": False,
        "is_global": False,
    },
    {
        "pipeline_name": "Global Baseline",
        "apply_augmentation": False,
        "is_global": True,
    },
    {
        "pipeline_name": "Local Augmented",
        "apply_augmentation": True,
        "is_global": False,
    },
    {
        "pipeline_name": "Global Augmented",
        "apply_augmentation": True,
        "is_global": True,
    },
]


def run_ablation(
    pipeline_configs: list,
    local_train_files: list,
    local_prefixes: list,
    global_train_file: Path,
    seeds: List[int],
    bottleneck_dims: List[int],
    device: torch.device,
    epochs: int,
    results_dir: Path,
    timestamp: str,
    num_gpus: int = 1,
) -> pd.DataFrame:
    all_rows = []
    total_runs = len(bottleneck_dims) * len(pipeline_configs) * len(seeds)
    run_idx = 0

    for latent_dim in bottleneck_dims:
        logger.info("=" * 70)
        logger.info("BOTTLENECK DIM: %d", latent_dim)
        logger.info("=" * 70)

        for cfg in pipeline_configs:
            train_paths = [global_train_file] if cfg["is_global"] else local_train_files
            common_kwargs = dict(
                latent_dim=latent_dim,
                pipeline_name=cfg["pipeline_name"],
                train_paths=[str(p) for p in train_paths],
                test_prefixes=local_prefixes,
                apply_augmentation=cfg["apply_augmentation"],
                is_global=cfg["is_global"],
                epochs=epochs,
            )

            dim_cfg_rows = []

            if num_gpus > 1:
                # Distribute seeds across GPUs; spawn fresh processes to avoid CUDA fork issues.
                work_items = [
                    {**common_kwargs, "seed": seed, "gpu_id": i % num_gpus}
                    for i, seed in enumerate(seeds)
                ]
                with ProcessPoolExecutor(
                    max_workers=min(len(seeds), num_gpus),
                    mp_context=mp.get_context("spawn"),
                ) as executor:
                    futures = {
                        executor.submit(_parallel_worker, item): item
                        for item in work_items
                    }
                    for fut in as_completed(futures):
                        rows = fut.result()
                        dim_cfg_rows.extend(rows)
                        run_idx += 1
                        logger.info("Completed run %d/%d", run_idx, total_runs)
            else:
                for seed in seeds:
                    logger.info(
                        "--- Run %d/%d: %s | dim=%d | seed=%d ---",
                        run_idx + 1,
                        total_runs,
                        cfg["pipeline_name"],
                        latent_dim,
                        seed,
                    )
                    rows = run_single_seed(seed=seed, device=device, **common_kwargs)
                    dim_cfg_rows.extend(rows)
                    run_idx += 1

            all_rows.extend(dim_cfg_rows)

            # Write partial results immediately so progress survives interruptions.
            partial_df = pd.DataFrame(dim_cfg_rows)
            partial_path = (
                results_dir
                / f"{timestamp}_ablation_dim{latent_dim}_{cfg['pipeline_name'].lower().replace(' ', '_')}.csv"
            )
            partial_df.to_csv(partial_path, index=False)
            logger.info("Partial results saved → %s", partial_path)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "Precision",
    "Recall",
    "F1_Score",
    "FPR",
    "Precision_PA",
    "Recall_PA",
    "F1_Score_PA",
    "FPR_PA",
]
GROUP_KEYS = ["BottleneckDim", "Pipeline", "Station", "Percentile"]


def aggregate_seed_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-seed mean and variance for all metrics."""
    agg = df.groupby(GROUP_KEYS)[METRIC_COLS].agg(["mean", "var"]).reset_index()
    # Flatten MultiIndex columns: ('Precision', 'mean') → 'Precision_mean'
    agg.columns = ["_".join(c).rstrip("_") if c[1] else c[0] for c in agg.columns]
    return agg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info("Device: %s  |  GPUs available: %d", device, num_gpus)

    # Warm up numba JIT so first-epoch timing isn't skewed by compilation.
    logger.info("Warming up Numba JIT...")
    _dummy = np.zeros(10, dtype=np.int64)
    _apply_point_adjustment_nb(_dummy, _dummy, 0.1)
    _binary_metrics_nb(_dummy, _dummy)
    _rolling_mean_nb(np.zeros(10, dtype=np.float64), 3)
    logger.info("Numba JIT warm-up complete.")

    data_dir = Path("data/stations/processed/gold")
    if not data_dir.exists():
        logger.error(
            "Data directory %s does not exist. Run preprocessing first.", data_dir
        )
        return

    IGNORED_STATIONS = ["recinto-guapiles"]

    all_train_files = sorted(
        f
        for f in data_dir.glob("*_X_train.npy")
        if not any(ig in f.name for ig in IGNORED_STATIONS)
    )
    local_train_files = [f for f in all_train_files if "global" not in f.name]
    local_prefixes = [str(f).replace("_X_train.npy", "") for f in local_train_files]

    global_train_file = data_dir / "global_X_train.npy"
    if not global_train_file.exists():
        logger.error("Missing global_X_train.npy. Run extract_global_gold_layer first.")
        return

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y_%m_%d_%H%M")

    logger.info(
        "Ablation grid: %d seeds × %d bottleneck dims × %d pipeline configs = %d total runs",
        len(ABLATION_SEEDS),
        len(BOTTLENECK_DIMS),
        len(PIPELINE_CONFIGS),
        len(ABLATION_SEEDS) * len(BOTTLENECK_DIMS) * len(PIPELINE_CONFIGS),
    )

    raw_df = run_ablation(
        pipeline_configs=PIPELINE_CONFIGS,
        local_train_files=local_train_files,
        local_prefixes=local_prefixes,
        global_train_file=global_train_file,
        seeds=ABLATION_SEEDS,
        bottleneck_dims=BOTTLENECK_DIMS,
        device=device,
        epochs=100,
        results_dir=results_dir,
        timestamp=timestamp,
        num_gpus=max(1, num_gpus),
    )

    raw_path = results_dir / f"{timestamp}_ablation_all_seeds_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    logger.info("Raw per-seed results saved → %s", raw_path)

    agg_df = aggregate_seed_results(raw_df)
    agg_path = results_dir / f"{timestamp}_ablation_aggregated.csv"
    agg_df.to_csv(agg_path, index=False)
    logger.info("Aggregated results (mean ± var across seeds) saved → %s", agg_path)

    # Print summary table: best F1_Score_mean per (Pipeline, BottleneckDim)
    summary_cols = [
        "BottleneckDim",
        "Pipeline",
        "Percentile",
        "F1_Score_mean",
        "F1_Score_var",
        "F1_Score_PA_mean",
        "F1_Score_PA_var",
        "Recall_PA_mean",
        "FPR_PA_mean",
    ]
    available = [c for c in summary_cols if c in agg_df.columns]
    summary = (
        agg_df[available]
        .groupby(["BottleneckDim", "Pipeline", "Percentile"])
        .mean(numeric_only=True)
    )
    logger.info(
        "\n--- Aggregated Summary (mean across stations) ---\n%s", summary.to_string()
    )


if __name__ == "__main__":
    main()
