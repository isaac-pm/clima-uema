"""Dataset and DataLoader utilities for Gold-layer NumPy arrays.

This module supports:
- reading ``.npy`` sequence arrays with optional memory mapping,
- physics-aware data augmentation (Gaussian noise, temporal masking) for continuous features,
- preserving cyclical features,
- batching via ``DataLoader``,
- optional device transfer through a lightweight wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class NpySequenceDataset(Dataset[Union[Tensor, Tuple[Tensor, Tensor]]]):
    """PyTorch dataset for sequence tensors stored in a ``.npy`` file.

    Expected input shape is ``(num_samples, window_size, num_features)``.
    """

    def __init__(
        self,
        npy_path: Union[str, Path],
        station_ids: Optional[Sequence[int]] = None,
        apply_augmentation: bool = False,
        jitter_scale: float = 0.01,
        dropout_prob: float = 0.1,
        dropout_length: Tuple[int, int] = (2, 4),
        continuous_cols: Sequence[int] = (0, 1, 2),
        cyclical_cols: Sequence[int] = (3, 4, 5, 6),
        mmap_mode: Optional[str] = None,
        return_target: bool = True,
        chunk_size: int = 2048,
    ) -> None:
        """Initialize dataset.

        Args:
            npy_path: Path to ``X_train.npy`` (or compatible 3D array).
            apply_augmentation: If True, applies physics-aware augmentations in ``__getitem__``.
            jitter_scale: Multiplicative factor for per-feature std to set noise sigma.
            dropout_prob: Probability of applying temporal masking dropout per sequence.
            dropout_length: Range of consecutive timesteps to mask out.
            continuous_cols: Indices of continuous features (e.g. Pressure, Precipitation, Luminous Intensity).
            cyclical_cols: Indices of cyclical features (e.g. sine/cosine of hour/day).
            mmap_mode: NumPy mmap mode (for example ``"r"``) or None.
            return_target: If True, returns ``(noisy_x, pristine_x)`` for autoencoder training.
            chunk_size: Number of samples per chunk for std computation.
        """
        self.npy_path = Path(npy_path)
        self.apply_augmentation = apply_augmentation
        self.jitter_scale = float(jitter_scale)
        self.dropout_prob = float(dropout_prob)
        self.dropout_length = dropout_length
        self.continuous_cols = list(continuous_cols)
        self.cyclical_cols = list(cyclical_cols)
        self.return_target = return_target
        self.station_ids = list(station_ids) if station_ids is not None else None

        self.data = np.load(self.npy_path, mmap_mode=mmap_mode)
        if self.data.ndim != 3:
            raise ValueError(
                f"Expected 3D array (N, T, F), got shape {self.data.shape} from {self.npy_path}",
            )

        self.num_samples = int(self.data.shape[0])
        self.window_size = int(self.data.shape[1])
        self.num_features = int(self.data.shape[2])

        assert max(self.dropout_length) < self.window_size

        feature_std = self._compute_feature_std(chunk_size=chunk_size)
        sigma = np.maximum(feature_std * self.jitter_scale, 1e-8)
        self.noise_sigma = torch.tensor(sigma, dtype=torch.float32)

    def _compute_feature_std(self, chunk_size: int = 2048) -> np.ndarray:
        """Compute per-feature std over all timesteps using chunked iteration."""
        if self.num_samples == 0:
            return np.ones((self.num_features,), dtype=np.float64)

        sum_x = np.zeros((self.num_features,), dtype=np.float64)
        sum_x2 = np.zeros((self.num_features,), dtype=np.float64)
        total_count = 0

        for start in range(0, self.num_samples, chunk_size):
            end = min(start + chunk_size, self.num_samples)
            chunk = np.asarray(self.data[start:end], dtype=np.float64)
            sum_x += chunk.sum(axis=(0, 1))
            sum_x2 += np.square(chunk).sum(axis=(0, 1))
            total_count += chunk.shape[0] * chunk.shape[1]

        mean = sum_x / total_count
        var = (sum_x2 / total_count) - np.square(mean)
        var = np.maximum(var, 1e-12)
        return np.sqrt(var)

    def compute_feature_std_for_indices(
        self, indices: Sequence[int], chunk_size: int = 2048
    ) -> np.ndarray:
        """Compute per-feature std for a subset of sample indices."""
        if not indices:
            return np.ones((self.num_features,), dtype=np.float64)

        sum_x = np.zeros((self.num_features,), dtype=np.float64)
        sum_x2 = np.zeros((self.num_features,), dtype=np.float64)
        total_count = 0

        for start in range(0, len(indices), chunk_size):
            batch_idx = indices[start : start + chunk_size]
            chunk = np.asarray(self.data[batch_idx], dtype=np.float64)
            sum_x += chunk.sum(axis=(0, 1))
            sum_x2 += np.square(chunk).sum(axis=(0, 1))
            total_count += chunk.shape[0] * chunk.shape[1]

        mean = sum_x / total_count
        var = (sum_x2 / total_count) - np.square(mean)
        var = np.maximum(var, 1e-12)
        return np.sqrt(var)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        sequence_np = np.asarray(self.data[index], dtype=np.float32)
        sequence = torch.from_numpy(sequence_np.copy())
        target = sequence.clone()

        if self.apply_augmentation:
            # 1. Gaussian Noise Injection (only on continuous features)
            noise = torch.randn_like(sequence) * self.noise_sigma.view(1, -1)
            for col in self.continuous_cols:
                sequence[:, col] += noise[:, col]

            # 2. Temporal Masking (Dropout)
            if torch.rand(1).item() < self.dropout_prob:
                drop_len = torch.randint(
                    self.dropout_length[0], self.dropout_length[1] + 1, (1,)
                ).item()
                start_idx = torch.randint(
                    0, self.window_size - drop_len + 1, (1,)
                ).item()
                # Zero out only the continuous timesteps block
                for col in self.continuous_cols:
                    sequence[start_idx : start_idx + drop_len, col] = 0.0

            # 3. Cyclical Feature Preservation:
            # They are inherently preserved because we only applied noise/dropout to continuous_cols.

        station_id = None
        if self.station_ids is not None:
            station_id = torch.tensor(self.station_ids[index], dtype=torch.long)

        if self.return_target:
            if station_id is None:
                return sequence, target
            return sequence, target, station_id
        if station_id is None:
            return sequence
        return sequence, station_id


def move_to_device(
    batch: Union[Tensor, Sequence[Tensor]], device: torch.device
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Move a tensor batch (or tuple/list of tensors) to a device."""
    if isinstance(batch, Tensor):
        return batch.to(device, non_blocking=True)

    moved = [tensor.to(device, non_blocking=True) for tensor in batch]
    return tuple(moved)


class DeviceDataLoader:
    """Wrap DataLoader to move each batch to a target device."""

    def __init__(self, dataloader: DataLoader, device: torch.device) -> None:
        self.dataloader = dataloader
        self.device = device

    def __iter__(self) -> Iterator[Union[Tensor, Tuple[Tensor, ...]]]:
        for batch in self.dataloader:
            yield move_to_device(batch, self.device)

    def __len__(self) -> int:
        return len(self.dataloader)


def build_train_dataloader(
    npy_path: Union[str, Path],
    batch_size: int = 128,
    apply_augmentation: bool = True,
    jitter_scale: float = 0.01,
    dropout_prob: float = 0.1,
    mmap_mode: Optional[str] = "r",
    num_workers: int = 0,
    pin_memory: bool = True,
    device: Optional[torch.device] = None,
) -> Union[DataLoader, DeviceDataLoader]:
    """Create DataLoader for autoencoder training from ``X_train.npy``.

    If ``device`` is provided, wraps the DataLoader so each yielded batch is
    transferred from CPU to the chosen device.
    """
    dataset = NpySequenceDataset(
        npy_path=npy_path,
        apply_augmentation=apply_augmentation,
        jitter_scale=jitter_scale,
        dropout_prob=dropout_prob,
        mmap_mode=mmap_mode,
        return_target=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if device is not None:
        return DeviceDataLoader(dataloader, device)
    return dataloader
