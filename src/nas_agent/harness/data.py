"""
data.py
-------
CIFAR-10 dataset utilities for the NAS-Agent fixed training harness.

Requirements & Design (Blueprint Section 6.2 & Study Guide):
1. Fixed Stratified Split:
   - Macro proxy: 10,000 images (stratified: exactly 1,000 images per class).
   - Calibration reference: 20,000 images (2,000 per class).
   - Validation set: fixed 5,000 or 10,000 images (stratified across classes).
2. GPU-Resident Tensors:
   - The entire training subset and validation set are loaded directly into device memory
     (GPU VRAM if CUDA is available, CPU RAM otherwise) as contiguous tensors.
   - Storing uint8 tensors on device requires only ~30.7 MB for 10k images (3x32x32).
     Batch conversion and normalization happen on-the-fly on GPU.
3. GPU-Side Augmentation (Zero DataLoader Workers):
   - Augmentation is performed vectorized on the device:
     - Random horizontal flip (p=0.5).
     - Random crop 32x32 with padding 4 (reflection or replicate padding).
     - Normalization with CIFAR-10 channel statistics.
   - No Python multiprocessing, no DataLoader workers, no IPC bottlenecks.
4. Determinism & Offline / Mock Mode:
   - Stratified sampling is deterministically seeded.
   - Offline / mock mode generates synthetic balanced tensors for rapid testing and CI.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# CIFAR-10 channel-wise statistics (RGB)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10


@dataclass
class DatasetSplit:
    """Device-resident dataset split containing images and integer labels."""
    images: torch.Tensor  # (N, 3, 32, 32) uint8 or float32 on device
    labels: torch.Tensor  # (N,) int64 on device

    def __len__(self) -> int:
        return self.images.size(0)

    @property
    def device(self) -> torch.device:
        return self.images.device


def apply_gpu_augmentation(
    x: torch.Tensor,
    padding: int = 4,
    hflip_prob: float = 0.5,
) -> torch.Tensor:
    """Vectorized on-device data augmentation for CIFAR-10.

    Parameters
    ----------
    x : torch.Tensor
        Batch of images of shape (B, 3, 32, 32), float32.
    padding : int
        Padding size for random crop (default 4).
    hflip_prob : float
        Probability of random horizontal flip (default 0.5).

    Returns
    -------
    torch.Tensor
        Augmented batch of images of shape (B, 3, 32, 32).
    """
    B, C, H, W = x.shape
    device = x.device

    # 1. Random horizontal flip
    if hflip_prob > 0.0:
        flip_mask = torch.rand(B, device=device) < hflip_prob
        if flip_mask.any():
            x = x.clone()
            x[flip_mask] = x[flip_mask].flip(-1)

    # 2. Random crop with reflection padding
    if padding > 0:
        # Pad to (B, C, H + 2*pad, W + 2*pad)
        padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
        pad_h, pad_w = H + 2 * padding, W + 2 * padding

        # Random offsets in [0, 2*padding] for each batch element
        offsets_y = torch.randint(0, 2 * padding + 1, (B,), device=device)
        offsets_x = torch.randint(0, 2 * padding + 1, (B,), device=device)

        # Batch slicing without Python loop using grid_sample or gather
        # For simplicity and maximum efficiency on small batches:
        # Create standard sampling grids or slice
        crops = []
        for b in range(B):
            oy = offsets_y[b].item()
            ox = offsets_x[b].item()
            crops.append(padded[b:b+1, :, oy:oy + H, ox:ox + W])
        x = torch.cat(crops, dim=0)

    return x


class FastBatchLoader:
    """High-performance in-memory batch iterator for device-resident tensors.

    Features:
    - Zero worker processes, zero IPC copying.
    - Deterministic per-epoch shuffling with a reproducible seed.
    - GPU-side uint8 -> float32 conversion, augmentation, and normalization.
    """

    def __init__(
        self,
        split: DatasetSplit,
        batch_size: int = 128,
        shuffle: bool = True,
        augment: bool = False,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed
        self.drop_last = drop_last
        self.num_samples = len(split)

        device = split.device
        self._mean = torch.tensor(CIFAR10_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(CIFAR10_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator(device=device if device.type == "cpu" else "cpu")
            self._generator.manual_seed(seed)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        device = self.split.device
        if self.shuffle:
            if self._generator is not None:
                perm = torch.randperm(self.num_samples, generator=self._generator)
                if device.type != "cpu":
                    perm = perm.to(device)
            else:
                perm = torch.randperm(self.num_samples, device=device)
        else:
            perm = torch.arange(self.num_samples, device=device)

        n_batches = len(self)
        for i in range(n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and (end - start) < self.batch_size:
                break

            idx = perm[start:end]
            bx = self.split.images[idx]
            by = self.split.labels[idx]

            # Convert to float in [0, 1] if uint8
            if bx.dtype == torch.uint8:
                bx = bx.to(dtype=torch.float32).div_(255.0)
            elif bx.dtype != torch.float32:
                bx = bx.to(dtype=torch.float32)

            # Apply training augmentation
            if self.augment:
                bx = apply_gpu_augmentation(bx)

            # Standardize with CIFAR-10 statistics
            bx = (bx - self._mean) / self._std

            yield bx, by


def create_mock_data(
    num_train: int = 1000,
    num_val: int = 500,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[DatasetSplit, DatasetSplit]:
    """Create synthetic balanced CIFAR-like dataset tensors for testing/CI."""
    rng = torch.Generator().manual_seed(seed)
    
    # Stratified labels
    train_labels = torch.arange(num_train) % NUM_CLASSES
    train_labels = train_labels[torch.randperm(num_train, generator=rng)]
    train_images = torch.randint(0, 256, (num_train, 3, 32, 32), dtype=torch.uint8, generator=rng)

    val_labels = torch.arange(num_val) % NUM_CLASSES
    val_labels = val_labels[torch.randperm(num_val, generator=rng)]
    val_images = torch.randint(0, 256, (num_val, 3, 32, 32), dtype=torch.uint8, generator=rng)

    target_dev = torch.device(device)
    train_split = DatasetSplit(images=train_images.to(target_dev), labels=train_labels.to(target_dev))
    val_split = DatasetSplit(images=val_images.to(target_dev), labels=val_labels.to(target_dev))

    return train_split, val_split


def get_stratified_subset_indices(
    labels: np.ndarray,
    subset_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate deterministic stratified indices with equal samples per class."""
    rng = np.random.RandomState(seed)
    samples_per_class = subset_size // NUM_CLASSES
    selected_indices = []

    for c in range(NUM_CLASSES):
        cls_indices = np.where(labels == c)[0]
        if len(cls_indices) < samples_per_class:
            raise ValueError(
                f"Class {c} only has {len(cls_indices)} samples, but {samples_per_class} requested"
            )
        chosen = rng.choice(cls_indices, size=samples_per_class, replace=False)
        selected_indices.extend(chosen)

    rng.shuffle(selected_indices)
    return np.array(selected_indices, dtype=np.int64)


def get_cifar10_data(
    data_dir: str = "data",
    subset_size: int = 10000,
    val_size: int = 5000,
    device: str = "cuda",
    seed: int = 42,
    mock: bool = False,
) -> Tuple[DatasetSplit, DatasetSplit]:
    """Load stratified CIFAR-10 data onto the specified device.

    Parameters
    ----------
    data_dir : str
        Directory to cache downloaded/preprocessed CIFAR-10 dataset files.
    subset_size : int
        Training subset size (default 10,000 for macro proxy; 20,000 for calibration).
    val_size : int
        Validation set size (default 5,000 from test split).
    device : str
        Target device ('cuda' or 'cpu'). Falls back to 'cpu' if CUDA is unavailable.
    seed : int
        Random seed for deterministic stratified split.
    mock : bool
        If True or env NAS_MOCK_DATA is set, generate synthetic data instantly.

    Returns
    -------
    Tuple[DatasetSplit, DatasetSplit]
        (train_split, val_split) resident on the requested device.
    """
    if mock or os.getenv("NAS_MOCK_DATA") == "1":
        actual_device = "cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        return create_mock_data(num_train=subset_size, num_val=val_size, device=actual_device, seed=seed)

    actual_device = "cuda" if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.makedirs(data_dir, exist_ok=True)

    cache_train_path = os.path.join(data_dir, f"cifar10_train_strat_{subset_size}_s{seed}.pt")
    cache_val_path = os.path.join(data_dir, f"cifar10_val_strat_{val_size}_s{seed}.pt")

    if os.path.exists(cache_train_path) and os.path.exists(cache_val_path):
        train_dict = torch.load(cache_train_path, map_location=actual_device, weights_only=True)
        val_dict = torch.load(cache_val_path, map_location=actual_device, weights_only=True)
        return (
            DatasetSplit(images=train_dict["images"], labels=train_dict["labels"]),
            DatasetSplit(images=val_dict["images"], labels=val_dict["labels"]),
        )

    # Download or load via torchvision
    try:
        import torchvision
        import torchvision.datasets as datasets
    except ImportError as e:
        raise ImportError("torchvision is required to download/prepare CIFAR-10") from e

    raw_train = datasets.CIFAR10(root=data_dir, train=True, download=True)
    raw_test = datasets.CIFAR10(root=data_dir, train=False, download=True)

    # Convert raw uint8 arrays (N, 32, 32, 3) -> (N, 3, 32, 32)
    train_imgs_all = torch.from_numpy(raw_train.data.transpose(0, 3, 1, 2))
    train_labels_all = np.array(raw_train.targets)

    test_imgs_all = torch.from_numpy(raw_test.data.transpose(0, 3, 1, 2))
    test_labels_all = np.array(raw_test.targets)

    # Stratified selection
    train_idx = get_stratified_subset_indices(train_labels_all, subset_size, seed=seed)
    train_imgs = train_imgs_all[train_idx]
    train_lbls = torch.from_numpy(train_labels_all[train_idx]).to(torch.long)

    val_idx = get_stratified_subset_indices(test_labels_all, val_size, seed=seed)
    val_imgs = test_imgs_all[val_idx]
    val_lbls = torch.from_numpy(test_labels_all[val_idx]).to(torch.long)

    # Save cache for fast subsequent loads
    torch.save({"images": train_imgs, "labels": train_lbls}, cache_train_path)
    torch.save({"images": val_imgs, "labels": val_lbls}, cache_val_path)

    # Move to target device
    train_split = DatasetSplit(images=train_imgs.to(actual_device), labels=train_lbls.to(actual_device))
    val_split = DatasetSplit(images=val_imgs.to(actual_device), labels=val_lbls.to(actual_device))

    return train_split, val_split
