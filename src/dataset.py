"""
Dataset loader for the KLA semiconductor-inspection restoration task.

Pairs on disk (confirmed by inspecting the actual delivered dataset):
    train/GT/<id>.npy        -> float32, HxW = 256x256, range ~[0, 1]
    train/NoisyLR/<id>.npy   -> float32, HxW = 128x128, range can exceed
                                 [0, 1] (speckle noise pushes pixels past
                                 the true signal range -- this is expected,
                                 per the problem statement, so we do NOT
                                 clip the input before feeding the model).

The held-out folder at the repo root (`NoisyLR/`, no matching GT) mirrors
KLA's blind test set: same 128x128 unclipped format, no ground truth.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class RestorationDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, train=True, extra_noise_std=0.03, crop_size=None):
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
        self.noisy_dir = noisy_dir
        self.train = train
        self.extra_noise_std = extra_noise_std
        self.crop_size = crop_size

        ids = [os.path.basename(p) for p in self.gt_paths]
        missing = [i for i in ids if not os.path.exists(os.path.join(noisy_dir, i))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} GT files have no matching NoisyLR file, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.gt_paths)

    def _augment_extra_degradation(self, noisy_lr):
        """Domain-randomization on top of the provided degradation, so the
        model doesn't just memorize KLA's fixed noise recipe -- this is
        what the OOD generalization requirement in the problem statement
        is actually testing for."""
        if not self.train:
            return noisy_lr

        # Randomly strengthen speckle noise (multiplicative) at varying levels.
        if random.random() < 0.5:
            speckle_std = random.uniform(0.0, self.extra_noise_std)
            noisy_lr = noisy_lr * (1.0 + np.random.randn(*noisy_lr.shape).astype(np.float32) * speckle_std)

        # Randomly add extra Gaussian noise (additive) at varying levels.
        if random.random() < 0.5:
            gauss_std = random.uniform(0.0, self.extra_noise_std)
            noisy_lr = noisy_lr + np.random.randn(*noisy_lr.shape).astype(np.float32) * gauss_std

        return noisy_lr

    def _random_crop(self, gt, noisy_lr):
        """Crop_size was accepted but never applied -- random spatial crops
        are cheap augmentation that multiplies effective training diversity
        on a fixed 2880-image set, without touching val/eval (train-only)."""
        if not self.train or self.crop_size is None:
            return gt, noisy_lr

        h, w = noisy_lr.shape
        cs = self.crop_size
        if cs >= h or cs >= w:
            return gt, noisy_lr

        top = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        noisy_crop = noisy_lr[top:top + cs, left:left + cs]
        gt_crop = gt[top * 2:(top + cs) * 2, left * 2:(left + cs) * 2]
        return gt_crop, noisy_crop

    def _geometric_augment(self, gt, noisy_lr):
        if not self.train:
            return gt, noisy_lr

        if random.random() < 0.5:
            gt, noisy_lr = np.fliplr(gt).copy(), np.fliplr(noisy_lr).copy()
        if random.random() < 0.5:
            gt, noisy_lr = np.flipud(gt).copy(), np.flipud(noisy_lr).copy()
        k = random.randint(0, 3)
        if k:
            gt, noisy_lr = np.rot90(gt, k).copy(), np.rot90(noisy_lr, k).copy()

        return gt, noisy_lr

    def __getitem__(self, idx):
        gt_path = self.gt_paths[idx]
        fname = os.path.basename(gt_path)
        noisy_path = os.path.join(self.noisy_dir, fname)

        gt = np.load(gt_path).astype(np.float32)
        noisy_lr = np.load(noisy_path).astype(np.float32)

        gt, noisy_lr = self._random_crop(gt, noisy_lr)
        gt, noisy_lr = self._geometric_augment(gt, noisy_lr)
        noisy_lr = self._augment_extra_degradation(noisy_lr)

        gt_t = torch.from_numpy(gt).unsqueeze(0)          # 1 x 256 x 256
        noisy_t = torch.from_numpy(noisy_lr).unsqueeze(0)  # 1 x 128 x 128

        return {"noisy_lr": noisy_t, "gt": gt_t, "filename": fname}


def make_splits(gt_dir, noisy_dir, val_fraction=0.1, seed=42):
    """Deterministic train/val split by filename, so the val set stays
    fixed across runs and doesn't leak into training."""
    all_ids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(gt_dir, "*.npy")))
    rng = random.Random(seed)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_ids = set(shuffled[:n_val])
    train_ids = [i for i in all_ids if i not in val_ids]
    val_ids = [i for i in all_ids if i in val_ids]
    return train_ids, val_ids


class SubsetRestorationDataset(RestorationDataset):
    """Same as RestorationDataset but restricted to a given filename list."""

    def __init__(self, gt_dir, noisy_dir, filenames, train=True, extra_noise_std=0.03, crop_size=None):
        super().__init__(gt_dir, noisy_dir, train=train, extra_noise_std=extra_noise_std, crop_size=crop_size)
        keep = set(filenames)
        self.gt_paths = [p for p in self.gt_paths if os.path.basename(p) in keep]
