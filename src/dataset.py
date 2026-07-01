from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CaseRecord:
    caseid: str
    noisy_path: Path
    clean_path: Path


def read_cases(csv_path: Path, data_root: Path, max_cases: Optional[int] = None) -> List[CaseRecord]:
    df = pd.read_csv(csv_path, dtype={"caseid": str})
    required = {"caseid", "noisy_path", "clean_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    if max_cases is not None:
        df = df.head(max_cases)

    records: List[CaseRecord] = []
    for row in df.itertuples(index=False):
        noisy_path = Path(row.noisy_path)
        clean_path = Path(row.clean_path)
        if not noisy_path.is_absolute():
            noisy_path = data_root / noisy_path
        if not clean_path.is_absolute():
            clean_path = data_root / clean_path
        records.append(CaseRecord(str(row.caseid), noisy_path, clean_path))
    return records


def _load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {arr.shape} for {path}")
    return arr


def nifti_shape(path: Path) -> Tuple[int, int, int]:
    shape = nib.load(str(path)).shape
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {shape} for {path}")
    return int(shape[0]), int(shape[1]), int(shape[2])


@lru_cache(maxsize=64)
def load_pair(noisy_path: str, clean_path: str) -> Tuple[np.ndarray, np.ndarray]:
    noisy = _load_nifti(Path(noisy_path))
    clean = _load_nifti(Path(clean_path))
    if noisy.shape != clean.shape:
        raise ValueError(f"Shape mismatch: {noisy_path} {noisy.shape} vs {clean_path} {clean.shape}")

    foreground = (np.abs(clean) > 1e-6) | (np.abs(noisy) > 1e-6)
    values = np.concatenate([noisy[foreground], clean[foreground]]) if foreground.any() else np.concatenate([noisy.ravel(), clean.ravel()])
    p1, p99 = np.percentile(values, [1, 99])
    scale = max(float(p99 - p1), 1e-6)
    noisy = np.clip((noisy - p1) / scale, 0.0, 1.0).astype(np.float32)
    clean = np.clip((clean - p1) / scale, 0.0, 1.0).astype(np.float32)
    return noisy, clean


def valid_slice_indices(clean: np.ndarray, min_foreground_ratio: float) -> List[int]:
    indices: List[int] = []
    for idx in range(clean.shape[2]):
        mask = clean[:, :, idx] > 1e-4
        if float(mask.mean()) >= min_foreground_ratio:
            indices.append(idx)
    if not indices:
        indices = list(range(clean.shape[2]))
    return indices


class SliceDenoiseDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CaseRecord],
        target_size: int = 256,
        min_foreground_ratio: float = 0.01,
        max_slices_per_case: Optional[int] = None,
    ) -> None:
        self.records = list(records)
        self.target_size = target_size
        self.samples: List[Tuple[int, int]] = []

        for case_idx, record in enumerate(self.records):
            noisy, clean = load_pair(str(record.noisy_path), str(record.clean_path))
            indices = valid_slice_indices(clean, min_foreground_ratio)
            if max_slices_per_case is not None and len(indices) > max_slices_per_case:
                positions = np.linspace(0, len(indices) - 1, max_slices_per_case).round().astype(int)
                indices = [indices[int(pos)] for pos in positions]
            self.samples.extend((case_idx, slice_idx) for slice_idx in indices)

        if not self.samples:
            raise ValueError("No valid slices found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_idx, slice_idx = self.samples[index]
        record = self.records[case_idx]
        noisy, clean = load_pair(str(record.noisy_path), str(record.clean_path))

        noisy_slice = torch.from_numpy(noisy[:, :, slice_idx]).float().unsqueeze(0).unsqueeze(0)
        clean_slice = torch.from_numpy(clean[:, :, slice_idx]).float().unsqueeze(0).unsqueeze(0)
        noisy_slice = F.interpolate(noisy_slice, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        clean_slice = F.interpolate(clean_slice, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return noisy_slice.squeeze(0), clean_slice.squeeze(0)


class Slice25DDenoiseDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CaseRecord],
        target_size: int = 256,
        context: int = 1,
        min_foreground_ratio: float = 0.01,
        max_slices_per_case: Optional[int] = None,
    ) -> None:
        self.records = list(records)
        self.target_size = target_size
        self.context = context
        self.samples: List[Tuple[int, int]] = []

        for case_idx, record in enumerate(self.records):
            shape = nifti_shape(record.clean_path)
            margin = max(1, int(round(shape[2] * 0.08)))
            indices = list(range(margin, max(margin + 1, shape[2] - margin)))
            if max_slices_per_case is not None and len(indices) > max_slices_per_case:
                positions = np.linspace(0, len(indices) - 1, max_slices_per_case).round().astype(int)
                indices = [indices[int(pos)] for pos in positions]
            self.samples.extend((case_idx, slice_idx) for slice_idx in indices)

        if not self.samples:
            raise ValueError("No valid slices found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_idx, slice_idx = self.samples[index]
        record = self.records[case_idx]
        noisy, clean = load_pair(str(record.noisy_path), str(record.clean_path))
        depth = noisy.shape[2]
        stack = []
        for offset in range(-self.context, self.context + 1):
            idx = min(max(slice_idx + offset, 0), depth - 1)
            stack.append(noisy[:, :, idx])
        noisy_stack = torch.from_numpy(np.stack(stack, axis=0)).float().unsqueeze(0)
        clean_slice = torch.from_numpy(clean[:, :, slice_idx]).float().unsqueeze(0).unsqueeze(0)
        noisy_stack = F.interpolate(noisy_stack, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        clean_slice = F.interpolate(clean_slice, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        return noisy_stack.squeeze(0), clean_slice.squeeze(0)


class Patch3DDenoiseDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CaseRecord],
        patch_size: Tuple[int, int, int] = (96, 96, 32),
        patches_per_case: int = 8,
        min_foreground_ratio: float = 0.01,
    ) -> None:
        self.records = list(records)
        self.patch_size = patch_size
        self.patches_per_case = patches_per_case
        self.min_foreground_ratio = min_foreground_ratio
        self.samples: List[Tuple[int, int]] = []
        for case_idx, record in enumerate(self.records):
            nifti_shape(record.clean_path)
            self.samples.extend((case_idx, patch_idx) for patch_idx in range(patches_per_case))
        if not self.samples:
            raise ValueError("No valid 3D patches found.")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _foreground_bounds(clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        coords = np.argwhere(clean > 1e-4)
        if coords.size == 0:
            low = np.zeros(3, dtype=np.int64)
            high = np.asarray(clean.shape, dtype=np.int64)
            return low, high
        return coords.min(axis=0), coords.max(axis=0) + 1

    @staticmethod
    def _crop_or_pad(volume: np.ndarray, starts: Sequence[int], patch_size: Sequence[int]) -> np.ndarray:
        slices = []
        pads = []
        for axis, (start, size) in enumerate(zip(starts, patch_size)):
            end = start + size
            src_start = max(start, 0)
            src_end = min(end, volume.shape[axis])
            slices.append(slice(src_start, src_end))
            pads.append((max(0, -start), max(0, end - volume.shape[axis])))
        patch = volume[tuple(slices)]
        if any(before or after for before, after in pads):
            patch = np.pad(patch, pads, mode="edge")
        return patch.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_idx, patch_idx = self.samples[index]
        record = self.records[case_idx]
        noisy, clean = load_pair(str(record.noisy_path), str(record.clean_path))
        low, high = self._foreground_bounds(clean)
        patch = np.asarray(self.patch_size, dtype=np.int64)
        center_span_low = low
        center_span_high = np.maximum(high, low + 1)
        rng = np.random.default_rng(seed=case_idx * 1000003 + patch_idx)
        center = np.asarray(
            [rng.integers(center_span_low[axis], center_span_high[axis]) for axis in range(3)],
            dtype=np.int64,
        )
        starts = center - patch // 2
        noisy_patch = self._crop_or_pad(noisy, starts, patch)
        clean_patch = self._crop_or_pad(clean, starts, patch)
        # Convert from HWD to CDHW for PyTorch Conv3d.
        noisy_t = torch.from_numpy(np.transpose(noisy_patch, (2, 0, 1))).unsqueeze(0).float()
        clean_t = torch.from_numpy(np.transpose(clean_patch, (2, 0, 1))).unsqueeze(0).float()
        return noisy_t, clean_t


def build_dataset(
    split_csv: Path,
    data_root: Path,
    target_size: int = 256,
    max_cases: Optional[int] = None,
    max_slices_per_case: Optional[int] = None,
) -> SliceDenoiseDataset:
    records = read_cases(split_csv, data_root, max_cases=max_cases)
    return SliceDenoiseDataset(
        records,
        target_size=target_size,
        max_slices_per_case=max_slices_per_case,
    )


def build_25d_dataset(
    split_csv: Path,
    data_root: Path,
    target_size: int = 256,
    context: int = 1,
    max_cases: Optional[int] = None,
    max_slices_per_case: Optional[int] = None,
) -> Slice25DDenoiseDataset:
    records = read_cases(split_csv, data_root, max_cases=max_cases)
    return Slice25DDenoiseDataset(
        records,
        target_size=target_size,
        context=context,
        max_slices_per_case=max_slices_per_case,
    )


def build_3d_dataset(
    split_csv: Path,
    data_root: Path,
    patch_size: Tuple[int, int, int] = (32, 96, 96),
    patches_per_case: int = 8,
    max_cases: Optional[int] = None,
) -> Patch3DDenoiseDataset:
    records = read_cases(split_csv, data_root, max_cases=max_cases)
    patch_dhw = tuple(int(v) for v in patch_size)
    patch_hwd = (patch_dhw[1], patch_dhw[2], patch_dhw[0])
    return Patch3DDenoiseDataset(
        records,
        patch_size=patch_hwd,
        patches_per_case=patches_per_case,
    )
