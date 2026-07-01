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
