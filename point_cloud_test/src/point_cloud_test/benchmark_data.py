from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class DatasetSplit:
    train: list[str]
    val: list[str]
    test: list[str]


def load_manifest(dataset_root: Path) -> list[dict]:
    return json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))


def create_split(dataset_root: Path, seed: int = 42, train_ratio: float = 0.6, val_ratio: float = 0.2) -> DatasetSplit:
    sample_names = sorted(path.stem for path in dataset_root.glob("sample_*.npz"))
    rng = np.random.default_rng(seed)
    rng.shuffle(sample_names)
    n_total = len(sample_names)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_test = max(1, n_total - n_train - n_val)
    if n_train + n_val + n_test > n_total:
        n_test = n_total - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)
    return DatasetSplit(
        train=sample_names[:n_train],
        val=sample_names[n_train : n_train + n_val],
        test=sample_names[n_train + n_val : n_train + n_val + n_test],
    )


def save_split(dataset_root: Path, split: DatasetSplit) -> Path:
    split_path = dataset_root / "splits.json"
    split_path.write_text(
        json.dumps({"train": split.train, "val": split.val, "test": split.test}, indent=2),
        encoding="utf-8",
    )
    return split_path


def load_split(dataset_root: Path) -> DatasetSplit:
    payload = json.loads((dataset_root / "splits.json").read_text(encoding="utf-8"))
    return DatasetSplit(train=payload["train"], val=payload["val"], test=payload["test"])


def load_sample(dataset_root: Path, sample_name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(dataset_root / f"{sample_name}.npz")
    return data["points_xyzrgb"].astype(np.float32), data["class_id"].astype(np.int64)


class PointCloudSegmentationDataset(Dataset):
    def __init__(self, dataset_root: Path, sample_names: list[str], points_per_sample: int = 2048) -> None:
        self.dataset_root = dataset_root
        self.sample_names = sample_names
        self.points_per_sample = points_per_sample

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample_name = self.sample_names[index]
        points, labels = load_sample(self.dataset_root, sample_name)
        rng = np.random.default_rng(index)
        choice = _stratified_choice(labels, self.points_per_sample, rng)
        points = points[choice]
        labels = labels[choice]

        xyz = points[:, :3]
        rgb = points[:, 3:]
        xyz = xyz - xyz.mean(axis=0, keepdims=True)
        scale = np.maximum(np.linalg.norm(xyz, axis=1).max(), 1e-6)
        xyz = xyz / scale
        features = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        return {
            "points": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
        }


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "points": torch.stack([item["points"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
    }


def _stratified_choice(labels: np.ndarray, points_per_sample: int, rng: np.random.Generator) -> np.ndarray:
    all_indices = np.arange(len(labels))
    foreground = all_indices[labels > 0]
    background = all_indices[labels == 0]
    if len(all_indices) <= points_per_sample:
        return rng.choice(all_indices, size=points_per_sample, replace=True)

    fg_target = min(len(foreground), max(points_per_sample // 2, 1))
    bg_target = points_per_sample - fg_target
    choices = []
    if fg_target > 0 and len(foreground) > 0:
        choices.append(rng.choice(foreground, size=fg_target, replace=len(foreground) < fg_target))
    if bg_target > 0 and len(background) > 0:
        choices.append(rng.choice(background, size=bg_target, replace=len(background) < bg_target))
    if not choices:
        return rng.choice(all_indices, size=points_per_sample, replace=False)
    merged = np.concatenate(choices)
    rng.shuffle(merged)
    return merged
