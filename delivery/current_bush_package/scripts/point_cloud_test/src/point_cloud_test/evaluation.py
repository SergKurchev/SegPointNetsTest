from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_NAMES = ["background", "ripe", "half-ripe", "unripe"]


@dataclass(slots=True)
class MetricsSummary:
    overall_accuracy: float
    macro_iou: float
    macro_f1: float
    per_class_iou: dict[str, float]
    per_class_f1: dict[str, float]


def compute_metrics(target: np.ndarray, prediction: np.ndarray, class_names: list[str] | None = None) -> MetricsSummary:
    if class_names is None:
        class_names = CLASS_NAMES

    target = target.astype(np.int64)
    prediction = prediction.astype(np.int64)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have the same shape")

    overall_accuracy = float((target == prediction).mean())
    ious: dict[str, float] = {}
    f1s: dict[str, float] = {}
    for class_id, class_name in enumerate(class_names):
        tp = np.logical_and(target == class_id, prediction == class_id).sum()
        fp = np.logical_and(target != class_id, prediction == class_id).sum()
        fn = np.logical_and(target == class_id, prediction != class_id).sum()
        denom_iou = tp + fp + fn
        denom_f1 = 2 * tp + fp + fn
        ious[class_name] = float(tp / denom_iou) if denom_iou else 0.0
        f1s[class_name] = float((2 * tp) / denom_f1) if denom_f1 else 0.0

    return MetricsSummary(
        overall_accuracy=overall_accuracy,
        macro_iou=float(np.mean(list(ious.values()))),
        macro_f1=float(np.mean(list(f1s.values()))),
        per_class_iou=ious,
        per_class_f1=f1s,
    )


def save_summary(path: Path, summaries: dict[str, MetricsSummary]) -> None:
    serializable = {
        name: {
            "overall_accuracy": metrics.overall_accuracy,
            "macro_iou": metrics.macro_iou,
            "macro_f1": metrics.macro_f1,
            "per_class_iou": metrics.per_class_iou,
            "per_class_f1": metrics.per_class_f1,
        }
        for name, metrics in summaries.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
