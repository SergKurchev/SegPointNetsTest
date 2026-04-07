from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _point_cloud_root() -> Path:
    return _workspace_root() / "point_cloud_test"


def _insert_point_cloud_src() -> None:
    src_root = _point_cloud_root() / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_insert_point_cloud_src()

from point_cloud_test.evaluation import CLASS_NAMES, compute_metrics  # noqa: E402


def _load_sample(dataset_root: Path, sample_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(dataset_root / f"{sample_name}.npz")
    return payload["points_xyzrgb"].astype(np.float32), payload["class_id"].astype(np.int64), payload["segment_id"].astype(np.int32)


def _majority_class_per_segment(class_ids: np.ndarray, segment_ids: np.ndarray) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for segment_id in sorted(int(value) for value in np.unique(segment_ids) if value >= 0):
        mask = segment_ids == segment_id
        values, counts = np.unique(class_ids[mask], return_counts=True)
        lookup[segment_id] = int(values[np.argmax(counts)])
    return lookup


def _segment_recall50_proxy(gt_class_ids: np.ndarray, gt_segment_ids: np.ndarray, pred_class_ids: np.ndarray) -> dict[str, float]:
    gt_lookup = _majority_class_per_segment(gt_class_ids, gt_segment_ids)
    gt_ids_by_class: dict[int, list[int]] = defaultdict(list)
    for segment_id, class_id in gt_lookup.items():
        if class_id > 0:
            gt_ids_by_class[class_id].append(segment_id)

    metrics: dict[str, float] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        if class_id == 0:
            continue
        gt_segments = gt_ids_by_class.get(class_id, [])
        if not gt_segments:
            metrics[f"segment_recall50_proxy_{class_name}"] = 0.0
            continue

        matched = 0
        used_gt: set[int] = set()
        pred_foreground = pred_class_ids == class_id
        for gt_segment_id in gt_segments:
            gt_mask = gt_segment_ids == gt_segment_id
            intersection = np.logical_and(gt_mask, pred_foreground).sum()
            union = np.logical_or(gt_mask, pred_foreground).sum()
            iou = float(intersection / union) if union else 0.0
            if iou >= 0.5 and gt_segment_id not in used_gt:
                matched += 1
                used_gt.add(gt_segment_id)
        metrics[f"segment_recall50_proxy_{class_name}"] = float(matched / len(gt_segments))
    thing_scores = [value for key, value in metrics.items() if key.startswith("segment_recall50_proxy_")]
    metrics["segment_recall50_proxy_things"] = float(np.mean(thing_scores)) if thing_scores else 0.0
    return metrics


def evaluate_prediction_dir(dataset_root: Path, predictions_root: Path, sample_names: list[str]) -> dict[str, object]:
    all_target: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    per_sample: dict[str, dict[str, object]] = {}
    instance_scores: list[dict[str, float]] = []

    for sample_name in sample_names:
        points, target, segment_ids = _load_sample(dataset_root, sample_name)
        prediction_path = predictions_root / f"{sample_name}.npy"
        if not prediction_path.exists():
            continue
        prediction = np.load(prediction_path).astype(np.int64).reshape(-1)
        if prediction.shape[0] != target.shape[0]:
            raise ValueError(
                f"Prediction length mismatch for {sample_name}: got {prediction.shape[0]}, expected {target.shape[0]}"
            )

        metrics = compute_metrics(target, prediction)
        ap50 = _segment_recall50_proxy(target, segment_ids, prediction)
        per_sample[sample_name] = {
            "num_points": int(len(points)),
            "overall_accuracy": metrics.overall_accuracy,
            "macro_iou": metrics.macro_iou,
            "macro_f1": metrics.macro_f1,
            "per_class_iou": metrics.per_class_iou,
            "per_class_f1": metrics.per_class_f1,
            **ap50,
        }
        all_target.append(target)
        all_pred.append(prediction)
        instance_scores.append(ap50)

    if not all_target:
        raise FileNotFoundError(f"No prediction files found in {predictions_root}")

    stacked_target = np.concatenate(all_target, axis=0)
    stacked_pred = np.concatenate(all_pred, axis=0)
    metrics = compute_metrics(stacked_target, stacked_pred)

    averaged_ap50: dict[str, float] = {}
    ap50_keys = sorted(instance_scores[0].keys())
    for key in ap50_keys:
        averaged_ap50[key] = float(np.mean([payload[key] for payload in instance_scores]))

    fruit_classes = ["ripe", "half-ripe", "unripe"]
    fruit_macro_iou = float(np.mean([metrics.per_class_iou[name] for name in fruit_classes]))
    fruit_macro_f1 = float(np.mean([metrics.per_class_f1[name] for name in fruit_classes]))
    return {
        "overall_accuracy": metrics.overall_accuracy,
        "macro_iou": metrics.macro_iou,
        "macro_f1": metrics.macro_f1,
        "fruit_macro_iou": fruit_macro_iou,
        "fruit_macro_f1": fruit_macro_f1,
        "per_class_iou": metrics.per_class_iou,
        "per_class_f1": metrics.per_class_f1,
        **averaged_ap50,
        "num_evaluated_samples": len(per_sample),
        "per_sample": per_sample,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate point-level strawberry predictions from official model adapters.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_point_cloud_root() / "artifacts" / "dataset",
        help="Ground-truth sample dataset root.",
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        required=True,
        help="Directory with <model_name>/<sample_name>.npy point-class predictions.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["pointtransformerv3", "oneformer3d", "odin", "openyolo3d"],
        help="Model subfolders to evaluate.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which sample split to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_payload = json.loads((args.dataset_root / "splits.json").read_text(encoding="utf-8"))
    if args.split == "all":
        sample_names = sorted({name for values in split_payload.values() for name in values})
    else:
        sample_names = split_payload[args.split]

    summary: dict[str, dict[str, object]] = {}
    for model_name in args.models:
        summary[model_name] = evaluate_prediction_dir(args.dataset_root, args.predictions_root / model_name, sample_names)

    text = json.dumps(summary, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
