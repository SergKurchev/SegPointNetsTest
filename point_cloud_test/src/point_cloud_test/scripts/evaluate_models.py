from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from point_cloud_test.evaluation import compute_metrics, save_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=None,
        help="Folder with <model_name>/<sample_name>.npy class predictions.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["pointnet", "pointnet2", "dgcnn", "randlanet"],
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    predictions_root = (
        args.predictions_root.resolve()
        if args.predictions_root is not None
        else (dataset_root.parent / "predictions").resolve()
    )
    sample_paths = sorted(dataset_root.glob("sample_*.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"No dataset samples found in {dataset_root}")

    summaries = {}
    for model_name in args.models:
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []
        model_dir = predictions_root / model_name
        for sample_path in sample_paths:
            sample = np.load(sample_path)
            y_true = sample["class_id"]
            prediction_path = model_dir / f"{sample_path.stem}.npy"
            if not prediction_path.exists():
                continue
            y_pred = np.load(prediction_path)
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

        if not y_true_all:
            continue
        metrics = compute_metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))
        summaries[model_name] = metrics

    summary_path = dataset_root.parent / "evaluation_summary.json"
    save_summary(summary_path, summaries)
    print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
