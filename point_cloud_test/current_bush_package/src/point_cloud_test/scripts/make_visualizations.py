from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from point_cloud_test.benchmark_data import PointCloudSegmentationDataset, load_split

CLASS_COLORS = {
    0: "#b8b8b8",
    1: "#d73027",
    2: "#fdae61",
    3: "#1a9850",
}


def _set_equal_aspect(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins) / 2.0), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_metric_bars(summary: dict, output_dir: Path) -> None:
    models = list(summary.keys())
    fruit_iou = [summary[m]["fruit_macro_iou"] for m in models]
    macro_iou = [summary[m]["macro_iou"] for m in models]
    overall = [summary[m]["overall_accuracy"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, values, title in zip(
        axes,
        [fruit_iou, macro_iou, overall],
        ["Fruit Macro IoU", "Macro IoU", "Overall Accuracy"],
        strict=True,
    ):
        ax.bar(models, values, color=["#3b6fb6", "#5aa469", "#9f5f80", "#d07c3f"])
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        ax.tick_params(axis="x", rotation=25)
        for i, value in enumerate(values):
            ax.text(i, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(summary: dict, output_dir: Path) -> None:
    models = list(summary.keys())
    train_s = [summary[m]["train_seconds"] for m in models]
    infer_s = [summary[m]["inference_seconds"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, train_s, width, label="train_seconds", color="#5f0f40")
    ax.bar(x + width / 2, infer_s, width, label="inference_seconds", color="#0f4c5c")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25)
    ax.set_title("Runtime comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pointcloud_views(dataset_root: Path, prediction_root: Path, output_dir: Path, sample_name: str = "sample_0001") -> None:
    split = load_split(dataset_root)
    if sample_name not in split.test:
        sample_name = split.test[0]
    dataset = PointCloudSegmentationDataset(dataset_root, [sample_name], points_per_sample=2048)
    batch = dataset[0]
    points = batch["points"].numpy()
    labels = batch["labels"].numpy()

    pred_files = {
        "gt": labels,
        "oneformer3d": np.load(prediction_root / "oneformer3d" / f"{sample_name}.npy"),
        "pointtransformerv3": np.load(prediction_root / "pointtransformerv3" / f"{sample_name}.npy"),
        "odin": np.load(prediction_root / "odin" / f"{sample_name}.npy"),
        "open-yolo-3d": np.load(prediction_root / "open-yolo-3d" / f"{sample_name}.npy"),
    }

    max_points = min(2048, len(points))
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    xyz = points[idx, :3]

    fig = plt.figure(figsize=(18, 8))
    for subplot_idx, (name, y_pred) in enumerate(pred_files.items(), start=1):
        ax = fig.add_subplot(2, 3, subplot_idx, projection="3d")
        y_vis = y_pred[idx]
        colors = [CLASS_COLORS[int(label)] for label in y_vis]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=2, alpha=0.8)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        _set_equal_aspect(ax, xyz)
    fig.tight_layout()
    fig.savefig(output_dir / f"{sample_name}_qualitative.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    benchmark_root = Path(r".\artifacts\benchmark")
    dataset_root = Path(r".\artifacts\dataset")
    summary = load_json(benchmark_root / "benchmark_summary.json")
    plot_metric_bars(summary, benchmark_root)
    plot_runtime(summary, benchmark_root)
    plot_pointcloud_views(dataset_root, benchmark_root / "predictions", benchmark_root)
    print(f"Saved visualizations to {benchmark_root}")


if __name__ == "__main__":
    main()
