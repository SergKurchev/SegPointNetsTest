from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from point_cloud_test.point_cloud_ops import voxel_downsample


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


def _style_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
    except AttributeError:
        pass


def render_bush_views(
    dataset_root: Path,
    output_path: Path,
    max_samples: int | None = None,
    voxel_size_m: float = 0.01,
) -> Path:
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if max_samples is not None:
        manifest = manifest[:max_samples]

    clouds: list[np.ndarray] = []
    for sample_meta in manifest:
        sample_name = sample_meta["sample_name"]
        sample_path = dataset_root / f"{sample_name}.npz"
        if sample_path.exists():
            sample = np.load(sample_path)
            clouds.append(sample["points_xyzrgb"].astype(np.float32))

    if not clouds:
        raise FileNotFoundError(f"No point clouds found under {dataset_root}")

    merged = voxel_downsample(np.concatenate(clouds, axis=0), voxel_size_m=voxel_size_m)
    xyz = merged[:, :3]
    rgb = np.clip(merged[:, 3:], 0.0, 1.0)

    fig = plt.figure(figsize=(15, 5))
    views = [
        ("Front", 20, -85),
        ("Side", 18, 5),
        ("Top-Oblique", 65, -55),
    ]
    for subplot_idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, subplot_idx, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=3.0, alpha=0.95, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        _set_equal_aspect(ax, xyz)
        _style_axes(ax)
        ax.set_title(title)

    fig.suptitle("Merged bush point cloud from all samples", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    dataset_root = Path(r"D:\MyProjects\Robostrawberry\point_cloud_test\artifacts\dataset")
    output_path = Path(r"D:\MyProjects\Robostrawberry\point_cloud_test\artifacts\proof\merged_bush_3_views.png")
    result = render_bush_views(dataset_root=dataset_root, output_path=output_path)
    print(f"Saved {result}")


if __name__ == "__main__":
    main()
