from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from point_cloud_test.geometry import load_rgb, load_frame_metadata, project_world_to_image


CLASS_COLORS_BGR = {
    0: (180, 180, 180),
    1: (40, 40, 220),
    2: (50, 180, 240),
    3: (50, 170, 50),
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


def draw_overlay(sample_name: str = "sample_0001") -> None:
    dataset_root = Path(r".\artifacts\dataset")
    orbit_root = Path(r"..\Perception_In_Robotics\FP\strawberry_orbit_data_new\data")
    proof_root = Path(r".\artifacts\proof")
    proof_root.mkdir(parents=True, exist_ok=True)

    sample_meta = json.loads((dataset_root / f"{sample_name}.json").read_text(encoding="utf-8"))
    sample = np.load(dataset_root / f"{sample_name}.npz")
    points_world = sample["points_xyzrgb"][:, :3]
    class_id = sample["class_id"]

    for frame_id in sample_meta["frame_group"]:
        frame_meta = load_frame_metadata(orbit_root / "coords" / f"frame_{frame_id:04d}.json")
        rgb = load_rgb(orbit_root / "images" / f"rgb_{frame_id:04d}.png")
        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pixels, valid = project_world_to_image(
            points_world,
            frame_meta["camera_gt"]["rotation_matrix_3x3"],
            frame_meta["camera_gt"]["position_world_m"],
            frame_meta["intrinsics_px"],
        )
        h, w = rgb.shape[:2]
        pixels = pixels[valid]
        labels = class_id[valid]
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < w)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < h)
        )
        pixels = pixels[inside].astype(np.int32)
        labels = labels[inside]
        for (u, v), label in zip(pixels, labels, strict=True):
            color = CLASS_COLORS_BGR[int(label)]
            cv2.circle(overlay, (int(u), int(v)), 1, color, thickness=-1)
        output_path = proof_root / f"{sample_name}_frame_{frame_id:04d}_overlay.png"
        cv2.imwrite(str(output_path), overlay)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    rgb_colors = sample["points_xyzrgb"][:, 3:]
    idx = np.linspace(0, len(points_world) - 1, min(8000, len(points_world))).astype(int)
    ax.scatter(
        points_world[idx, 0],
        points_world[idx, 1],
        points_world[idx, 2],
        c=rgb_colors[idx],
        s=2,
        alpha=0.8,
    )
    ax.set_title(f"{sample_name} raw RGB-colored cloud")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    _set_equal_aspect(ax, points_world[idx])
    fig.tight_layout()
    fig.savefig(proof_root / f"{sample_name}_rgb_cloud.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_overlay()
