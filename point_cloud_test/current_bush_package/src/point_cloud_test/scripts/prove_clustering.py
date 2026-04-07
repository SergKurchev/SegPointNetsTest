from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from point_cloud_test.track_reid import load_tracking


def _label_color(label: int) -> tuple[int, int, int]:
    palette = [
        (231, 76, 60),
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
        (243, 156, 18),
        (39, 174, 96),
    ]
    return palette[label % len(palette)]


def _draw_labels(image_rgb: np.ndarray, labels: list[dict], title: str) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for item in labels:
        x1, y1, x2, y2 = map(int, item["bbox"])
        color = _label_color(int(item["label"]))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"S{int(item['label'])}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        title,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (25, 25, 25),
        2,
        lineType=cv2.LINE_AA,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def main(sample_name: str = "sample_0001", frame_id: int | None = None) -> None:
    project_root = Path(r".")
    dataset_root = project_root / "artifacts" / "dataset"
    proof_root = project_root / "artifacts" / "proof"
    proof_root.mkdir(parents=True, exist_ok=True)

    orbit_root = Path(r"..\Perception_In_Robotics\FP\strawberry_orbit_data_new")
    sample_meta = json.loads((dataset_root / f"{sample_name}.json").read_text(encoding="utf-8"))
    if frame_id is None:
        frame_id = int(sample_meta["frame_group"][-1])
    tracking = load_tracking(orbit_root / "data" / "tracking.json")
    old_unified = pd.read_csv(orbit_root / "unified_detections.csv")
    remap = json.loads((project_root / "artifacts" / "reid" / "strawberry_remap.json").read_text(encoding="utf-8"))
    new_map = {int(k): int(v) for k, v in remap["tracking_id_to_strawberry"].items()}

    frame_key = f"frame_{frame_id:03d}"
    tracks = tracking.get(frame_key, [])
    rgb = cv2.cvtColor(
        cv2.imread(str(orbit_root / "data" / "images" / f"rgb_{frame_id:04d}.png"), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )

    old_rows = old_unified[old_unified["frame_id"] == frame_id]
    old_map = {
        int(row["tracking_id"]): int(row["strawberry_id"])
        for _, row in old_rows.iterrows()
    }
    old_labels = []
    new_labels = []
    for track in tracks:
        tracking_id = int(track["id"])
        bbox = track["bbox"]
        old_labels.append({"bbox": bbox, "label": old_map.get(tracking_id, -1)})
        new_labels.append({"bbox": bbox, "label": new_map.get(tracking_id, tracking_id)})

    old_image = _draw_labels(rgb, old_labels, "Old remap from unified_detections.csv")
    new_image = _draw_labels(rgb, new_labels, "New remap with co-visibility constraint")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(old_image)
    axes[1].imshow(new_image)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(proof_root / f"frame_{frame_id:04d}_old_vs_new_reid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sample = np.load(dataset_root / f"{sample_name}.npz")
    points = sample["points_xyzrgb"][:, :3]
    segment_id = sample["segment_id"]
    target_id = int(sample_meta["target_segment_id"])
    foreground = segment_id >= 0
    unique_segments = sorted(int(value) for value in np.unique(segment_id[foreground]))
    color_lookup = {
        segment: np.array(_label_color(index), dtype=np.float32) / 255.0
        for index, segment in enumerate(unique_segments)
    }
    colors = np.array(
        [color_lookup.get(int(seg), np.array([0.8, 0.8, 0.8], dtype=np.float32)) for seg in segment_id],
        dtype=np.float32,
    )

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    idx = np.linspace(0, len(points) - 1, min(10000, len(points))).astype(int)
    ax1.scatter(points[idx, 0], points[idx, 1], points[idx, 2], c=sample["points_xyzrgb"][idx, 3:], s=2, alpha=0.8)
    ax1.set_title("Raw RGB cloud")
    ax2.scatter(points[idx, 0], points[idx, 1], points[idx, 2], c=colors[idx], s=2, alpha=0.8)
    ax2.set_title("Segments by stable strawberry_id")
    target_mask = segment_id == target_id
    target_points = points[target_mask]
    target_colors = sample["points_xyzrgb"][target_mask, 3:]
    target_idx = np.linspace(0, len(target_points) - 1, min(4000, len(target_points))).astype(int)
    ax3.scatter(
        target_points[target_idx, 0],
        target_points[target_idx, 1],
        target_points[target_idx, 2],
        c=target_colors[target_idx],
        s=3,
        alpha=0.9,
    )
    ax3.set_title(f"Target strawberry S{target_id}")
    for ax in (ax1, ax2, ax3):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(proof_root / f"{sample_name}_cluster_proof.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "sample_name": sample_name,
        "frame_id": frame_id,
        "new_reid_threshold_m": remap["distance_threshold_m"],
        "old_unique_labels_in_frame": len(set(item["label"] for item in old_labels)),
        "new_unique_labels_in_frame": len(set(item["label"] for item in new_labels)),
        "sample_unique_segments": unique_segments,
        "target_segment_id": target_id,
        "old_global_co_visibility_violations": _count_co_visibility_violations(tracking, old_unified),
        "new_global_co_visibility_violations": _count_co_visibility_violations(tracking, None, new_map),
    }
    (proof_root / "clustering_proof_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _count_co_visibility_violations(
    tracking: dict[str, list[dict]],
    old_unified: pd.DataFrame | None,
    new_map: dict[int, int] | None = None,
) -> int:
    total = 0
    old_lookup = None
    if old_unified is not None:
        old_lookup = {
            (int(row["frame_id"]), int(row["tracking_id"])): int(row["strawberry_id"])
            for _, row in old_unified.iterrows()
        }
    for frame_key, tracks in tracking.items():
        frame_id = int(frame_key.split("_")[1])
        labels = []
        for track in tracks:
            tracking_id = int(track["id"])
            if old_lookup is not None:
                label = old_lookup.get((frame_id, tracking_id), -1)
            else:
                label = -1 if new_map is None else new_map.get(tracking_id, -1)
            if label >= 0:
                labels.append(label)
        counts: dict[int, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        total += sum(max(0, count - 1) for count in counts.values())
    return int(total)


if __name__ == "__main__":
    main()
