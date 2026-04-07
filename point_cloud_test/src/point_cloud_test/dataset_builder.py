from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coco_utils import CocoIndex, segmentation_to_mask
from .config import BuildConfig
from .geometry import (
    backproject_depth,
    camera_to_world,
    load_depth,
    load_frame_metadata,
    load_rgb,
    write_ascii_ply,
)
from .point_cloud_ops import (
    filter_points_to_centered_ellipsoid,
    filter_points_to_bush_volume,
    filter_points_to_sphere,
    keep_largest_connected_component,
    trim_points_by_radius,
    voxel_downsample,
)
from .track_reid import build_strawberry_remap, save_strawberry_remap

CLASS_NAME_TO_ID = {"background": 0, "ripe": 1, "half-ripe": 2, "unripe": 3}


@dataclass(slots=True)
class SegmentRecord:
    segment_id: int
    class_name: str
    points_xyzrgb: np.ndarray


def _load_tracking(tracking_path: Path) -> dict[str, list[dict]]:
    return json.loads(tracking_path.read_text(encoding="utf-8"))


def _xyxy_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _ripeness_class(colors_rgb_01: np.ndarray) -> str:
    if colors_rgb_01.size == 0:
        return "half-ripe"
    rgb = np.clip(colors_rgb_01.mean(axis=0) * 255.0, 0, 255).astype(np.uint8)[None, None, :]
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0]
    hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if sat < 40 and val < 80:
        return "half-ripe"
    if hue <= 15 or hue >= 170:
        return "ripe"
    if 35 <= hue <= 95:
        return "unripe"
    return "half-ripe"


def _frame_key(frame_id: int) -> str:
    return f"frame_{frame_id:03d}"


def _paths_for_frame(config: BuildConfig, frame_id: int) -> tuple[Path, Path, Path]:
    image_name = f"rgb_{frame_id:04d}.png"
    rgb_path = config.orbit_data_dir / "images" / image_name
    depth_path = config.orbit_data_dir / "depth_est" / f"distance_to_image_plane_{frame_id:04d}.npy"
    frame_json = config.orbit_data_dir / "coords" / f"frame_{frame_id:04d}.json"
    return rgb_path, depth_path, frame_json


def _clean_segment_points(
    points_world: np.ndarray,
    colors: np.ndarray,
    bush_world: dict,
    profile: dict | None,
    is_target: bool,
) -> tuple[np.ndarray, np.ndarray]:
    points_world, colors = filter_points_to_bush_volume(points_world, colors, bush_world)
    if len(points_world) == 0:
        return points_world, colors
    bush_center = np.array([bush_world["x"], bush_world["y"], bush_world["z"]], dtype=np.float32)
    points_world, colors = filter_points_to_centered_ellipsoid(
        points_world,
        colors,
        bush_center,
        radii_m=(0.42, 0.42, 0.30),
    )
    if len(points_world) == 0:
        return points_world, colors
    if profile is not None:
        center = np.asarray(profile["center_world"], dtype=np.float32)
        radius = float(profile["radius_m"])
        margin = 0.03 if is_target else 0.02
        points_world, colors = filter_points_to_sphere(points_world, colors, center, radius + margin)
        if len(points_world) == 0:
            return points_world, colors
    points_world, colors = keep_largest_connected_component(
        points_world,
        colors,
        radius_m=0.03 if is_target else 0.035,
        min_component_size=12 if is_target else 20,
    )
    return points_world, colors


def build_dataset(config: BuildConfig, max_samples: int | None = None) -> list[dict]:
    coco = CocoIndex(config.coco_annotations)
    tracking = _load_tracking(config.tracking_json)
    reid_payload = build_strawberry_remap(
        orbit_data_dir=config.orbit_data_dir,
        tracking_json=config.tracking_json,
        distance_threshold_m=config.strawberry_cluster_threshold_m,
    )
    save_strawberry_remap(reid_payload, config.reid_output_dir)
    tracking_to_strawberry = {
        int(tracking_id): int(strawberry_id)
        for tracking_id, strawberry_id in reid_payload["tracking_id_to_strawberry"].items()
    }
    strawberry_profiles = {
        int(profile["strawberry_id"]): profile for profile in reid_payload["strawberry_profiles"]
    }

    visibility: dict[int, list[int]] = defaultdict(list)
    for frame_key, tracks in tracking.items():
        frame_id = int(frame_key.split("_")[1])
        for track in tracks:
            strawberry_id = tracking_to_strawberry.get(int(track["id"]), int(track["id"]))
            visibility[strawberry_id].append(frame_id)

    grouped_candidates: dict[int, list[list[int]]] = {}
    for segment_id, frame_ids in visibility.items():
        unique_frames = sorted(set(frame_ids))
        if len(unique_frames) < config.min_tracking_views:
            continue
        segment_groups: list[list[int]] = []
        for start in range(0, len(unique_frames) - config.views_per_sample + 1, config.views_per_sample):
            group = unique_frames[start : start + config.views_per_sample]
            if len(group) == config.views_per_sample:
                segment_groups.append(group)
        if segment_groups:
            grouped_candidates[int(segment_id)] = segment_groups

    candidate_groups: list[tuple[int, list[int]]] = []
    ordered_segment_ids = sorted(
        grouped_candidates,
        key=lambda segment_id: (-len(grouped_candidates[segment_id]), segment_id),
    )
    max_groups_per_segment = max((len(groups) for groups in grouped_candidates.values()), default=0)
    for group_index in range(max_groups_per_segment):
        for segment_id in ordered_segment_ids:
            groups = grouped_candidates[segment_id]
            if group_index < len(groups):
                candidate_groups.append((segment_id, groups[group_index]))

    if max_samples is not None:
        candidate_groups = candidate_groups[:max_samples]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for sample_index, (target_segment_id, frame_group) in enumerate(candidate_groups):
        segment_records: list[SegmentRecord] = []
        background_records: list[np.ndarray] = []
        frame_summaries: list[dict] = []
        target_profile = strawberry_profiles.get(int(target_segment_id))
        target_radius = float(target_profile["radius_m"]) if target_profile is not None else 0.08

        for frame_id in frame_group:
            rgb_path, depth_path, frame_json = _paths_for_frame(config, frame_id)
            frame_meta = load_frame_metadata(frame_json)
            intrinsics = frame_meta["intrinsics_px"]
            rgb = load_rgb(rgb_path)
            depth = load_depth(depth_path)
            coco_image, annotations = coco.annotations_for_original_name(frame_meta["rgb_file"])
            tracks = tracking.get(_frame_key(frame_id), [])
            frame_segment_count = 0

            full_crop_mask = np.zeros((coco_image.height, coco_image.width), dtype=bool)
            for track in tracks:
                strawberry_id = tracking_to_strawberry.get(int(track["id"]), int(track["id"]))
                if strawberry_id != target_segment_id:
                    continue
                x1, y1, x2, y2 = track["bbox"]
                x1 = max(0, int(x1) - config.bbox_expand_px)
                y1 = max(0, int(y1) - config.bbox_expand_px)
                x2 = min(coco_image.width, int(x2) + config.bbox_expand_px)
                y2 = min(coco_image.height, int(y2) + config.bbox_expand_px)
                full_crop_mask[y1:y2, x1:x2] = True

            if not full_crop_mask.any():
                continue

            occupied_mask = np.zeros_like(full_crop_mask)
            for ann in annotations:
                mask = segmentation_to_mask(ann.segmentation, coco_image.height, coco_image.width)
                if not np.logical_and(mask, full_crop_mask).any():
                    continue

                ann_box = np.array(
                    [
                        ann.bbox_xywh[0],
                        ann.bbox_xywh[1],
                        ann.bbox_xywh[0] + ann.bbox_xywh[2],
                        ann.bbox_xywh[1] + ann.bbox_xywh[3],
                    ],
                    dtype=np.float32,
                )
                best_track_id = -1
                best_iou = 0.0
                for track in tracks:
                    track_box = np.asarray(track["bbox"], dtype=np.float32)
                    iou = _xyxy_iou(ann_box, track_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = tracking_to_strawberry.get(int(track["id"]), int(track["id"]))
                if best_iou < config.tracking_iou_threshold:
                    best_track_id = 100000 + int(ann.annotation_id)

                points_camera, colors = backproject_depth(
                    depth,
                    rgb,
                    mask & full_crop_mask,
                    intrinsics,
                    min_depth_m=0.05,
                )
                if len(points_camera) == 0:
                    continue
                points_world = camera_to_world(
                    points_camera,
                    frame_meta["camera_gt"]["rotation_matrix_3x3"],
                    frame_meta["camera_gt"]["position_world_m"],
                )
                segment_profile = strawberry_profiles.get(int(best_track_id))
                points_world, colors = _clean_segment_points(
                    points_world,
                    colors,
                    frame_meta["bush_world"],
                    segment_profile,
                    is_target=best_track_id == target_segment_id,
                )
                if len(points_world) == 0:
                    continue
                points_xyzrgb = np.concatenate([points_world, colors], axis=1)
                if len(points_xyzrgb) > config.max_points_per_segment_per_view:
                    step = max(1, len(points_xyzrgb) // config.max_points_per_segment_per_view)
                    points_xyzrgb = points_xyzrgb[::step][: config.max_points_per_segment_per_view]
                class_name = _ripeness_class(points_xyzrgb[:, 3:])
                segment_records.append(
                    SegmentRecord(
                        segment_id=best_track_id,
                        class_name=class_name,
                        points_xyzrgb=points_xyzrgb.astype(np.float32),
                    )
                )
                frame_segment_count += 1
                occupied_mask |= mask

            background_mask = full_crop_mask & (~occupied_mask)
            background_points, background_colors = backproject_depth(
                depth,
                rgb,
                background_mask,
                intrinsics,
                min_depth_m=0.05,
            )
            if len(background_points):
                background_world = camera_to_world(
                    background_points,
                    frame_meta["camera_gt"]["rotation_matrix_3x3"],
                    frame_meta["camera_gt"]["position_world_m"],
                )
                background_world, background_colors = filter_points_to_bush_volume(
                    background_world,
                    background_colors,
                    frame_meta["bush_world"],
                    xy_radius_m=0.45,
                    z_margin_down_m=0.02,
                    z_margin_up_m=0.35,
                )
                if len(background_world) == 0:
                    frame_summaries.append({"frame_id": frame_id, "num_segments": frame_segment_count})
                    continue
                background_xyzrgb = np.concatenate([background_world, background_colors], axis=1)
                if len(background_xyzrgb) > config.max_points_per_segment_per_view:
                    step = max(1, len(background_xyzrgb) // config.max_points_per_segment_per_view)
                    background_xyzrgb = background_xyzrgb[::step][: config.max_points_per_segment_per_view]
                background_records.append(background_xyzrgb.astype(np.float32))

            frame_summaries.append({"frame_id": frame_id, "num_segments": frame_segment_count})

        if not segment_records and not background_records:
            continue

        points_list: list[np.ndarray] = []
        segment_ids: list[np.ndarray] = []
        class_ids: list[np.ndarray] = []
        summary_segments: list[dict] = []

        for record in segment_records:
            points_list.append(record.points_xyzrgb)
            segment_ids.append(np.full(len(record.points_xyzrgb), record.segment_id, dtype=np.int32))
            class_ids.append(
                np.full(len(record.points_xyzrgb), CLASS_NAME_TO_ID[record.class_name], dtype=np.int16)
            )
        for background in background_records:
            points_list.append(background)
            segment_ids.append(np.full(len(background), -1, dtype=np.int32))
            class_ids.append(np.full(len(background), CLASS_NAME_TO_ID["background"], dtype=np.int16))

        merged_points = np.concatenate(points_list, axis=0).astype(np.float32)
        merged_segment_ids = np.concatenate(segment_ids, axis=0)
        merged_class_ids = np.concatenate(class_ids, axis=0)
        merged_with_labels = np.concatenate(
            [
                merged_points,
                merged_segment_ids[:, None].astype(np.float32),
                merged_class_ids[:, None].astype(np.float32),
            ],
            axis=1,
        )
        merged_with_labels = voxel_downsample(merged_with_labels, voxel_size_m=0.0075)
        merged_points = merged_with_labels[:, :6].astype(np.float32)
        merged_segment_ids = np.rint(merged_with_labels[:, 6]).astype(np.int32)
        merged_class_ids = np.rint(merged_with_labels[:, 7]).astype(np.int16)
        if target_profile is not None:
            target_center = np.asarray(target_profile["center_world"], dtype=np.float32)
            merged_labeled = np.concatenate(
                [
                    merged_points,
                    merged_segment_ids[:, None].astype(np.float32),
                    merged_class_ids[:, None].astype(np.float32),
                ],
                axis=1,
            )
            merged_labeled = trim_points_by_radius(
                merged_labeled,
                center_world=target_center,
                radius_m=max(0.45, target_radius + 0.32),
            )
            if len(merged_labeled) == 0:
                continue
            merged_points = merged_labeled[:, :6].astype(np.float32)
            merged_segment_ids = np.rint(merged_labeled[:, 6]).astype(np.int32)
            merged_class_ids = np.rint(merged_labeled[:, 7]).astype(np.int16)

        unique_segment_ids = [value for value in np.unique(merged_segment_ids) if value >= 0]
        for segment_id in unique_segment_ids:
            segment_mask = merged_segment_ids == segment_id
            segment_classes = merged_class_ids[segment_mask]
            class_values, class_counts = np.unique(segment_classes, return_counts=True)
            dominant_class_id = int(class_values[np.argmax(class_counts)])
            dominant_class_name = next(
                name for name, class_id in CLASS_NAME_TO_ID.items() if class_id == dominant_class_id
            )
            summary_segments.append(
                {
                    "segment_id": int(segment_id),
                    "class_name": dominant_class_name,
                    "num_points": int(segment_mask.sum()),
                }
            )

        sample_name = f"sample_{sample_index:04d}"
        sample_dir = config.output_dir
        np.savez_compressed(
            sample_dir / f"{sample_name}.npz",
            points_xyzrgb=merged_points,
            segment_id=merged_segment_ids,
            class_id=merged_class_ids,
        )
        write_ascii_ply(sample_dir / f"{sample_name}.ply", merged_points)
        metadata = {
            "sample_name": sample_name,
            "target_segment_id": int(target_segment_id),
            "frame_group": frame_group,
            "num_points": int(len(merged_points)),
            "segments": summary_segments,
            "frames": frame_summaries,
            "target_center_world": target_profile["center_world"] if target_profile is not None else None,
            "target_radius_m": target_radius,
            "reid_threshold_m": reid_payload["distance_threshold_m"],
        }
        (sample_dir / f"{sample_name}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        manifest.append(metadata)

    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
