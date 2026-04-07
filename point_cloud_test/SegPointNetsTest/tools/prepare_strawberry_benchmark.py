from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


CLASS_ID_TO_NAME = {
    0: "background",
    1: "ripe",
    2: "half-ripe",
    3: "unripe",
}
THING_CLASS_IDS = [1, 2, 3]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _point_cloud_root() -> Path:
    return _workspace_root() / "point_cloud_test"


def _insert_point_cloud_src() -> None:
    src_root = _point_cloud_root() / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_insert_point_cloud_src()

from point_cloud_test.benchmark_data import create_split, load_manifest, save_split  # noqa: E402
from point_cloud_test.coco_utils import CocoIndex, segmentation_to_mask  # noqa: E402
from point_cloud_test.config import default_config  # noqa: E402
from point_cloud_test.geometry import load_depth, load_frame_metadata, load_rgb, write_ascii_ply  # noqa: E402
from point_cloud_test.scripts.render_full_bush_views import build_full_bush_cloud  # noqa: E402
from point_cloud_test.track_reid import build_strawberry_remap, load_tracking  # noqa: E402


def _xyxy_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _frame_key(frame_id: int) -> str:
    return f"frame_{frame_id:03d}"


def _load_npz(dataset_root: Path, sample_name: str) -> dict[str, np.ndarray]:
    payload = np.load(dataset_root / f"{sample_name}.npz")
    return {key: payload[key] for key in payload.files}


def _ensure_split(dataset_root: Path) -> dict[str, list[str]]:
    split_path = dataset_root / "splits.json"
    if not split_path.exists():
        save_split(dataset_root, create_split(dataset_root))
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return {name: [value for value in values if (dataset_root / f"{value}.npz").exists()] for name, values in payload.items()}


def _instance_ids_for_sample(segment_ids: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    unique_segments = sorted(int(value) for value in np.unique(segment_ids) if value >= 0)
    instance_lookup = {segment_id: idx for idx, segment_id in enumerate(unique_segments)}
    mapped = np.full(len(segment_ids), -1, dtype=np.int32)
    for row_idx, segment_id in enumerate(segment_ids.astype(np.int32)):
        if int(class_ids[row_idx]) == 0 or segment_id < 0:
            continue
        mapped[row_idx] = instance_lookup[int(segment_id)]
    return mapped


def _superpoints_for_sample(segment_ids: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    superpoints = np.full(len(segment_ids), -1, dtype=np.int64)
    foreground_segments = sorted(int(value) for value in np.unique(segment_ids) if value >= 0)
    next_id = 0
    for segment_id in foreground_segments:
        mask = segment_ids == segment_id
        superpoints[mask] = next_id
        next_id += 1

    background_mask = segment_ids < 0
    if background_mask.any():
        voxel_keys = np.floor(points_xyz[background_mask] / 0.015).astype(np.int32)
        _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
        superpoints[background_mask] = inverse.astype(np.int64) + next_id
    return superpoints


def _bounding_boxes(points_xyz: np.ndarray, class_ids: np.ndarray, instance_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boxes: list[np.ndarray] = []
    labels: list[int] = []
    for instance_id in sorted(int(value) for value in np.unique(instance_ids) if value >= 0):
        mask = instance_ids == instance_id
        pts = points_xyz[mask]
        if len(pts) == 0:
            continue
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        size = np.maximum(maxs - mins, 1e-4)
        boxes.append(np.concatenate([center, size], axis=0).astype(np.float32))
        labels.append(int(class_ids[mask][0]) - 1)
    if not boxes:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(boxes, axis=0), np.asarray(labels, dtype=np.int64)


def export_pointcept(dataset_root: Path, export_root: Path, split: dict[str, list[str]]) -> dict[str, int]:
    pointcept_root = export_root / "pointcept"
    pointcept_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split_name, sample_names in split.items():
        counts[split_name] = len(sample_names)
        split_dir = pointcept_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        index_entries: list[str] = []
        for sample_name in sample_names:
            sample = _load_npz(dataset_root, sample_name)
            instance_ids = _instance_ids_for_sample(sample["segment_id"], sample["class_id"])
            sample_dir = split_dir / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.save(sample_dir / "coord.npy", sample["points_xyzrgb"][:, :3].astype(np.float32))
            np.save(sample_dir / "color.npy", np.clip(sample["points_xyzrgb"][:, 3:] * 255.0, 0, 255).astype(np.float32))
            np.save(sample_dir / "segment.npy", sample["class_id"].astype(np.int32))
            np.save(sample_dir / "instance.npy", instance_ids.astype(np.int32))
            index_entries.append(f"{split_name}/{sample_name}")
        (pointcept_root / f"{split_name}.json").write_text(json.dumps(index_entries, indent=2), encoding="utf-8")
    return counts


def export_oneformer3d(dataset_root: Path, export_root: Path, split: dict[str, list[str]]) -> dict[str, int]:
    oneformer_root = export_root / "oneformer3d"
    for folder_name in ["points", "instance_mask", "semantic_mask", "super_points", "meta_data"]:
        (oneformer_root / folder_name).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    info_payload: dict[str, list[dict]] = {}
    for split_name, sample_names in split.items():
        counts[split_name] = len(sample_names)
        infos: list[dict] = []
        for sample_name in sample_names:
            sample = _load_npz(dataset_root, sample_name)
            xyz = sample["points_xyzrgb"][:, :3].astype(np.float32)
            rgb255 = np.clip(sample["points_xyzrgb"][:, 3:] * 255.0, 0, 255).astype(np.float32)
            points = np.concatenate([xyz, rgb255], axis=1).astype(np.float32)
            instance_ids = _instance_ids_for_sample(sample["segment_id"], sample["class_id"])
            semantic_ids = sample["class_id"].astype(np.int64)
            superpoints = _superpoints_for_sample(sample["segment_id"], xyz)
            gt_boxes, gt_classes = _bounding_boxes(xyz, semantic_ids, instance_ids)

            points.tofile(oneformer_root / "points" / f"{sample_name}.bin")
            instance_ids.astype(np.int64).tofile(oneformer_root / "instance_mask" / f"{sample_name}.bin")
            semantic_ids.astype(np.int64).tofile(oneformer_root / "semantic_mask" / f"{sample_name}.bin")
            superpoints.astype(np.int64).tofile(oneformer_root / "super_points" / f"{sample_name}.bin")

            info = {
                "point_cloud": {"num_features": 6, "lidar_idx": sample_name},
                "pts_path": f"points/{sample_name}.bin",
                "super_pts_path": f"super_points/{sample_name}.bin",
                "pts_instance_mask_path": f"instance_mask/{sample_name}.bin",
                "pts_semantic_mask_path": f"semantic_mask/{sample_name}.bin",
                "annos": {
                    "gt_num": int(len(gt_boxes)),
                    "name": np.asarray([CLASS_ID_TO_NAME[int(class_id) + 1] for class_id in gt_classes], dtype=object),
                    "location": gt_boxes[:, :3],
                    "dimensions": gt_boxes[:, 3:6],
                    "gt_boxes_upright_depth": gt_boxes,
                    "unaligned_location": gt_boxes[:, :3],
                    "unaligned_dimensions": gt_boxes[:, 3:6],
                    "unaligned_gt_boxes_upright_depth": gt_boxes,
                    "index": np.arange(len(gt_boxes), dtype=np.int32),
                    "class": gt_classes.astype(np.int32),
                    "axis_align_matrix": np.eye(4, dtype=np.float32),
                },
            }
            infos.append(info)

        info_payload[split_name] = infos
        (oneformer_root / "meta_data" / f"scannetv2_{split_name}.txt").write_text(
            "\n".join(sample_names) + ("\n" if sample_names else ""),
            encoding="utf-8",
        )

    for split_name, infos in info_payload.items():
        with (oneformer_root / f"scannet_oneformer3d_infos_{split_name}.pkl").open("wb") as handle:
            pickle.dump(infos, handle)
    return counts


def _camera_pose_matrix(frame_meta: dict) -> np.ndarray:
    rotation = np.asarray(frame_meta["camera_gt"]["rotation_matrix_3x3"], dtype=np.float32)
    basis = np.stack([rotation[0], -rotation[1], -rotation[2]], axis=1)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = basis
    pose[:3, 3] = np.asarray(
        [
            frame_meta["camera_gt"]["position_world_m"]["x"],
            frame_meta["camera_gt"]["position_world_m"]["y"],
            frame_meta["camera_gt"]["position_world_m"]["z"],
        ],
        dtype=np.float32,
    )
    return pose


def _intrinsic_matrix_3x3(frame_meta: dict) -> np.ndarray:
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = float(frame_meta["intrinsics_px"]["fx"])
    intrinsic[1, 1] = float(frame_meta["intrinsics_px"]["fy"])
    intrinsic[0, 2] = float(frame_meta["intrinsics_px"]["cx"])
    intrinsic[1, 2] = float(frame_meta["intrinsics_px"]["cy"])
    return intrinsic


def _intrinsic_matrix_4x4(frame_meta: dict) -> np.ndarray:
    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[:3, :3] = _intrinsic_matrix_3x3(frame_meta)
    return intrinsic


def _rgb_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _depth_png(depth_m: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(depth_m * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def _match_annotations_to_tracks(
    annotations,
    tracks: list[dict],
    tracking_to_strawberry: dict[int, int],
    track_profiles: dict[int, dict],
) -> list[dict]:
    matched: list[dict] = []
    for ann in annotations:
        ann_box = np.array(
            [
                ann.bbox_xywh[0],
                ann.bbox_xywh[1],
                ann.bbox_xywh[0] + ann.bbox_xywh[2],
                ann.bbox_xywh[1] + ann.bbox_xywh[3],
            ],
            dtype=np.float32,
        )
        best_track = None
        best_iou = 0.0
        for track in tracks:
            track_box = np.asarray(track["bbox"], dtype=np.float32)
            iou = _xyxy_iou(ann_box, track_box)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track is None or best_iou < 0.05:
            continue
        tracking_id = int(best_track["id"])
        strawberry_id = tracking_to_strawberry.get(tracking_id, tracking_id)
        ripeness = track_profiles.get(tracking_id, {}).get("ripeness_hint", "half-ripe")
        class_id = next((key for key, value in CLASS_ID_TO_NAME.items() if value == ripeness), 2)
        matched.append(
            {
                "annotation": ann,
                "tracking_id": tracking_id,
                "strawberry_id": strawberry_id,
                "class_id": class_id,
                "ripeness": ripeness,
            }
        )
    return matched


def export_multiview_scene(export_root: Path, frame_stride: int = 2) -> dict[str, object]:
    config = default_config(_point_cloud_root())
    orbit_root = config.orbit_data_dir
    coco = CocoIndex(config.coco_annotations)
    tracking = load_tracking(config.tracking_json)
    reid_payload = build_strawberry_remap(
        orbit_data_dir=config.orbit_data_dir,
        tracking_json=config.tracking_json,
        distance_threshold_m=None,
    )
    tracking_to_strawberry = {
        int(tracking_id): int(strawberry_id)
        for tracking_id, strawberry_id in reid_payload["tracking_id_to_strawberry"].items()
    }
    track_profiles = {int(key): value for key, value in reid_payload["track_profiles"].items()}

    scene_root = export_root / "multiview"
    openyolo_root = scene_root / "openyolo3d_scene" / "strawberry_bush"
    odin_root = scene_root / "odin_frames" / "strawberry_bush"
    for folder in [
        openyolo_root / "poses",
        openyolo_root / "color",
        openyolo_root / "depth",
        odin_root / "color",
        odin_root / "depth",
        odin_root / "pose",
        odin_root / "intrinsic",
        odin_root / "segments",
        odin_root / "valids",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted((orbit_root / "coords").glob("frame_*.json"))[::frame_stride]
    if not frame_paths:
        raise FileNotFoundError(f"No frame metadata found under {orbit_root / 'coords'}")

    coco_output = {
        "info": {"description": "Strawberry bush multiview export for ODIN"},
        "licenses": [],
        "images": [],
        "depths": [],
        "poses": [],
        "intrinsics": [],
        "valids": [],
        "segments": [],
        "annotations": [],
        "categories": [
            {"id": class_id, "name": CLASS_ID_TO_NAME[class_id], "supercategory": "strawberry"}
            for class_id in THING_CLASS_IDS
        ],
    }

    annotation_id = 1
    rendered_frames: list[dict[str, object]] = []
    for image_id, frame_path in enumerate(frame_paths, start=1):
        frame_meta = load_frame_metadata(frame_path)
        frame_id = int(frame_meta["frame_id"])
        rgb = load_rgb(orbit_root / "images" / frame_meta["rgb_file"])
        depth_m = load_depth(orbit_root / "depth_est" / f"distance_to_image_plane_{frame_id:04d}.npy")
        coco_image, annotations = coco.annotations_for_original_name(frame_meta["rgb_file"])
        tracks = tracking.get(_frame_key(frame_id), [])
        matched = _match_annotations_to_tracks(annotations, tracks, tracking_to_strawberry, track_profiles)

        color_name = f"{image_id - 1}.jpg"
        depth_name = f"{image_id - 1}.png"
        pose_name = f"{image_id - 1}.txt"
        intrinsic_name = f"{image_id - 1}.txt"
        segment_name = f"{image_id - 1}.png"
        valid_name = f"{image_id - 1}.png"

        cv2.imwrite(str(openyolo_root / "color" / color_name), _rgb_bgr(rgb), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(openyolo_root / "depth" / depth_name), _depth_png(depth_m))
        np.savetxt(openyolo_root / "poses" / pose_name, _camera_pose_matrix(frame_meta), fmt="%.8f")

        cv2.imwrite(str(odin_root / "color" / color_name), _rgb_bgr(rgb), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(odin_root / "depth" / depth_name), _depth_png(depth_m))
        np.savetxt(odin_root / "pose" / pose_name, _camera_pose_matrix(frame_meta), fmt="%.8f")
        np.savetxt(odin_root / "intrinsic" / intrinsic_name, _intrinsic_matrix_3x3(frame_meta), fmt="%.8f")

        valid_mask = np.where(np.isfinite(depth_m) & (depth_m > 0.05), 255, 0).astype(np.uint8)
        cv2.imwrite(str(odin_root / "valids" / valid_name), valid_mask)

        segments = np.zeros((coco_image.height, coco_image.width), dtype=np.uint16)
        for match in matched:
            ann = match["annotation"]
            mask = segmentation_to_mask(ann.segmentation, coco_image.height, coco_image.width)
            semantic_instance_id = int(match["class_id"]) * 1000 + int(match["strawberry_id"]) + 1
            segments[mask] = semantic_instance_id

            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(match["class_id"]),
                    "bbox": [float(value) for value in ann.bbox_xywh],
                    "area": float(ann.bbox_xywh[2] * ann.bbox_xywh[3]),
                    "iscrowd": 0,
                    "segmentation": ann.segmentation,
                    "semantic_instance_id_scannet": semantic_instance_id,
                    "strawberry_id": int(match["strawberry_id"]),
                    "tracking_id": int(match["tracking_id"]),
                }
            )
            annotation_id += 1

        cv2.imwrite(str(odin_root / "segments" / segment_name), segments)

        image_record = {
            "id": image_id,
            "file_name": f"strawberry_bush/color/{color_name}",
            "width": int(coco_image.width),
            "height": int(coco_image.height),
        }
        coco_output["images"].append(image_record)
        coco_output["depths"].append({"id": image_id, "file_name": f"strawberry_bush/depth/{depth_name}"})
        coco_output["poses"].append({"id": image_id, "file_name": f"strawberry_bush/pose/{pose_name}"})
        coco_output["intrinsics"].append({"id": image_id, "file_name": f"strawberry_bush/intrinsic/{intrinsic_name}"})
        coco_output["valids"].append({"id": image_id, "file_name": f"strawberry_bush/valids/{valid_name}"})
        coco_output["segments"].append({"id": image_id, "file_name": f"strawberry_bush/segments/{segment_name}"})
        rendered_frames.append({"frame_id": frame_id, "image_id": image_id, "num_annotations": len(matched)})

    first_frame_meta = load_frame_metadata(frame_paths[0])
    np.savetxt(openyolo_root / "intrinsics.txt", _intrinsic_matrix_4x4(first_frame_meta), fmt="%.8f")

    full_cloud = build_full_bush_cloud(orbit_root=orbit_root, frame_stride=frame_stride, pixel_stride=8)
    write_ascii_ply(openyolo_root / "strawberry_bush_mesh.ply", full_cloud)
    np.save(openyolo_root / "strawberry_bush.npy", full_cloud.astype(np.float32))

    gt_lines = []
    for row_idx, point in enumerate(full_cloud, start=1):
        gt_lines.append(
            f"{row_idx} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} 0"
        )
    ground_truth_root = scene_root / "openyolo3d_scene" / "ground_truth"
    ground_truth_root.mkdir(parents=True, exist_ok=True)
    (ground_truth_root / "strawberry_bush.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")

    coco_json_path = scene_root / "odin_strawberry_bush_val.coco.json"
    coco_json_path.write_text(json.dumps(coco_output, indent=2), encoding="utf-8")

    return {
        "num_frames": len(frame_paths),
        "frame_stride": frame_stride,
        "rendered_frames": rendered_frames,
        "openyolo_scene_dir": str(openyolo_root),
        "odin_scene_root": str(odin_root.parent),
        "odin_coco_json": str(coco_json_path),
        "full_bush_point_cloud": str(openyolo_root / "strawberry_bush_mesh.ply"),
    }


def write_common_manifest(dataset_root: Path, export_root: Path, split: dict[str, list[str]], multiview_summary: dict[str, object]) -> Path:
    manifest = {
        "dataset_root": str(dataset_root),
        "class_map": CLASS_ID_TO_NAME,
        "splits": split,
        "models": {
            "pointtransformerv3": {
                "type": "point-cloud-semantic-segmentation",
                "dataset_format": "Pointcept DefaultDataset",
                "export_root": str(export_root / "pointcept"),
            },
            "oneformer3d": {
                "type": "point-cloud-unified-segmentation",
                "dataset_format": "ScanNet-style bins+pkl",
                "export_root": str(export_root / "oneformer3d"),
            },
            "odin": {
                "type": "multi-view RGB-D -> 3D",
                "dataset_format": "ScanNet-context COCO-like scene",
                "export_root": str(export_root / "multiview"),
                "scene_json": multiview_summary["odin_coco_json"],
            },
            "openyolo3d": {
                "type": "multi-view RGB-D + reconstructed 3D scene",
                "dataset_format": "Replica-like scene folder",
                "export_root": str(export_root / "multiview" / "openyolo3d_scene"),
            },
        },
        "multiview_summary": multiview_summary,
    }
    manifest_path = export_root / "strawberry_benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the strawberry bush dataset for 4 official segmentation models.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_point_cloud_root() / "artifacts" / "dataset",
        help="Path to the generated merged point-cloud dataset.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_exports" / "strawberry_benchmark",
        help="Directory where all model-specific exports will be written.",
    )
    parser.add_argument("--frame-stride", type=int, default=2, help="Use every Nth orbit frame for multiview scene export.")
    parser.add_argument("--force", action="store_true", help="Delete the previous export directory before writing a new one.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    export_root = args.export_root.resolve()
    if args.force and export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    split = _ensure_split(dataset_root)
    pointcept_counts = export_pointcept(dataset_root, export_root, split)
    oneformer_counts = export_oneformer3d(dataset_root, export_root, split)
    multiview_summary = export_multiview_scene(export_root, frame_stride=args.frame_stride)
    manifest_path = write_common_manifest(dataset_root, export_root, split, multiview_summary)

    summary = {
        "pointcept_counts": pointcept_counts,
        "oneformer3d_counts": oneformer_counts,
        "multiview_frames": multiview_summary["num_frames"],
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
