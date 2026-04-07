from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import camera_to_world, load_depth, load_frame_metadata, load_rgb


@dataclass(slots=True)
class TrackProfile:
    tracking_id: int
    mean_world: np.ndarray
    frames: set[int]
    num_observations: int
    ripeness_hint: str


def load_tracking(tracking_path: Path) -> dict[str, list[dict]]:
    return json.loads(tracking_path.read_text(encoding="utf-8"))


def _frame_key(frame_id: int) -> str:
    return f"frame_{frame_id:03d}"


def _estimate_ripeness(rgb_patch: np.ndarray) -> str:
    if rgb_patch.size == 0:
        return "half-ripe"
    rgb_mean = np.clip(rgb_patch.mean(axis=(0, 1)) * 255.0, 0, 255).astype(np.uint8)[None, None, :]
    import cv2

    hsv = cv2.cvtColor(rgb_mean, cv2.COLOR_RGB2HSV)[0, 0]
    hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if sat < 40 and val < 80:
        return "half-ripe"
    if hue <= 15 or hue >= 170:
        return "ripe"
    if 35 <= hue <= 95:
        return "unripe"
    return "half-ripe"


def build_track_profiles(orbit_data_dir: Path, tracking_json: Path) -> dict[int, TrackProfile]:
    tracking = load_tracking(tracking_json)
    observations: dict[int, list[np.ndarray]] = defaultdict(list)
    frames_by_id: dict[int, set[int]] = defaultdict(set)
    ripeness_votes: dict[int, list[str]] = defaultdict(list)

    for frame_key, tracks in tracking.items():
        frame_id = int(frame_key.split("_")[1])
        frame_meta = load_frame_metadata(orbit_data_dir / "coords" / f"frame_{frame_id:04d}.json")
        depth = load_depth(orbit_data_dir / "depth_est" / f"distance_to_image_plane_{frame_id:04d}.npy")
        rgb = load_rgb(orbit_data_dir / "images" / f"rgb_{frame_id:04d}.png")

        fx = float(frame_meta["intrinsics_px"]["fx"])
        fy = float(frame_meta["intrinsics_px"]["fy"])
        cx = float(frame_meta["intrinsics_px"]["cx"])
        cy = float(frame_meta["intrinsics_px"]["cy"])

        for track in tracks:
            tracking_id = int(track["id"])
            x1, y1, x2, y2 = track["bbox"]
            u = float((x1 + x2) / 2.0)
            v = float((y1 + y2) / 2.0)
            ui = int(round(u))
            vi = int(round(v))
            if not (0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0]):
                continue
            z = float(depth[vi, ui])
            if not np.isfinite(z) or z <= 0.01:
                continue

            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            point_world = camera_to_world(
                np.array([[x_cam, y_cam, z]], dtype=np.float32),
                frame_meta["camera_gt"]["rotation_matrix_3x3"],
                frame_meta["camera_gt"]["position_world_m"],
            )[0]
            observations[tracking_id].append(point_world.astype(np.float32))
            frames_by_id[tracking_id].add(frame_id)

            px1 = max(0, int(x1))
            py1 = max(0, int(y1))
            px2 = min(rgb.shape[1], int(x2))
            py2 = min(rgb.shape[0], int(y2))
            if px2 > px1 and py2 > py1:
                ripeness_votes[tracking_id].append(_estimate_ripeness(rgb[py1:py2, px1:px2]))

    profiles: dict[int, TrackProfile] = {}
    for tracking_id, world_points in observations.items():
        votes = ripeness_votes.get(tracking_id, [])
        ripeness_hint = max(set(votes), key=votes.count) if votes else "half-ripe"
        profiles[tracking_id] = TrackProfile(
            tracking_id=tracking_id,
            mean_world=np.mean(world_points, axis=0).astype(np.float32),
            frames=frames_by_id[tracking_id],
            num_observations=len(world_points),
            ripeness_hint=ripeness_hint,
        )
    return profiles


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


def estimate_merge_threshold(track_profiles: dict[int, TrackProfile]) -> float:
    if len(track_profiles) < 2:
        return 0.05
    positions = np.stack([profile.mean_world for profile in track_profiles.values()], axis=0)
    distances = _pairwise_distances(positions)
    nearest = []
    for row in distances:
        valid = row[row > 1e-9]
        if valid.size:
            nearest.append(float(valid.min()))
    if not nearest:
        return 0.05
    threshold = float(np.percentile(nearest, 75))
    return float(np.clip(threshold, 0.03, 0.06))


def build_strawberry_remap(
    orbit_data_dir: Path,
    tracking_json: Path,
    distance_threshold_m: float | None = None,
) -> dict:
    track_profiles = build_track_profiles(orbit_data_dir, tracking_json)
    if not track_profiles:
        return {
            "distance_threshold_m": 0.05,
            "tracking_id_to_strawberry": {},
            "strawberry_profiles": [],
            "track_profiles": {},
        }

    ids = sorted(track_profiles)
    positions = np.stack([track_profiles[tracking_id].mean_world for tracking_id in ids], axis=0)
    distances = _pairwise_distances(positions)
    threshold = estimate_merge_threshold(track_profiles) if distance_threshold_m is None else distance_threshold_m
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for i, left_id in enumerate(ids):
        left = track_profiles[left_id]
        for j in range(i + 1, len(ids)):
            right_id = ids[j]
            right = track_profiles[right_id]
            if left.frames & right.frames:
                continue
            if left.ripeness_hint != right.ripeness_hint:
                continue
            if float(distances[i, j]) <= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index, tracking_id in enumerate(ids):
        clusters[find(index)].append(tracking_id)

    refined_clusters: list[list[int]] = []
    for cluster in clusters.values():
        refined_clusters.extend(_split_incompatible_cluster(cluster, track_profiles, threshold))

    sorted_clusters = sorted(refined_clusters, key=lambda cluster: (-len(cluster), min(cluster)))
    tracking_id_to_strawberry: dict[int, int] = {}
    strawberry_profiles: list[dict] = []
    for strawberry_id, cluster in enumerate(sorted_clusters):
        cluster_points = np.stack([track_profiles[tracking_id].mean_world for tracking_id in cluster], axis=0)
        cluster_frames = sorted({frame_id for tracking_id in cluster for frame_id in track_profiles[tracking_id].frames})
        cluster_ripeness = [track_profiles[tracking_id].ripeness_hint for tracking_id in cluster]
        dominant_ripeness = max(set(cluster_ripeness), key=cluster_ripeness.count)
        center = np.median(cluster_points, axis=0).astype(np.float32)
        radii = np.linalg.norm(cluster_points - center, axis=1)
        radius = float(np.clip(np.percentile(radii, 90) + 0.03, 0.04, 0.12))
        for tracking_id in cluster:
            tracking_id_to_strawberry[tracking_id] = strawberry_id
        strawberry_profiles.append(
            {
                "strawberry_id": strawberry_id,
                "tracking_ids": cluster,
                "center_world": center.tolist(),
                "radius_m": radius,
                "ripeness_hint": dominant_ripeness,
                "num_frames": len(cluster_frames),
                "num_tracks": len(cluster),
            }
        )

    return {
        "distance_threshold_m": float(threshold),
        "tracking_id_to_strawberry": tracking_id_to_strawberry,
        "strawberry_profiles": strawberry_profiles,
        "track_profiles": {
            tracking_id: {
                "mean_world": profile.mean_world.tolist(),
                "frames": sorted(profile.frames),
                "num_observations": profile.num_observations,
                "ripeness_hint": profile.ripeness_hint,
            }
            for tracking_id, profile in track_profiles.items()
        },
    }


def _split_incompatible_cluster(
    cluster: list[int],
    track_profiles: dict[int, TrackProfile],
    distance_threshold_m: float,
) -> list[list[int]]:
    ordered = sorted(
        cluster,
        key=lambda tracking_id: (-track_profiles[tracking_id].num_observations, tracking_id),
    )
    subclusters: list[list[int]] = []
    for tracking_id in ordered:
        profile = track_profiles[tracking_id]
        placed = False
        for subcluster in subclusters:
            if any(track_profiles[other].frames & profile.frames for other in subcluster):
                continue
            if any(track_profiles[other].ripeness_hint != profile.ripeness_hint for other in subcluster):
                continue
            subcluster_center = np.mean(
                np.stack([track_profiles[other].mean_world for other in subcluster], axis=0),
                axis=0,
            )
            if float(np.linalg.norm(profile.mean_world - subcluster_center)) > distance_threshold_m:
                continue
            subcluster.append(tracking_id)
            placed = True
            break
        if not placed:
            subclusters.append([tracking_id])
    return subclusters


def save_strawberry_remap(payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strawberry_remap.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
