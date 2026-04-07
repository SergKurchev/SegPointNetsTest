from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BuildConfig:
    orbit_data_dir: Path
    coco_annotations: Path
    output_dir: Path
    tracking_json: Path
    reid_output_dir: Path
    min_tracking_views: int = 4
    views_per_sample: int = 4
    max_points_per_segment_per_view: int = 4000
    bbox_expand_px: int = 48
    tracking_iou_threshold: float = 0.05
    strawberry_cluster_threshold_m: float | None = None


def default_config(project_root: Path | None = None) -> BuildConfig:
    if project_root is None:
        project_root = Path(r"D:\MyProjects\Robostrawberry\point_cloud_test")

    orbit_data_dir = Path(
        r"D:\MyProjects\Skoltech\Perception_In_Robotics\FP\strawberry_orbit_data_new\data"
    )
    tracking_root = Path(r"D:\MyProjects\Robostrawberry\strawberry_tracking")
    return BuildConfig(
        orbit_data_dir=orbit_data_dir,
        coco_annotations=tracking_root
        / r"Robo Strawberry.coco-segmentation\train\_annotations.coco.json",
        tracking_json=orbit_data_dir / "tracking.json",
        output_dir=project_root / "artifacts" / "dataset",
        reid_output_dir=project_root / "artifacts" / "reid",
    )
