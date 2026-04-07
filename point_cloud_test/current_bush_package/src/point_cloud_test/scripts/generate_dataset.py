from __future__ import annotations

import argparse
from pathlib import Path

from point_cloud_test.config import default_config
from point_cloud_test.dataset_builder import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--views-per-sample", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strawberry-cluster-threshold-m", type=float, default=None)
    args = parser.parse_args()

    config = default_config()
    config.views_per_sample = args.views_per_sample
    config.min_tracking_views = args.views_per_sample
    config.strawberry_cluster_threshold_m = args.strawberry_cluster_threshold_m
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    manifest = build_dataset(config=config, max_samples=args.max_samples)
    print(f"Generated {len(manifest)} samples into {config.output_dir}")


if __name__ == "__main__":
    main()
