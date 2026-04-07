from __future__ import annotations

import argparse
import json
from pathlib import Path

from point_cloud_test.benchmark_runner import BenchmarkConfig, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\MyProjects\Robostrawberry\point_cloud_test\artifacts\dataset"))
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\MyProjects\Robostrawberry\point_cloud_test\artifacts\benchmark"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--points-per-sample", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--models", nargs="*", default=["odin", "pointtransformerv3", "oneformer3d", "open-yolo-3d"])
    args = parser.parse_args()

    config = BenchmarkConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        points_per_sample=args.points_per_sample,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    summary = run_benchmark(config, models=args.models)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
