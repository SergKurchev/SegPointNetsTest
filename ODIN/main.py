from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--points-per-sample", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2] / "point_cloud_test"
    sys.path.insert(0, str(project_root / "src"))
    from point_cloud_test.benchmark_runner import BenchmarkConfig, run_benchmark

    summary = run_benchmark(
        BenchmarkConfig(
            dataset_root=project_root / "artifacts" / "dataset",
            output_root=project_root / "artifacts" / "benchmark",
            epochs=args.epochs,
            points_per_sample=args.points_per_sample,
            batch_size=args.batch_size,
        ),
        models=["odin"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
