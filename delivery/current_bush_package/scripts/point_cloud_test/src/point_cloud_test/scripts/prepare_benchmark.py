from __future__ import annotations

from pathlib import Path

from point_cloud_test.benchmark_runner import prepare_dataset


def main() -> None:
    dataset_root = Path(r"D:\MyProjects\Robostrawberry\point_cloud_test\artifacts\dataset")
    split_path = prepare_dataset(dataset_root)
    print(f"Saved benchmark split to {split_path}")


if __name__ == "__main__":
    main()
