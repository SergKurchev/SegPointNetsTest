from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Pointcept result/*.npy predictions into the common benchmark layout.")
    parser.add_argument("--result-dir", type=Path, required=True, help="Pointcept save_path/result directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/predictions/pointtransformerv3"),
        help="Output directory for sample_xxxx.npy predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(args.result_dir.glob("*_pred.npy")):
        target = args.output_dir / path.name.replace("_pred", "")
        np.save(target, np.load(path))
        copied += 1
    print(f"Collected {copied} Pointcept predictions into {args.output_dir}")


if __name__ == "__main__":
    main()
