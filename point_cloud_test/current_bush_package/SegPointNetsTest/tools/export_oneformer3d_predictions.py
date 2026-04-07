from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export OneFormer3D semantic predictions into the common benchmark layout.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/strawberry/oneformer3d_strawberry.py"),
        help="OneFormer3D config file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file to load.")
    parser.add_argument(
        "--train-first",
        action="store_true",
        help="Train the model first in the current process, then export predictions without reloading the checkpoint.")
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Number of epochs for --train-first.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/predictions/oneformer3d"),
        help="Output directory for sample_xxxx.npy predictions.")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Optional MMEngine config overrides.")
    return parser.parse_args()


def _load_model_with_spconv_fix(runner: Runner, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = runner.model.state_dict()
    fixed_state = {}

    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if value.shape == target.shape:
            fixed_state[key] = value
            continue
        if value.ndim == 5:
            permuted = value.permute(1, 2, 3, 4, 0).contiguous()
            if permuted.shape == target.shape:
                fixed_state[key] = permuted
                continue
        fixed_state[key] = value

    missing, unexpected = runner.model.load_state_dict(fixed_state, strict=False)
    if missing:
        print(f"Missing keys after checkpoint load: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys after checkpoint load: {len(unexpected)}")


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(str(args.config))
    cfg.launcher = "none"
    cfg.work_dir = str((args.checkpoint.parent if args.checkpoint else Path("runs/oneformer3d_smoke")))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if args.train_first:
        cfg.train_cfg.max_epochs = args.train_epochs
        cfg.train_cfg.val_interval = args.train_epochs + 1
        cfg.default_hooks.checkpoint.interval = 1

    runner = Runner.from_cfg(cfg)

    if args.train_first:
        runner.train()
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --train-first is used")
        _load_model_with_spconv_fix(runner, args.checkpoint)
    runner.model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0

    with torch.no_grad():
        for batch in runner.test_dataloader:
            outputs = runner.model.test_step(batch)
            for data_sample in outputs:
                sample_name = Path(data_sample.lidar_path).stem
                semantic_pred = np.asarray(
                    data_sample.pred_pts_seg.pts_semantic_mask[0], dtype=np.int64)
                np.save(args.output_dir / f"{sample_name}.npy", semantic_pred)
                exported += 1

    print(f"Exported {exported} OneFormer3D predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
