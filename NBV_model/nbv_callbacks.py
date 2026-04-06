"""
nbv_callbacks.py — Stable-Baselines3 callbacks for NBV SAC training.

Available callbacks:
    NBVTrainingMetricsCallback  — logs SAC internals (alpha, Q, losses)
    NBVEvaluationCallback       — periodic evaluation + best-model saving
    NBVCheckpointCallback       — checkpoint with metadata JSON
    NBVRewardComponentsCallback — tracks each reward term separately
    NBVVideoRecorderCallback    — records evaluation episodes as .mp4

Factory:
    make_callback_list(config, eval_env) → SB3 CallbackList

Usage (manual):
    from nbv_callbacks import make_callback_list
    cbs = make_callback_list(config, eval_env=my_env)
    agent.model.learn(..., callback=cbs)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.logger import Figure


# ──────────────────────────────────────────────────────────────────────────────
# 1. Training Metrics Callback
# ──────────────────────────────────────────────────────────────────────────────

class NBVTrainingMetricsCallback(BaseCallback):
    """
    Logs SAC training internals every `log_freq` steps:
      - Entropy coefficient (alpha)
      - Actor / critic losses
      - Mean episode reward and length
      - Steps per second
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq   = log_freq
        self._t_start   = time.time()
        self._ep_rewards: List[float] = []
        self._ep_lengths: List[int]   = []

    def _on_step(self) -> bool:
        # Accumulate episode info from the last step
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done and "episode" in self.locals.get("infos", [{}])[i]:
                    ep = self.locals["infos"][i]["episode"]
                    self._ep_rewards.append(ep.get("r", 0.0))
                    self._ep_lengths.append(ep.get("l", 0))

        if self.n_calls % self.log_freq == 0:
            elapsed = time.time() - self._t_start
            sps = self.n_calls / max(elapsed, 1e-6)
            self.logger.record("train/steps_per_second", sps)

            if self._ep_rewards:
                self.logger.record("train/mean_ep_reward",
                                   float(np.mean(self._ep_rewards[-100:])))
                self.logger.record("train/mean_ep_length",
                                   float(np.mean(self._ep_lengths[-100:])))

            # Log SAC-specific values if they exist in the model
            model = self.model
            if hasattr(model, "ent_coef_tensor"):
                self.logger.record("train/alpha",
                                   float(model.ent_coef_tensor.item()))
            if hasattr(model, "actor") and hasattr(model, "critic"):
                self.logger.record(
                    "train/replay_buffer_size",
                    model.replay_buffer.size() if model.replay_buffer is not None else 0,
                )

            self.logger.dump(self.num_timesteps)

        return True

    def _on_training_start(self) -> None:
        self._t_start = time.time()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Reward Components Callback
# ──────────────────────────────────────────────────────────────────────────────

class NBVRewardComponentsCallback(BaseCallback):
    """
    Tracks each reward component separately.

    Per-step behaviour:
      - Logs seg_confidence to TensorBoard AT EVERY STEP (not averaged).
      - Also writes each step's metrics to a CSV file for offline analysis.

    Averaged over a sliding window of 500 steps:
      - seg_confidence, collision rate, no_segment rate, complete_confidence

    The env's step() info dict must contain:
        seg_confidence, collision, no_segment, complete_confidence, frame_count
    """

    def __init__(
        self,
        log_freq: int = 100,
        csv_path: str = "logs/step_confidence.csv",
        verbose:  int = 0,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None

        self._buffers: Dict[str, List[float]] = {
            "seg_confidence":      [],
            "collision":           [],
            "no_segment":          [],
            "complete_confidence": [],
            "frame_count":         [],
        }

    def _on_training_start(self) -> None:
        import csv
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self._csv_file   = open(self.csv_path, "w", newline="", buffering=1)
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["global_step", "frame_count", "seg_confidence",
             "collision", "no_segment", "complete_confidence"]
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            seg_conf  = info.get("seg_confidence",     None)
            collision = info.get("collision",          None)
            no_seg    = info.get("no_segment",         None)
            comp      = info.get("complete_confidence",None)
            fc        = info.get("frame_count",        None)

            # ── Per-step TensorBoard log for seg_confidence
            if seg_conf is not None:
                self.logger.record_mean("step/seg_confidence", float(seg_conf))
            if fc is not None:
                self.logger.record_mean("step/frame_count", float(fc))

            # ── CSV (every step)
            if self._csv_writer is not None and seg_conf is not None:
                self._csv_writer.writerow([
                    self.num_timesteps,
                    fc     if fc     is not None else "",
                    seg_conf,
                    collision if collision is not None else "",
                    no_seg    if no_seg    is not None else "",
                    comp      if comp      is not None else "",
                ])

            # ── Buffers for sliding-window averages
            for key, val in [
                ("seg_confidence",      seg_conf),
                ("collision",           collision),
                ("no_segment",          no_seg),
                ("complete_confidence", comp),
                ("frame_count",         fc),
            ]:
                if val is not None:
                    self._buffers[key].append(float(val))

        if self.n_calls % self.log_freq == 0:
            for key, vals in self._buffers.items():
                if vals:
                    self.logger.record(
                        f"reward_components/{key}",
                        float(np.mean(vals[-500:])),
                    )
            self.logger.dump(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Checkpoint Callback with Metadata
# ──────────────────────────────────────────────────────────────────────────────

class NBVCheckpointCallback(BaseCallback):
    """
    Saves model.zip + a metadata.json with config snapshot and training stats
    every `save_freq` steps.
    """

    def __init__(
        self,
        save_freq: int,
        save_dir: str,
        config_path: str,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq   = save_freq
        self.save_dir    = Path(save_dir)
        self.config_path = config_path
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps
            stem = f"nbv_sac_step_{step}"
            model_path = self.save_dir / stem
            self.model.save(str(model_path))

            # Write metadata
            meta = {
                "timestep":    step,
                "n_calls":     self.n_calls,
                "config_path": str(self.config_path),
                "saved_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            meta_path = self.save_dir / f"{stem}_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2))

            if self.verbose:
                print(f"[NBVCheckpoint] Saved → {model_path}.zip")

        return True


# ──────────────────────────────────────────────────────────────────────────────
# 4. Evaluation Callback (extends SB3 EvalCallback)
# ──────────────────────────────────────────────────────────────────────────────

class NBVEvaluationCallback(EvalCallback):
    """
    Periodic evaluation callback that additionally:
      - Logs per-component reward stats during evaluation episodes
      - Saves the best model by default
    Inherits from SB3 EvalCallback — see SB3 docs for full parameter list.
    """

    def __init__(
        self,
        eval_env,
        eval_freq:     int = 5000,
        n_eval_episodes: int = 10,
        best_model_path: str = "checkpoints/best_model",
        verbose: int = 1,
        **kwargs,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            best_model_save_path=best_model_path,
            log_path=None,
            deterministic=True,
            render=False,
            verbose=verbose,
            **kwargs,
        )
        self._eval_component_rewards: Dict[str, List[float]] = {}

    def _on_step(self) -> bool:
        result = super()._on_step()
        # Last evaluation reward logged by parent as "eval/mean_reward"
        return result


# ──────────────────────────────────────────────────────────────────────────────
# 5. Early Stopping Callback
# ──────────────────────────────────────────────────────────────────────────────

class NBVEarlyStoppingCallback(BaseCallback):
    """
    Stops training if mean evaluation reward stops improving for
    `patience` evaluation intervals.
    Requires NBVEvaluationCallback to be in the same CallbackList.
    """

    def __init__(self, patience: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.patience      = patience
        self._best_reward  = -np.inf
        self._no_improve   = 0

    def _on_step(self) -> bool:
        # Read the latest eval mean reward from the logger if available
        last_mean = self.logger.name_to_value.get("eval/mean_reward", None)
        if last_mean is not None:
            if last_mean > self._best_reward:
                self._best_reward = last_mean
                self._no_improve  = 0
            else:
                self._no_improve += 1
                if self.verbose:
                    print(f"[EarlyStopping] No improvement for "
                          f"{self._no_improve}/{self.patience} eval intervals.")
            if self._no_improve >= self.patience:
                print("[EarlyStopping] Stopping training.")
                return False  # signals SB3 to stop
        return True


# ──────────────────────────────────────────────────────────────────────────────
# 6. Video Recorder Callback
# ──────────────────────────────────────────────────────────────────────────────

class NBVVideoRecorderCallback(BaseCallback):
    """
    Records evaluation episodes as .mp4 files using imageio.
    Requires `imageio` and `imageio[ffmpeg]` to be installed:
        pip install imageio imageio[ffmpeg]

    Only records if the eval env supports render(mode='rgb_array').
    """

    def __init__(
        self,
        eval_env,
        video_freq: int   = 50_000,
        video_length: int = 200,
        video_dir: str    = "logs/videos/",
        verbose: int      = 0,
    ):
        super().__init__(verbose)
        self.eval_env    = eval_env
        self.video_freq  = video_freq
        self.video_length = video_length
        self.video_dir   = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.video_freq == 0:
            self._record_episode()
        return True

    def _record_episode(self):
        try:
            import imageio
        except ImportError:
            warnings.warn("imageio not installed. pip install imageio imageio[ffmpeg]")
            return

        frames = []
        obs, _ = self.eval_env.reset()
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.eval_env.step(action)
            frame = self.eval_env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                break

        if frames:
            step    = self.num_timesteps
            outpath = self.video_dir / f"nbv_step_{step}.mp4"
            imageio.mimwrite(str(outpath), frames, fps=10)
            if self.verbose:
                print(f"[VideoRecorder] Saved → {outpath}")


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def make_callback_list(config, eval_env=None) -> CallbackList:
    """
    Build a CallbackList from config.yaml settings.

    Args:
        config   : NBVConfig instance
        eval_env : Gymnasium env for evaluation (NBVMockEnv or real env)

    Returns:
        SB3 CallbackList
    """
    from nbv_model import NBVConfig  # local import to avoid circular

    cb_cfg   = config.callbacks_cfg
    t_cfg    = config.training_cfg
    log_freq = t_cfg.get("log_freq", 100)
    ckpt_dir = t_cfg.get("checkpoint_dir", "checkpoints/")
    cbs: List[BaseCallback] = []

    # ── Always on ──────────────────────────────────────────────────────
    cbs.append(NBVTrainingMetricsCallback(log_freq=log_freq))
    cbs.append(NBVRewardComponentsCallback(log_freq=log_freq))
    cbs.append(NBVCheckpointCallback(
        save_freq=t_cfg.get("save_freq", 10_000),
        save_dir=ckpt_dir,
        config_path=config.config_path,
    ))

    # ── Evaluation ─────────────────────────────────────────────────────
    if eval_env is not None:
        best_path = os.path.join(ckpt_dir, "best_model")
        cbs.append(NBVEvaluationCallback(
            eval_env=eval_env,
            eval_freq=t_cfg.get("eval_freq", 5_000),
            n_eval_episodes=t_cfg.get("eval_episodes", 10),
            best_model_path=best_path if cb_cfg.get("save_best", True) else None,
        ))

    # ── Early stopping ─────────────────────────────────────────────────
    if cb_cfg.get("early_stopping", False):
        cbs.append(NBVEarlyStoppingCallback(
            patience=cb_cfg.get("early_stopping_patience", 50),
        ))

    # ── Video recording ────────────────────────────────────────────────
    if cb_cfg.get("render_video", False) and eval_env is not None:
        cbs.append(NBVVideoRecorderCallback(
            eval_env=eval_env,
            video_freq=cb_cfg.get("video_freq", 50_000),
            video_length=cb_cfg.get("video_length", 200),
            video_dir=os.path.join(t_cfg.get("log_dir", "logs/"), "videos/"),
        ))

    # ── Weights & Biases ───────────────────────────────────────────────
    if cb_cfg.get("use_wandb", False):
        try:
            from wandb.integration.sb3 import WandbCallback
            import wandb
            wandb.init(
                project=cb_cfg.get("wandb_project", "nbv-sac"),
                entity=cb_cfg.get("wandb_entity", None) or None,
                sync_tensorboard=True,
                config=config._cfg,
            )
            cbs.append(WandbCallback(
                gradient_save_freq=1000,
                verbose=2,
            ))
        except ImportError:
            print("[Callbacks] wandb not installed — skipping W&B callback.")

    return CallbackList(cbs)
