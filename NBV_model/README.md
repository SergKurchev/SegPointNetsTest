# NBV SAC Model

**Next Best View** via Soft Actor-Critic (SAC).
An RGB-D camera is mounted on a 3-joint robot arm.
The agent learns to choose the next camera pose to maximise scene coverage and segmentation quality.

**Algorithm:** SAC (Stable-Baselines3) | **Env:** Isaac Sim (mock included for offline testing)

---

## Repository Structure

```text
NBV_model/
├── README.md           — this file
├── pyproject.toml      — Python package & dependencies
├── config.yaml         — all hyperparameters
├── nbv_model.py        — NBVAgent, perception encoders, NBVMockEnv
├── nbv_callbacks.py    — SB3 callbacks (metrics, checkpoint, eval, video)
└── setup_odin.py       — installs ODIN backbone (clone + deps + weights)
```

---

## MDP Definition

### Episode Structure

Each episode represents one scene observation session.
The agent always **starts with a single frame** and accumulates up to 10:

```text
reset()  →  frame_count = 1   (first RGB-D frame captured)
step(a1) →  frame_count = 2   (agent moved to pose a1, captured frame 2)
step(a2) →  frame_count = 3
...
step(a9) →  frame_count = 10  → episode ends (done=True)
```

Frames at indices `>= frame_count` are **zero-padded** and masked out in the network.

---

### State Space

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `frames` | `(10, H, W, 4)` | `uint8` | RGB-D history, zero-padded. H=480, W=640 (ScanNet) |
| `frame_count` | `(1,)` | `int32` | Number of valid frames in this step (1–10) |
| `camera_poses` | `(10, 7)` | `float32` | Camera pose per frame: x, y, z, qw, qx, qy, qz |
| `joint_positions` | `(3, 3)` | `float32` | World-frame 3D position of each of the 3 arm joints |
| `seg_confidence` | `(10,)` | `float32` | Segmentation model confidence score per frame (0–1), zero for padded frames |
| `collision_flags` | `(10,)` | `float32` | 1 if the arm/camera collided during that capture, else 0 |

**Frame-indexed arrays** (`frames`, `camera_poses`, `seg_confidence`, `collision_flags`)
are always of size 10 but only indices `0 .. frame_count-1` contain valid data.

---

### Action Space

| Shape | Range | Description |
|-------|-------|-------------|
| `(7,)` | `[-1, 1]` | Next camera pose: x, y, z, qw, qx, qy, qz (normalised). The environment maps this to real-world coordinates. |

---

### Reward Function

$$R = +w_1 \cdot \hat{c}_\text{seg} \;-\; w_2 \cdot \mathbb{1}[\text{collision}] \;-\; w_3 \cdot \mathbb{1}[\text{no segments}] \;+\; w_4 \cdot \hat{c}_\text{complete}$$

| Term | Default weight | Trigger |
|------|---------------|---------|
| Segmentation confidence | **+10** per unit | Scaled by the confidence of the co-running segmentation model |
| Collision penalty | **−20** | Arm or camera hit any scene object |
| No-segment penalty | **−100** | The segmentation model found 0 instances across all frames |
| Completeness confidence | **+20** per unit | Model's certainty that no instances are hidden / unseen |

All weights are configurable in `config.yaml` under `reward:`.

---

## Perception Backbone

Set `perception.mode` in `config.yaml`.

### `"cnn"` — Direct RGB-D

Each of the N frames is processed independently by a CNN, then merged:

```text
(N, H, W, 4) → [CNN per frame] → (N, 512)
             → [Temporal Attention / GRU / Mean-pool] → (512,)
```

CNN options (`perception.cnn.backbone`): `efficientnet_v2_s` ✅, `resnet50`, `mobilenet_v3_large`.
The first conv layer is patched to accept 4 input channels (RGB + D).

### `"pointcloud"` — 3D Point Cloud

Depth frames are unprojected and merged into a single 3D point cloud, then encoded:

```text
(N, H, W, 4) + camera_poses → [Unproject + merge] → (max_points, 6) [xyz+rgb]
                             → [PointNet++ or ODIN]  → (512,)
```

Backend options (`perception.pointcloud.backend`):

| Backend | Deps | Notes |
|---------|------|-------|
| `pointnet2` | None (pure PyTorch) | Built-in 3-layer PointNet++, works out of the box |
| `odin` | `setup_odin.py --all` | ODIN 3D backbone (frozen) + NBV adapter head |

---

## Installation

### Base (required)

```bash
pip install -e .
# or without editable install:
pip install stable-baselines3[extra]>=2.3.0 gymnasium torch torchvision pyyaml numpy tqdm
```

### Optional extras

```bash
pip install -e ".[video]"   # episode video recording
pip install -e ".[wandb]"   # Weights & Biases logging
pip install -e ".[all]"     # video + wandb (excludes ODIN)
```

### ODIN backend (optional, heavy)

```bash
# One-shot full setup:
python setup_odin.py --all

# Step-by-step:
python setup_odin.py --install          # clone repo + install Detectron2 / PyTorch3D
python setup_odin.py --download-weights # download ScanNet200 checkpoint (~500 MB)
python setup_odin.py --verify           # sanity check
```

Then in `config.yaml`:

```yaml
perception:
  mode: "pointcloud"
  pointcloud:
    backend: "odin"
    odin:
      checkpoint: "weights/odin_scannet200.pth"
```

---

## Usage

### Smoke test (no environment needed)

```bash
python nbv_model.py --test
```

### Train on MockEnv

```bash
python nbv_model.py --train --steps 1000   # quick sanity run
python nbv_model.py --train                # full run (from config)
```

### From Python

```python
from nbv_model import NBVAgent, NBVMockEnv, NBVConfig

agent = NBVAgent("config.yaml")           # uses MockEnv by default
agent.train(total_timesteps=1_000_000)
agent.save()                              # → checkpoints/nbv_agent_final.zip

# Inference
obs, _ = agent.env.reset()
action, _ = agent.predict(obs, deterministic=True)
# action: np.ndarray, shape (7,)  — normalised next camera pose
```

---

## Isaac Sim Integration

### Requirements for the Isaac Sim environment

The environment class must:

1. **Inherit from `gymnasium.Env`** and implement `reset()`, `step()`, `render()`.
2. **Expose `observation_space`** as `gymnasium.spaces.Dict` with exactly these keys and shapes:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

observation_space = spaces.Dict({
    "frames":          spaces.Box(0, 255, (10, 480, 640, 4), dtype=np.uint8),
    "frame_count":     spaces.Box(1, 10, (1,), dtype=np.int32),
    "camera_poses":    spaces.Box(-np.inf, np.inf, (10, 7), dtype=np.float32),
    "joint_positions": spaces.Box(-np.inf, np.inf, (3, 3), dtype=np.float32),
    "seg_confidence":  spaces.Box(0.0, 1.0, (10,), dtype=np.float32),
    "collision_flags": spaces.Box(0.0, 1.0, (10,), dtype=np.float32),
})
action_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
```

3. **`reset()`** must:
   - Capture the first RGB-D frame and return `frame_count = 1`.
   - Zero-pad all frame-indexed arrays at indices `1..9`.
   - Record the initial camera pose and joint positions.
   - Start the segmentation co-model and return its confidence for frame 0.

4. **`step(action)`** must:
   - Decode the normalised action `(7,)` to a real-world camera pose using `pos_min`/`pos_max` from config.
   - Move the arm to the new pose (check collision before/after).
   - Capture the new RGB-D frame.
   - Query the segmentation model for confidence score.
   - Append new frame data at index `frame_count` and increment `frame_count`.
   - Return `done = True` when `frame_count == 10`.
   - Return `info` dict with keys: `seg_confidence`, `collision`, `no_segment`,
     `complete_confidence`, `frame_count`.

5. **Collision detection** must distinguish between:
   - Arm–object collision (triggers `collision` in reward).
   - Arm–arm self-collision (optional but recommended to penalise separately).

### Minimal Isaac Sim example

```python
# isaac_nbv_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Isaac Sim / OmniGym imports (adjust to your Isaac version)
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.core.utils.stage import add_reference_to_stage


class IsaacNBVEnv(gym.Env):
    """Minimal skeleton — fill in Isaac Sim API calls."""

    metadata = {"render_modes": ["rgb_array"]}

    MAX_FRAMES = 10
    H, W = 480, 640

    def __init__(self, config, segmentation_model):
        super().__init__()
        self.config   = config
        self.seg_model = segmentation_model   # your co-running seg model

        self.observation_space = spaces.Dict({
            "frames":          spaces.Box(0, 255, (self.MAX_FRAMES, self.H, self.W, 4), dtype=np.uint8),
            "frame_count":     spaces.Box(1, self.MAX_FRAMES, (1,), dtype=np.int32),
            "camera_poses":    spaces.Box(-np.inf, np.inf, (self.MAX_FRAMES, 7), dtype=np.float32),
            "joint_positions": spaces.Box(-np.inf, np.inf, (3, 3), dtype=np.float32),
            "seg_confidence":  spaces.Box(0.0, 1.0, (self.MAX_FRAMES,), dtype=np.float32),
            "collision_flags": spaces.Box(0.0, 1.0, (self.MAX_FRAMES,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)

        # Internal buffers
        self._frames      = np.zeros((self.MAX_FRAMES, self.H, self.W, 4), dtype=np.uint8)
        self._cam_poses   = np.zeros((self.MAX_FRAMES, 7), dtype=np.float32)
        self._seg_conf    = np.zeros(self.MAX_FRAMES, dtype=np.float32)
        self._coll_flags  = np.zeros(self.MAX_FRAMES, dtype=np.float32)
        self._frame_count = 0

    def _capture_rgbd(self) -> np.ndarray:
        """Call Isaac Sim camera sensor → (H, W, 4) uint8."""
        raise NotImplementedError  # fill with Isaac camera API

    def _get_joint_positions(self) -> np.ndarray:
        """Return (3, 3) array of joint world positions."""
        raise NotImplementedError  # fill with Isaac articulation API

    def _get_camera_pose(self) -> np.ndarray:
        """Return (7,) pose: x, y, z, qw, qx, qy, qz."""
        raise NotImplementedError

    def _move_to_pose(self, pose: np.ndarray) -> bool:
        """
        Move arm to target pose. Returns True if collision occurred.
        pose: (7,) normalised action decoded to world coords.
        """
        raise NotImplementedError

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalised [-1,1]^7 → real world pose."""
        pos_min = np.array(self.config.action_cfg.get("pos_min", [-2, -2, 0]))
        pos_max = np.array(self.config.action_cfg.get("pos_max", [2, 2, 2.5]))
        pos = (action[:3] + 1) / 2 * (pos_max - pos_min) + pos_min
        quat = action[3:] / (np.linalg.norm(action[3:]) + 1e-8)
        return np.concatenate([pos, quat])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._frames[:] = 0
        self._cam_poses[:] = 0
        self._seg_conf[:] = 0
        self._coll_flags[:] = 0
        self._frame_count = 1

        # Capture first frame
        self._frames[0]    = self._capture_rgbd()
        self._cam_poses[0] = self._get_camera_pose()
        conf, _            = self.seg_model.infer(self._frames[0])
        self._seg_conf[0]  = float(conf)

        return self._make_obs(), {}

    def step(self, action: np.ndarray):
        target_pose = self._decode_action(action)
        collision   = self._move_to_pose(target_pose)

        n = self._frame_count
        if n < self.MAX_FRAMES:
            self._frames[n]     = self._capture_rgbd()
            self._cam_poses[n]  = self._get_camera_pose()
            conf, n_segs        = self.seg_model.infer(self._frames[n])
            self._seg_conf[n]   = float(conf)
            self._coll_flags[n] = float(collision)
            self._frame_count  += 1

        r_cfg     = self.config.reward_cfg
        seg_conf  = float(self._seg_conf[self._frame_count - 1])
        no_seg    = float(n_segs == 0)
        comp_conf, _ = self.seg_model.completeness_score()  # custom call

        reward = (
            + r_cfg["seg_confidence_weight"]    * seg_conf
            - r_cfg["collision_penalty"]        * float(collision)
            - r_cfg["no_segment_penalty"]       * no_seg
            + r_cfg["completeness_conf_weight"] * comp_conf
        )
        done = self._frame_count >= self.MAX_FRAMES
        info = {
            "seg_confidence":      seg_conf,
            "collision":           float(collision),
            "no_segment":          no_seg,
            "complete_confidence": comp_conf,
            "frame_count":         self._frame_count,
        }
        return self._make_obs(), reward, done, False, info

    def _make_obs(self) -> dict:
        n = self._frame_count
        return {
            "frames":          self._frames.copy(),
            "frame_count":     np.array([n], dtype=np.int32),
            "camera_poses":    self._cam_poses.copy(),
            "joint_positions": self._get_joint_positions(),
            "seg_confidence":  self._seg_conf.copy(),
            "collision_flags": self._coll_flags.copy(),
        }

    def render(self):
        return self._frames[self._frame_count - 1, :, :, :3]
```

### Plugging into NBVAgent

```python
from nbv_model import NBVAgent
from isaac_nbv_env import IsaacNBVEnv
from my_seg_model import SegmentationModel

seg_model = SegmentationModel.load("path/to/checkpoint")
env       = IsaacNBVEnv(config=..., segmentation_model=seg_model)

agent = NBVAgent("config.yaml", env=env)
agent.train(total_timesteps=1_000_000)
```

---

## Logging & Checkpoints

Training produces:

```text
logs/
├── nbv_sac_1/          — TensorBoard logs
│   └── events.out.*
└── step_confidence.csv — per-step: global_step, frame_count, seg_confidence, ...
checkpoints/
├── nbv_sac_step_10000.zip
├── nbv_sac_step_10000_meta.json
└── best_model.zip
```

View TensorBoard:

```bash
tensorboard --logdir logs/
```

Key logged metrics:

| Tag | Description |
|-----|-------------|
| `step/seg_confidence` | Segmentation confidence at each env step |
| `step/frame_count` | Current frame count within episode |
| `reward_components/*` | Per-component reward (sliding mean) |
| `eval/mean_reward` | Mean return over evaluation episodes |
| `train/alpha` | SAC entropy coefficient |
