# %% [markdown]
# # Bayesian ODIN — ScanNet Training on Kaggle
# 
# **Что делает этот ноутбук:**
# 1. Клонирует репозиторий `bayesian_odin` с GitHub
# 2. Устанавливает все зависимости под GPU Kaggle (CUDA 12.1)
# 3. Компилирует кастомные CUDA-ядра
# 4. Подготавливает данные ScanNet из Kaggle Dataset (posed_images)
# 5. Запускает обучение с Bayesian головой и логированием метрик
# 
# **Датасет Kaggle:** `tiantiansyrinx1102/scannet-data` — папка `posed_images/` со структурой `scene*/color/`, `depth/`, `pose/`
# 
# > ⚠️ После ячейки 1 (установка) нужно **перезапустить ядро** (Kernel → Restart).

# %% [markdown]
# ## Ячейка 1 — Установка зависимостей
# > ⚠️ После выполнения этой ячейки **перезапустите ядро** (Kernel → Restart & Run All начиная с ячейки 2).

# %%
"""
Ячейка 1 — Установка зависимостей (v4 — точная копия рабочего оригинала)

Основан на проверенном рабочем коде из strawpick-segpointnet-odin.ipynb.
Единственное отличие: клонируем наш репозиторий bayesian_odin вместо ayushjain1144/odin.

⚠️  После выполнения этой ячейки обязательно ПЕРЕЗАПУСТИТЕ ЯДРО.
    Kernel → Restart (затем запускайте с ячейки 2.)
"""

import os
import subprocess

REPO_URL = "https://github.com/SergKurchev/bayesian_odin.git"
REPO_DIR = "/kaggle/working/bayesian_odin"

# --------------------------------------------------------------------------
# 1. PyTorch 2.2.0 + CUDA 12.1  (явная фиксация версии — ключ к успеху)
# --------------------------------------------------------------------------
os.system("pip install torch==2.2.0 torchvision==0.17.0 "
          "--index-url https://download.pytorch.org/whl/cu121")

os.system("pip install torch-scatter "
          "-f https://data.pyg.org/whl/torch-2.2.0+cu121.html")

# --------------------------------------------------------------------------
# 2. NumPy < 2 (обязательно до остальных пакетов)
# --------------------------------------------------------------------------
os.system("pip install 'numpy<2' --force-reinstall")
os.system("pip install 'Pillow>=10.2.0'")

# --------------------------------------------------------------------------
# 3. Клонирование репозитория + чистка конфликтов в requirements.txt
# --------------------------------------------------------------------------
if not os.path.isdir(REPO_DIR):
    os.system(f"git clone --depth=1 {REPO_URL} {REPO_DIR}")
else:
    print(f"✓ Репозиторий уже склонирован: {REPO_DIR}")

os.chdir(REPO_DIR)
print(f"Рабочая директория: {os.getcwd()}")

os.system("sed -i 's/pyyaml==5.3.1/pyyaml>=5.4.1/gi' requirements.txt")
os.system("sed -i '/detectron2/d' requirements.txt")
os.system("sed -i '/pytorch3d/d' requirements.txt")
os.system("sed -i '/ai2thor/d' requirements.txt")
os.system("sed -i '/prior/d' requirements.txt")

os.system("pip install -r requirements.txt")
os.system("pip install ninja fvcore iopath")

# --------------------------------------------------------------------------
# 4. Detectron2 + PyTorch3D из исходников
#    (работает только с torch==2.2.0, поэтому шаг 1 критичен)
# --------------------------------------------------------------------------
os.system("pip install git+https://github.com/facebookresearch/detectron2.git")
os.system("FORCE_CUDA=1 pip install "
          "git+https://github.com/facebookresearch/pytorch3d.git")

# --------------------------------------------------------------------------
# 5. Кастомные CUDA-ядра pointops2
#    TORCH_CUDA_ARCH_LIST: T4=7.5, P100=6.0, V100=7.0, A100=8.0
#    Перечисляем все популярные Kaggle GPU чтобы не угадывать
# --------------------------------------------------------------------------
os.chdir(os.path.join(REPO_DIR, "libs", "pointops2"))
os.system("rm -rf build dist *.egg-info")
os.system('TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6" '
          "python setup.py install --user")
os.chdir(REPO_DIR)

# --------------------------------------------------------------------------
# 6. Deformable attention CUDA-ядра (есть в нашем репо, нет в оригинале)
# --------------------------------------------------------------------------
_ops_dir = os.path.join(REPO_DIR, "odin", "modeling", "pixel_decoder", "ops")
if os.path.isdir(_ops_dir):
    os.chdir(_ops_dir)
    os.system('TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6" '
              "python setup.py build_ext --inplace")
    os.chdir(REPO_DIR)
    print("✓ pixel_decoder ops скомпилированы")
else:
    print(f"⚠️  {_ops_dir} не найдена, пропускаем")

# --------------------------------------------------------------------------
# 7. Скачивание весов модели (ResNet и Swin backbones)
# --------------------------------------------------------------------------
os.makedirs(os.path.join(REPO_DIR, "weights"), exist_ok=True)

# Pretrained COCO weights (стартовые для обучения)
os.system("wget -nc -q "
          "https://huggingface.co/katefgroup/odin/resolve/main/m2f_coco.pkl "
          f"-O {REPO_DIR}/weights/m2f_coco.pkl")

# Pretrained ScanNet weights (для инференса / файн-тюнинга)
os.system("wget -nc -q "
          "https://huggingface.co/katefgroup/odin/resolve/main/"
          "scannet_swin_semantic_77.8_64k_2k.pth "
          f"-O {REPO_DIR}/weights/scannet_swin.pth")

os.system("wget -nc -q "
          "https://huggingface.co/katefgroup/odin/resolve/main/m2f_coco_swin.pkl "
          f"-O {REPO_DIR}/weights/m2f_coco_swin.pkl")

print("✅ Веса загружены.")

# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("✅ ВСЕ ЗАВИСИМОСТИ УСТАНОВЛЕНЫ")
print()
print("🔴 ОБЯЗАТЕЛЬНО ПЕРЕЗАПУСТИТЕ ЯДРО:")
print("   Kernel → Restart")
print()
print("   После перезапуска запустите ячейки 2-11 (НЕ ячейку 1 повторно).")
print("=" * 60)


# %% [markdown]
# ## Ячейка 2 — Конфигурация
# > Настройте параметры под ваш эксперимент.

# %%
import sys, os, json

# ── Пути ──────────────────────────────────────────────────────────────────────
REPO_DIR  = "/kaggle/working/bayesian_odin"
DATA_ROOT = "/kaggle/input/datasets/tiantiansyrinx1102/scannet-data/scannet/posed_images"
# Папка для обработанных данных (запись разрешена только в /kaggle/working)
PROC_ROOT = "/kaggle/working/scannet_proc"

# Добавляем репозиторий в PYTHONPATH до любых импортов из него
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

NOTEBOOK_CONFIG = {
    # ── Paths ────────────────────────────────────────────────
    "DATA_ROOT":            PROC_ROOT,
    "REPO_ROOT":            REPO_DIR,
    "OUTPUT_DIR":           "/kaggle/working/output/scannet",
    "RESULTS_DIR":          "/kaggle/working/results",
    "RESULTS_CSV":          "results.csv",

    # ── Training ─────────────────────────────────────────────
    # Kaggle T4 = 16 GB VRAM → ResNet backbone, маленький батч
    "BACKBONE":             "resnet",   # "resnet" | "swin" (swin требует ≥24 GB)
    "NUM_GPUS":             1,
    "IMS_PER_BATCH":        2,
    "BASE_LR":              1e-4,
    "MAX_ITER":             100000,
    "CHECKPOINT_PERIOD":    4000,
    "EVAL_PERIOD":          4000,
    "DIST_URL":             "tcp://127.0.0.1:18473",
    "APPROX_EPOCH_ITERS":   5000,

    # ── Bayesian head ─────────────────────────────────────────
    "BAYESIAN_HEAD":        True,
    "BAYESIAN_KL_WEIGHT":   1e-4,
    "BAYESIAN_PRIOR_STD":   1.0,
    "BAYESIAN_INIT_RHO":    -3.0,

    # ── Pretrained weights ────────────────────────────────────
    # Оставьте пустым — веса скачаются автоматически из HuggingFace
    "MODEL_WEIGHTS":        "",

    # ── Input ────────────────────────────────────────────────
    # Уменьшено для T4: 25→15, IMAGE_SIZE 256→192
    "SAMPLING_FRAME_NUM":   15,
    "FRAME_LEFT":           7,
    "FRAME_RIGHT":          7,
    "IMAGE_SIZE":           192,
    "NUM_CLASSES":          20,

    # ── Logging / Plotting intervals ─────────────────────────
    "LOG_CSV_EVERY":         100,
    "PLOT_LR_CURVE_EVERY":   500,
    "LOG_QUALITY_EVERY":     4000,
    "PLOT_QUALITY_EVERY":    4000,
    "VIZ_EVERY":             4000,
    "NUM_VIZ_SAMPLES":       3,
}

CFG = NOTEBOOK_CONFIG
os.makedirs(CFG["RESULTS_DIR"], exist_ok=True)
os.makedirs(CFG["OUTPUT_DIR"], exist_ok=True)
os.makedirs(CFG["DATA_ROOT"], exist_ok=True)
print("Active config:")
print(json.dumps(CFG, indent=2, default=str))

# %% [markdown]
# ## Ячейка 3 — Подготовка данных
# 
# Датасет `tiantiansyrinx1102/scannet-data` имеет плоскую структуру `posed_images/scene*/`.
# Нам нужно:
# 1. Реорганизовать под стандарт ODIN (`scene*/color/, depth/, pose/`)
# 2. Скачать 3D mesh-индексы с HuggingFace
# 3. Создать симлинки для Detectron2
# 4. Сгенерировать COCO JSON аннотации

# %%
import subprocess, shutil, glob
import os

DATA_ROOT = CFG["DATA_ROOT"]
PROC_ROOT = DATA_ROOT
INPUT_POSED = "/kaggle/input/datasets/tiantiansyrinx1102/scannet-data/scannet/posed_images"
REPO_DIR = CFG["REPO_ROOT"]

def run(cmd, **kw):
    print(f">>> {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kw)

# ── Шаг 1: Преобразование flat структуры posed_images → color/depth/pose ──────
# posed_images/scene0000_00/00000.jpg, 00000.png, 00000.txt
# → frames_square_highres/scene0000_00/color/00000.jpg
#                                       depth/00000.png
#                                       pose/ 00000.txt

FRAMES_DIR = os.path.join(PROC_ROOT, "frames_square_highres")
SCENES_DONE_FLAG = os.path.join(PROC_ROOT, ".scenes_reorganized")

if not os.path.exists(SCENES_DONE_FLAG):
    print("Реорганизуем структуру данных posed_images → color/depth/pose ...")
    scenes = sorted(d for d in os.listdir(INPUT_POSED)
                    if os.path.isdir(os.path.join(INPUT_POSED, d)) and d.startswith("scene"))
    print(f"Найдено {len(scenes)} сцен")
    for scene in scenes:
        src = os.path.join(INPUT_POSED, scene)
        dst_color = os.path.join(FRAMES_DIR, scene, "color")
        dst_depth = os.path.join(FRAMES_DIR, scene, "depth")
        dst_pose  = os.path.join(FRAMES_DIR, scene, "pose")
        os.makedirs(dst_color, exist_ok=True)
        os.makedirs(dst_depth, exist_ok=True)
        os.makedirs(dst_pose,  exist_ok=True)
        for fname in os.listdir(src):
            fpath = os.path.join(src, fname)
            if fname.endswith(".jpg") or fname.endswith(".png") and not os.path.exists(
                    os.path.join(src, fname.replace(".png", ".jpg"))):
                # color: .jpg файлы
                if fname.endswith(".jpg"):
                    os.symlink(fpath, os.path.join(dst_color, fname))
                # depth: .png (без соответствующего .jpg)
                elif fname.endswith(".png"):
                    os.symlink(fpath, os.path.join(dst_depth, fname))
            elif fname.endswith(".txt"):
                os.symlink(fpath, os.path.join(dst_pose, fname))
    with open(SCENES_DONE_FLAG, "w") as f:
        f.write("done")
    print(f"✓ Реорганизовано {len(scenes)} сцен → {FRAMES_DIR}")
else:
    scenes = sorted(d for d in os.listdir(FRAMES_DIR) if d.startswith("scene"))
    print(f"✓ Данные уже реорганизованы ({len(scenes)} сцен)")

# ── Шаг 2: Скачать 3D mesh-индексы (train_validation_database.yaml) ──────────
YAML_PATH = os.path.join(PROC_ROOT, "mask3d_processed", "scannet",
                         "train_validation_database.yaml")
if not os.path.exists(YAML_PATH):
    print("\nСкачиваем 3D mesh-индексы с HuggingFace (~1 GB) ...")
    _zip = os.path.join(PROC_ROOT, "mask3d_processed.zip")
    if not os.path.exists(_zip):
        run(f'gdown --fuzzy "https://huggingface.co/katefgroup/odin/resolve/main/mask3d_processed.zip" -O "{_zip}"')
    run(f'unzip -q "{_zip}" -d "{PROC_ROOT}/"')
    print(f"✓ 3D индексы → {YAML_PATH}")
else:
    print(f"✓ 3D индексы уже есть: {YAML_PATH}")

# ── Шаг 3: Окружающие переменные для Detectron2 ───────────────────────────────
os.environ["DETECTRON2_DATASETS"] = PROC_ROOT
os.environ["SCANNET_DATA_DIR"]    = YAML_PATH
os.environ["SCANNET_RGBD_DIR"]    = FRAMES_DIR

# ── Шаг 4: Создать симлинки (Detectron2 ищет датасет по этому имени) ─────────
for alias in [
    "scannet_context_instance_train_20cls_single_highres_100k",
    "scannet_context_instance_val_20cls_single_highres_100k",
    "scannet_context_instance_train_eval_20cls_single_highres_100k",
]:
    link = os.path.join(PROC_ROOT, alias)
    if not os.path.exists(link):
        os.symlink(FRAMES_DIR, link)
        print(f"✓ Симлинк: {alias}")

# ── Шаг 5: Генерация COCO JSON аннотаций ─────────────────────────────────────
_coco_train = os.path.join(PROC_ROOT, "scannet_highres_train.coco.json")
if not os.path.exists(_coco_train):
    print("\nГенерируем COCO JSON (может занять 10-20 мин для всего датасета) ...")
    subprocess.run(
        [sys.executable, "data_preparation/scannet/scannet2coco.py"],
        cwd=REPO_DIR, check=True,
        env={**os.environ, "SCANNET_RGBD_DIR": FRAMES_DIR},
    )
    print("✓ COCO JSON сгенерированы")
else:
    print(f"✓ COCO JSON уже есть: {_coco_train}")

print(f"\nDETECTRON2_DATASETS = {PROC_ROOT}")
print(f"SCANNET_DATA_DIR    = {YAML_PATH}")
print(f"SCANNET_RGBD_DIR    = {FRAMES_DIR}")
print("\n✅ Данные готовы!")

# %% [markdown]
# ## Ячейка 4 — Импорты

# %%
import csv, time, copy, gc, warnings, weakref, itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

try:
    from IPython import get_ipython
    from IPython.display import display, clear_output
    _IN_NOTEBOOK = get_ipython() is not None
except ImportError:
    _IN_NOTEBOOK = False
    def display(x): pass
    def clear_output(**kw): pass

if _IN_NOTEBOOK:
    get_ipython().run_line_magic("matplotlib", "inline")

warnings.filterwarnings("ignore")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# Detectron2
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    DefaultTrainer, AMPTrainer, SimpleTrainer, launch, default_setup,
)
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
import detectron2.utils.comm as comm
from torch.nn.parallel import DistributedDataParallel

# ODIN
from odin import (
    ScannetDatasetMapper, Scannet3DEvaluator, ScannetSemantic3DEvaluator,
    COCOEvaluatorMemoryEfficient, add_maskformer2_video_config,
    add_maskformer2_config, build_detection_train_loader,
    build_detection_test_loader, get_detection_dataset_dicts,
)
from odin.global_vars import SCANNET_LIKE_DATASET

print(f"✓ Torch {torch.__version__}  CUDA: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory // 1024**3
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem} GB)")

# %% [markdown]
# ## Ячейка 5 — Метрики и CSV

# %%
RESULTS_CSV_PATH = os.path.join(CFG["RESULTS_DIR"], CFG["RESULTS_CSV"])
_CSV_HEADER = ["epoch", "global_step", "metric_name", "metric_value"]


def _init_csv(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)


def write_csv_rows(path, rows):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerows(rows)


class MetricsHistory:
    """Accumulates (step, value) pairs per metric for plotting."""

    def __init__(self):
        self._data: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    def log(self, step: int, name: str, value: float):
        self._data[name].append((step, float(value)))

    def steps(self, name: str) -> List[int]:
        return [s for s, _ in self._data.get(name, [])]

    def values(self, name: str) -> List[float]:
        return [v for _, v in self._data.get(name, [])]

    def keys(self) -> List[str]:
        return list(self._data.keys())


_metrics = MetricsHistory()
_init_csv(RESULTS_CSV_PATH)
print(f"✓ CSV → {RESULTS_CSV_PATH}")

# %% [markdown]
# ## Ячейка 6 — Утилиты визуализации

# %%
_VIZ_DIR     = os.path.join(CFG["RESULTS_DIR"], "visualizations")
_LC_DIR      = os.path.join(CFG["RESULTS_DIR"], "learning_curves")
_QUALITY_DIR = os.path.join(CFG["RESULTS_DIR"], "quality_metrics")
for _d in [_VIZ_DIR, _LC_DIR, _QUALITY_DIR]:
    os.makedirs(_d, exist_ok=True)

SCANNET20_COLORS = np.array([
    [174,199,232],[152,223,138],[ 31,119,180],[255,187,120],
    [188,189, 34],[140, 86, 75],[255,152,150],[214, 39, 40],
    [197,176,213],[148,103,189],[196,156,148],[ 23,190,207],
    [247,182,210],[219,219,141],[255,127, 14],[158,218,229],
    [ 44,160, 44],[112,128,144],[227,119,194],[ 82, 84,163],
], dtype=np.uint8)

SCANNET20_NAMES = [
    "wall","floor","cabinet","bed","chair","sofa","table","door",
    "window","bookshelf","picture","counter","desk","curtain",
    "refridgerator","shower curtain","toilet","sink","bathtub","otherfurniture",
]


def _save_show(fig, path):
    plt.savefig(path, dpi=100, bbox_inches="tight")
    if _IN_NOTEBOOK:
        display(fig)
    plt.close(fig)
    return path


def plot_learning_curve(history: MetricsHistory, step: int) -> Optional[str]:
    loss_keys = sorted(k for k in history.keys() if "loss" in k.lower())
    lr_keys   = [k for k in history.keys() if k == "lr"]
    if not loss_keys and not lr_keys:
        return None
    ncols = int(bool(loss_keys)) + int(bool(lr_keys))
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4))
    axes = [axes] if ncols == 1 else list(axes)
    idx = 0
    if loss_keys:
        ax = axes[idx]; idx += 1
        for k in loss_keys:
            s, v = history.steps(k), history.values(k)
            if s:
                ax.plot(s, v, label=k, linewidth=1.2, alpha=0.85)
        ax.set(xlabel="Step", ylabel="Loss", title=f"Training Losses (step {step:,})")
        ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    if lr_keys:
        ax = axes[idx]
        s, v = history.steps("lr"), history.values("lr")
        if s:
            ax.semilogy(s, v, color="crimson", linewidth=1.5)
        ax.set(xlabel="Step", ylabel="LR (log)", title="Learning Rate")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save_show(fig, os.path.join(_LC_DIR, f"lc_{step:07d}.png"))


def plot_quality_metrics(history: MetricsHistory, step: int) -> Optional[str]:
    groups = {
        "Segmentation":   ["AP", "AP25", "AP50"],
        "Classification": ["mIoU", "mAcc", "allAcc"],
        "Panoptic":       ["PQ", "SQ", "RQ"],
    }
    avail = {g: [k for k in ks if history.steps(k)] for g, ks in groups.items()}
    avail = {g: ks for g, ks in avail.items() if ks}
    if not avail:
        return None
    fig, axes = plt.subplots(1, len(avail), figsize=(6 * len(avail), 4))
    axes = [axes] if len(avail) == 1 else list(axes)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for ax, (gname, keys) in zip(axes, avail.items()):
        for c, k in zip(colors, keys):
            s, v = history.steps(k), history.values(k)
            ax.plot(s, v, label=k, color=c, marker="o", markersize=4, linewidth=1.5)
        ax.set(xlabel="Step", ylabel="Score (%)", title=gname, ylim=(0, 105))
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Eval Metrics (step {step:,})", fontsize=12)
    plt.tight_layout()
    return _save_show(fig, os.path.join(_QUALITY_DIR, f"quality_{step:07d}.png"))


def visualize_segmentation(batch: list, outputs: list, step: int) -> List[str]:
    saved = []
    for i in range(min(CFG["NUM_VIZ_SAMPLES"], len(batch))):
        data = batch[i]
        out  = outputs[i] if i < len(outputs) else {}
        img  = data.get("image", None)
        if img is None:
            continue
        t = img if img.dim() == 3 else img[0]
        img_np = t.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np / 255.0 if img_np.max() > 1.5 else img_np).clip(0, 1)
        overlay = img_np.copy()
        instances = (out or {}).get("instances", None)
        if instances is not None and len(instances) > 0:
            try:
                for mi, mask in enumerate(instances.pred_masks.cpu().numpy()):
                    c = SCANNET20_COLORS[mi % len(SCANNET20_COLORS)] / 255.0
                    overlay[mask] = overlay[mask] * 0.4 + c * 0.6
            except Exception:
                pass
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_np); axes[0].set_title("RGB input"); axes[0].axis("off")
        axes[1].imshow(overlay); axes[1].set_title("Predicted masks"); axes[1].axis("off")
        plt.suptitle(f"Segmentation — step {step:,}, sample {i+1}")
        plt.tight_layout()
        path = os.path.join(_VIZ_DIR, f"seg_s{step:07d}_i{i:02d}.png")
        _save_show(fig, path)
        saved.append(path)
    return saved


print("✓ Visualization utilities ready")

# %% [markdown]
# ## Ячейка 7 — Хуки Detectron2

# %%
_QUAL_KEYS = {"AP", "AP25", "AP50", "mIoU", "mAcc", "allAcc", "PQ", "SQ", "RQ"}


class TrainingMetricsHook(HookBase):
    """Log training losses to CSV and plot learning curve periodically."""

    def __init__(self, cfg_nb, history, csv_path):
        self._cfg   = cfg_nb
        self._hist  = history
        self._csv   = csv_path
        self._epoch = cfg_nb["APPROX_EPOCH_ITERS"]

    def after_step(self):
        storage = get_event_storage()
        step = int(storage.iter)
        try:
            latest = {k: float(v)
                      for k, (v, _) in storage.latest().items()
                      if isinstance(v, (int, float))}
        except Exception:
            latest = {}
        for name, val in latest.items():
            self._hist.log(step, name, val)
        if latest and step > 0 and step % self._cfg["LOG_CSV_EVERY"] == 0:
            epoch = step // self._epoch
            rows = [(epoch, step, n, f"{v:.6f}") for n, v in latest.items()]
            write_csv_rows(self._csv, rows)
        if step > 0 and step % self._cfg["PLOT_LR_CURVE_EVERY"] == 0:
            plot_learning_curve(self._hist, step)


class EvalMetricsHook(HookBase):
    """After evaluation: log quality metrics and plot quality charts."""

    def __init__(self, cfg_nb, history, csv_path, eval_period):
        self._cfg    = cfg_nb
        self._hist   = history
        self._csv    = csv_path
        self._period = eval_period
        self._epoch  = cfg_nb["APPROX_EPOCH_ITERS"]
        self._seen   = -1

    def after_step(self):
        storage = get_event_storage()
        step = int(storage.iter)
        if step == 0 or step % self._period != 0 or step == self._seen:
            return
        self._seen = step
        epoch = step // self._epoch
        try:
            latest = {k: float(v)
                      for k, (v, _) in storage.latest().items()
                      if isinstance(v, (int, float))}
        except Exception:
            latest = {}
        qual = {k: v for k, v in latest.items() if k in _QUAL_KEYS}
        if not qual:
            return
        for name, val in qual.items():
            self._hist.log(step, name, val)
        if step % self._cfg["LOG_QUALITY_EVERY"] == 0:
            rows = [(epoch, step, n, f"{v:.6f}") for n, v in qual.items()]
            write_csv_rows(self._csv, rows)
        if step % self._cfg["PLOT_QUALITY_EVERY"] == 0:
            plot_quality_metrics(self._hist, step)


class VisualizationHook(HookBase):
    """Render predicted segmentation masks periodically."""

    def __init__(self, cfg_nb, d2_cfg, get_model_fn):
        self._cfg      = cfg_nb
        self._d2_cfg   = d2_cfg
        self._model_fn = get_model_fn
        self._seen     = -1

    def after_step(self):
        storage = get_event_storage()
        step = int(storage.iter)
        if step == 0 or step % self._cfg["VIZ_EVERY"] != 0 or step == self._seen:
            return
        self._seen = step
        self._render(step)

    def after_train(self):
        self._render(self.trainer.iter)

    def _render(self, step):
        model = self._model_fn()
        was_train = model.training
        model.eval()
        dataset = self._d2_cfg.DATASETS.TEST[0]
        dd = get_detection_dataset_dicts([dataset], proposal_files=None)
        mapper = ScannetDatasetMapper(self._d2_cfg, is_train=False,
                                      dataset_name=dataset, dataset_dict=dd)
        loader = build_detection_test_loader(self._d2_cfg, mapper=mapper, dataset=dd)
        batch, outputs = [], []
        with torch.no_grad():
            for i, data in enumerate(loader):
                if i >= self._cfg["NUM_VIZ_SAMPLES"]:
                    break
                item = data[0] if isinstance(data, list) else data
                batch.append(item)
                try:
                    out = model([item])
                    outputs.append(out[0] if isinstance(out, list) else out)
                except Exception:
                    outputs.append({})
        if was_train:
            model.train()
        visualize_segmentation(batch, outputs, step)


print("✓ Custom hooks defined")

# %% [markdown]
# ## Ячейка 8 — Кастомный Trainer

# %%
class NotebookTrainer(DefaultTrainer):
    """DefaultTrainer + CSV / plotting / visualization hooks."""

    def __init__(self, cfg, cfg_nb):
        super(DefaultTrainer, self).__init__()
        from detectron2.utils.logger import setup_logger
        import logging
        if not logging.getLogger("detectron2").isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model       = self.build_model(cfg)
        optimizer   = self.build_optimizer(cfg, model)
        loader      = self.build_train_loader(cfg)
        find_unused = cfg.MULTI_TASK_TRAINING or cfg.FIND_UNUSED_PARAMETERS
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()],
                find_unused_parameters=find_unused,
            )
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, loader, optimizer
        )
        self.scheduler    = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self)
        )
        self.start_iter = 0
        self.max_iter   = cfg.SOLVER.MAX_ITER
        self.cfg        = cfg
        self.cfg_nb     = cfg_nb
        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        hooks = super().build_hooks()
        get_model = lambda: (
            self._trainer.model.module
            if hasattr(self._trainer.model, "module")
            else self._trainer.model
        )
        hooks += [
            TrainingMetricsHook(self.cfg_nb, _metrics, RESULTS_CSV_PATH),
            EvalMetricsHook(self.cfg_nb, _metrics, RESULTS_CSV_PATH,
                            self.cfg_nb["EVAL_PERIOD"]),
            VisualizationHook(self.cfg_nb, self.cfg, get_model),
        ]
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None, **_kw):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)
        evals = []
        if cfg.TEST.EVAL_3D and cfg.MODEL.DECODER_3D:
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                evals.append(ScannetSemantic3DEvaluator(
                    dataset_name, output_dir=output_folder,
                    eval_sparse=cfg.TEST.EVAL_SPARSE, cfg=cfg))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                evals.append(Scannet3DEvaluator(
                    dataset_name, output_dir=output_folder,
                    eval_sparse=cfg.TEST.EVAL_SPARSE, cfg=cfg))
        return evals

    @classmethod
    def build_train_loader(cls, cfg):
        name = cfg.DATASETS.TRAIN[0]
        if any(s in name for s in SCANNET_LIKE_DATASET):
            dd = get_detection_dataset_dicts(name, proposal_files=None)
            mapper = ScannetDatasetMapper(cfg, is_train=True,
                                          dataset_name=name, dataset_dict=dd)
            return build_detection_train_loader(cfg, mapper=mapper, dataset=dd)
        raise NotImplementedError(f"Unknown dataset: {name}")

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dd = get_detection_dataset_dicts([dataset_name], proposal_files=None)
        mapper = ScannetDatasetMapper(cfg, is_train=False,
                                      dataset_name=dataset_name, dataset_dict=dd)
        return build_detection_test_loader(cfg, mapper=mapper, dataset=dd)

    @classmethod
    def build_optimizer(cls, cfg, model):
        norm_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm, torch.nn.GroupNorm, torch.nn.LayerNorm,
        )
        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        params, memo = [], set()
        for mname, module in model.named_modules():
            for _, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                hp = copy.copy(defaults)
                if "backbone" in mname:
                    hp["lr"] *= cfg.SOLVER.BACKBONE_MULTIPLIER
                if isinstance(module, norm_types):
                    hp["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_NORM
                if isinstance(module, torch.nn.Embedding):
                    hp["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_EMBED
                params.append({"params": [value], **hp})
        opt = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)
        opt = maybe_add_gradient_clipping(cfg, opt)
        return opt

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)


print("✓ NotebookTrainer defined")

# %% [markdown]
# ## Ячейка 9 — Конфиг Detectron2

# %%
import argparse

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
torch.multiprocessing.set_sharing_strategy("file_system")

_CONFIG_MAP = {
    "resnet": "configs/scannet_context/3d.yaml",
    "swin":   "configs/scannet_context/swin_3d.yaml",
}
CONFIG_FILE   = _CONFIG_MAP[CFG["BACKBONE"]]
DATASET_TRAIN = "scannet_context_instance_train_20cls_single_highres_100k"
DATASET_VAL   = "scannet_context_instance_val_20cls_single_highres_100k"

YAML_PATH     = os.environ["SCANNET_DATA_DIR"]


def build_d2_cfg(cfg_nb):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(CONFIG_FILE)

    cfg.DATASETS.TRAIN = (DATASET_TRAIN,)
    cfg.DATASETS.TEST  = (DATASET_VAL,)

    cfg.SOLVER.IMS_PER_BATCH     = cfg_nb["IMS_PER_BATCH"]
    cfg.SOLVER.BASE_LR           = cfg_nb["BASE_LR"]
    cfg.SOLVER.MAX_ITER          = cfg_nb["MAX_ITER"]
    cfg.SOLVER.CHECKPOINT_PERIOD = cfg_nb["CHECKPOINT_PERIOD"]
    cfg.TEST.EVAL_PERIOD         = cfg_nb["EVAL_PERIOD"]

    cfg.INPUT.FRAME_LEFT          = cfg_nb["FRAME_LEFT"]
    cfg.INPUT.FRAME_RIGHT         = cfg_nb["FRAME_RIGHT"]
    cfg.INPUT.SAMPLING_FRAME_NUM  = cfg_nb["SAMPLING_FRAME_NUM"]
    cfg.INPUT.IMAGE_SIZE          = cfg_nb["IMAGE_SIZE"]
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = cfg_nb["NUM_CLASSES"]

    if cfg_nb["MODEL_WEIGHTS"]:
        cfg.MODEL.WEIGHTS = cfg_nb["MODEL_WEIGHTS"]
    # Иначе возьмём m2f_coco.pkl — стандартные pretrained веса из оригинального ODIN
    # Они прописаны в YAML конфиге или скачиваются detectron2 автоматически

    cfg.MODEL.BAYESIAN_HEAD       = cfg_nb["BAYESIAN_HEAD"]
    cfg.MODEL.BAYESIAN_KL_WEIGHT  = cfg_nb["BAYESIAN_KL_WEIGHT"]
    cfg.MODEL.BAYESIAN_PRIOR_STD  = cfg_nb["BAYESIAN_PRIOR_STD"]
    cfg.MODEL.BAYESIAN_INIT_RHO   = cfg_nb["BAYESIAN_INIT_RHO"]

    cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE    = True
    cfg.INPUT.CAMERA_DROP                 = True
    cfg.INPUT.STRONG_AUGS                 = True
    cfg.INPUT.AUGMENT_3D                  = True
    cfg.INPUT.VOXELIZE                    = True
    cfg.INPUT.SAMPLE_CHUNK_AUG            = True
    # Для T4 16 GB уменьшаем TRAIN_NUM_POINTS
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 12544
    cfg.MODEL.CROSS_VIEW_BACKBONE         = True
    cfg.MODEL.PIXEL_DECODER_PANET         = True
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.SKIP_CLASSES                      = "[19, 20]"
    cfg.USE_GHOST_POINTS                  = True
    cfg.MODEL.FREEZE_BACKBONE             = False
    cfg.SOLVER.TEST_IMS_PER_BATCH         = 1
    cfg.SAMPLING_STRATEGY                 = "consecutive"
    cfg.USE_SEGMENTS                      = True
    # Для Kaggle: меньше воркеров чтобы не упасть в OOM по RAM
    cfg.DATALOADER.NUM_WORKERS            = 4
    cfg.DATALOADER.TEST_NUM_WORKERS       = 2
    cfg.MAX_FRAME_NUM                     = -1
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT     = 6.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT     = 15.0
    cfg.USE_WANDB                         = False
    cfg.USE_MLP_POSITIONAL_ENCODING       = True
    cfg.SCANNET_DATA_DIR                  = YAML_PATH
    cfg.OUTPUT_DIR                        = cfg_nb["OUTPUT_DIR"]
    cfg.freeze()
    return cfg


D2_CFG = build_d2_cfg(CFG)

_fake_args = argparse.Namespace(
    config_file=CONFIG_FILE, resume=True, eval_only=False,
    num_gpus=CFG["NUM_GPUS"], num_machines=1, machine_rank=0,
    dist_url=CFG["DIST_URL"], opts=[],
)
default_setup(D2_CFG, _fake_args)
print(f"✓ D2 config ready → {D2_CFG.OUTPUT_DIR}")
print(f"  Backbone:      {CFG['BACKBONE']}")
print(f"  Bayesian head: {D2_CFG.MODEL.BAYESIAN_HEAD}")
print(f"  KL weight:     {D2_CFG.MODEL.BAYESIAN_KL_WEIGHT}")
print(f"  MAX_ITER:      {D2_CFG.SOLVER.MAX_ITER:,}")

# %% [markdown]
# ## Ячейка 10 — Запуск обучения
# 
# > 💡 Kaggle-сессии ограничены ~9 часами. При прерывании используйте `--resume` — тренер автоматически подхватит последний чекпоинт из `OUTPUT_DIR`.

# %%
def _run_training(cfg, cfg_nb):
    trainer = NotebookTrainer(cfg, cfg_nb)
    trainer.resume_or_load(resume=True)
    return trainer.train()


if CFG["NUM_GPUS"] > 1:
    launch(
        _run_training,
        num_gpus_per_machine=CFG["NUM_GPUS"],
        num_machines=1,
        machine_rank=0,
        dist_url=CFG["DIST_URL"],
        args=(D2_CFG, CFG),
    )
else:
    _run_training(D2_CFG, CFG)

print("\n✅ Обучение завершено!")
print(f"  Чекпоинты → {CFG['OUTPUT_DIR']}")
print(f"  CSV метрики → {RESULTS_CSV_PATH}")

# %% [markdown]
# ## Ячейка 11 — Финальные графики

# %%
if not os.path.exists(RESULTS_CSV_PATH):
    print("Нет results.csv — сначала запустите обучение.")
else:
    _final = MetricsHistory()
    with open(RESULTS_CSV_PATH, newline="") as _f:
        for _row in csv.DictReader(_f):
            try:
                _final.log(int(_row["global_step"]),
                           _row["metric_name"],
                           float(_row["metric_value"]))
            except (ValueError, KeyError):
                pass

    print(f"Загружено {len(_final.keys())} серий метрик")

    _last_loss_step = max(_final.steps("total_loss") or [0])
    _lc = plot_learning_curve(_final, _last_loss_step)
    if _lc:
        print(f"  Learning curve → {_lc}")

    _last_qual_step = max(_final.steps("AP") or _final.steps("mIoU") or [0])
    _qp = plot_quality_metrics(_final, _last_qual_step)
    if _qp:
        print(f"  Quality plot   → {_qp}")

    print("\nФинальные значения метрик:")
    for _key in sorted(_final.keys()):
        _vals = _final.values(_key)
        if _vals:
            print(f"  {_key:35s}: {_vals[-1]:.4f}")


