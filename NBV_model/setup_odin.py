"""
setup_odin.py — One-shot ODIN setup script for NBV model.

Downloads and installs the ODIN 3D instance segmentation model:
  1. Verifies CUDA / torch version compatibility
  2. Clones the ODIN repository
  3. Installs required dependencies (Detectron2, PyTorch3D, etc.)
  4. Downloads pretrained model weights
  5. Runs a quick smoke test

Usage:
    python setup_odin.py [--install] [--download-weights] [--verify] [--all]
    python setup_odin.py --all            # full setup in one go
    python setup_odin.py --verify         # check existing installation

The ODIN GitHub repository:
    https://github.com/ayushjain1144/odin
"""

import argparse
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ODIN_REPO_URL = "https://github.com/ayushjain1144/odin.git"
ODIN_REPO_DIR = Path(__file__).parent / "odin_repo"

# Official ODIN checkpoint URLs (update if authors change hosting)
# Check https://github.com/ayushjain1144/odin for the latest links
ODIN_CHECKPOINTS = {
    "odin_scannet200": {
        "url": "https://huggingface.co/ayushjain1144/odin/resolve/main/odin_scannet200.pth",
        "filename": "odin_scannet200.pth",
        "description": "ScanNet200 (200 classes, best for manipulation scenes)",
    },
    "odin_scannet": {
        "url": "https://huggingface.co/ayushjain1144/odin/resolve/main/odin_scannet.pth",
        "filename": "odin_scannet.pth",
        "description": "ScanNet (20 classes)",
    },
}

WEIGHTS_DIR = Path(__file__).parent / "weights"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: str, cwd: Path = None, check: bool = True) -> int:
    """Run a shell command, streaming output."""
    print(f"\n[setup_odin] $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=str(cwd) if cwd else None
    )
    if check and result.returncode != 0:
        print(f"[setup_odin] ERROR: command exited with code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def _pip(*packages: str, extra_args: str = "") -> None:
    """Install packages via pip."""
    pkgs = " ".join(packages)
    _run(f"{sys.executable} -m pip install {extra_args} {pkgs}")


def _print_section(title: str) -> None:
    line = "─" * 60
    print(f"\n{line}\n  {title}\n{line}")


# ---------------------------------------------------------------------------
# Step 1: verify environment
# ---------------------------------------------------------------------------

def verify_environment() -> dict:
    _print_section("Step 1 — Verifying environment")
    import importlib

    info = {}

    # Python
    info["python"] = sys.version
    print(f"  Python  : {sys.version}")

    # PyTorch
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        print(f"  PyTorch : {torch.__version__}")
        print(f"  CUDA    : {torch.version.cuda}  (available: {torch.cuda.is_available()})")
    except ImportError:
        print("  [ERROR] PyTorch not installed. Install it first:")
        print("  https://pytorch.org/get-started/locally/")
        sys.exit(1)

    # Check CUDA version is supported
    if not torch.cuda.is_available():
        print("\n  [WARNING] CUDA not available. ODIN will run on CPU (very slow).")

    # Optional: check if Detectron2 is already installed
    for pkg_name, import_name in [
        ("detectron2", "detectron2"),
        ("pytorch3d",  "pytorch3d"),
        ("open3d",     "open3d"),
    ]:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {pkg_name:<14}: {ver} (already installed)")
            info[pkg_name] = ver
        except ImportError:
            print(f"  {pkg_name:<14}: NOT installed")
            info[pkg_name] = None

    return info


# ---------------------------------------------------------------------------
# Step 2: clone ODIN repository
# ---------------------------------------------------------------------------

def clone_odin():
    _print_section("Step 2 — Cloning ODIN repository")

    if ODIN_REPO_DIR.exists() and (ODIN_REPO_DIR / "odin").exists():
        print(f"  ODIN already cloned at: {ODIN_REPO_DIR}")
        _run("git pull", cwd=ODIN_REPO_DIR, check=False)
        return

    _run(f"git clone --depth 1 {ODIN_REPO_URL} {ODIN_REPO_DIR}")
    print(f"  ✅ Cloned to: {ODIN_REPO_DIR}")


# ---------------------------------------------------------------------------
# Step 3: install dependencies
# ---------------------------------------------------------------------------

def install_dependencies(env_info: dict):
    _print_section("Step 3 — Installing dependencies")

    import torch

    cuda_ver = (torch.version.cuda or "").replace(".", "")   # e.g. "118"
    torch_ver = torch.__version__.split("+")[0]               # e.g. "2.1.0"
    torch_ver_short = "".join(torch_ver.split(".")[:2])       # e.g. "21"

    # ── Core deps ──────────────────────────────────────────────────────
    _pip("ninja", "setuptools>=67", "wheel")
    _pip("open3d", "plyfile", "trimesh", "einops", "timm")
    _pip("scipy", "scikit-learn", "tqdm", "omegaconf")

    # ── Detectron2 (pre-built wheel) ────────────────────────────────────
    if env_info.get("detectron2") is None:
        print("\n  Installing Detectron2 (pre-built wheel)...")
        # Try to find the right pre-built wheel
        d2_index = (
            f"https://dl.fbaipublicfiles.com/detectron2/wheels/"
            f"cu{cuda_ver}/torch{torch_ver_short}/index.html"
        )
        rc = _run(
            f"{sys.executable} -m pip install detectron2 "
            f"--extra-index-url {d2_index}",
            check=False
        )
        if rc != 0:
            print("\n  Pre-built wheel failed. Building from source (slow ~10 min)...")
            _run(
                f"{sys.executable} -m pip install "
                "'git+https://github.com/facebookresearch/detectron2.git'"
            )

    # ── PyTorch3D ───────────────────────────────────────────────────────
    if env_info.get("pytorch3d") is None:
        print("\n  Installing PyTorch3D...")
        # Check for Kaggle / colab pre-built wheels first
        pt3d_wheel = (
            f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/"
            f"py{sys.version_info.major}{sys.version_info.minor}_"
            f"cu{cuda_ver}_pyt{torch_ver_short}/download.html"
        )
        rc = _run(
            f"{sys.executable} -m pip install pytorch3d "
            f"--extra-index-url {pt3d_wheel}",
            check=False
        )
        if rc != 0:
            print("\n  Pre-built wheel failed. Building from source...")
            _run(
                f"{sys.executable} -m pip install "
                "'git+https://github.com/facebookresearch/pytorch3d.git'"
            )

    # ── pointops2 (custom CUDA kernels from ODIN repo) ──────────────────
    pointops_dir = ODIN_REPO_DIR / "odin" / "modeling" / "backbone" / "pointops2"
    if pointops_dir.exists():
        print("\n  Compiling pointops2 CUDA kernels...")
        _run(f"{sys.executable} setup.py install", cwd=pointops_dir)
    else:
        print(f"  [WARNING] pointops2 not found at {pointops_dir}. Skipping.")

    # ── deformable attention (from ODIN repo) ───────────────────────────
    defattn_dir = ODIN_REPO_DIR / "odin" / "modeling" / "pixel_decoder" / "ops"
    if defattn_dir.exists():
        print("\n  Compiling deformable attention CUDA kernels...")
        _run("bash make.sh", cwd=defattn_dir, check=False)
    else:
        print(f"  [WARNING] deformable attention not found at {defattn_dir}. Skipping.")

    # ── Install ODIN itself in editable mode ────────────────────────────
    if (ODIN_REPO_DIR / "setup.py").exists():
        _run(f"{sys.executable} -m pip install -e .", cwd=ODIN_REPO_DIR)
    elif (ODIN_REPO_DIR / "pyproject.toml").exists():
        _run(f"{sys.executable} -m pip install -e .", cwd=ODIN_REPO_DIR)

    print("\n  ✅ Dependencies installed.")


# ---------------------------------------------------------------------------
# Step 4: download pretrained weights
# ---------------------------------------------------------------------------

def download_weights(model_key: str = "odin_scannet200"):
    _print_section("Step 4 — Downloading pretrained weights")

    if model_key not in ODIN_CHECKPOINTS:
        print(f"  [ERROR] Unknown model key: {model_key}")
        print(f"  Available: {list(ODIN_CHECKPOINTS.keys())}")
        return

    WEIGHTS_DIR.mkdir(exist_ok=True)
    ckpt = ODIN_CHECKPOINTS[model_key]
    dest = WEIGHTS_DIR / ckpt["filename"]

    if dest.exists():
        print(f"  Weights already exist: {dest}  (delete to re-download)")
        return

    print(f"  Model   : {model_key}")
    print(f"  Desc    : {ckpt['description']}")
    print(f"  URL     : {ckpt['url']}")
    print(f"  Dest    : {dest}")
    print(f"  Downloading...")

    try:
        def _progress(count, block_size, total_size):
            if total_size > 0:
                pct = count * block_size * 100 / total_size
                print(f"\r  Progress: {min(pct, 100):.1f}%", end="", flush=True)

        urllib.request.urlretrieve(ckpt["url"], str(dest), reporthook=_progress)
        print()
        print(f"  ✅ Saved to: {dest}")
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print(f"\n  Please download manually:")
        print(f"    {ckpt['url']}")
        print(f"  and place at:")
        print(f"    {dest}")


# ---------------------------------------------------------------------------
# Step 5: verify installation
# ---------------------------------------------------------------------------

def verify_installation() -> bool:
    _print_section("Step 5 — Verification")
    ok = True

    # Check ODIN can be imported
    sys.path.insert(0, str(ODIN_REPO_DIR))
    try:
        import odin  # noqa
        print("  ✅ odin package importable")
    except ImportError as e:
        print(f"  ❌ odin import failed: {e}")
        ok = False

    # Check weights exist
    if not WEIGHTS_DIR.exists() or not list(WEIGHTS_DIR.glob("*.pth")):
        print("  ⚠️  No .pth weight files found in weights/")
        ok = False
    else:
        for f in WEIGHTS_DIR.glob("*.pth"):
            size_mb = f.stat().st_size / 1e6
            print(f"  ✅ {f.name}  ({size_mb:.0f} MB)")

    # Print config hint
    print("\n  Add to config.yaml:")
    print("    perception:")
    print("      pointcloud:")
    print("        backend: odin")
    print("        odin:")
    if list(WEIGHTS_DIR.glob("*.pth")):
        first = next(WEIGHTS_DIR.glob("*.pth"))
        print(f"          checkpoint: {first}")
    else:
        print(f"          checkpoint: weights/odin_scannet200.pth")
    print(f"          repo_dir: {ODIN_REPO_DIR}")

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ODIN setup for NBV model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--install",          action="store_true", help="Clone repo + install deps")
    parser.add_argument("--download-weights", action="store_true", help="Download pretrained weights")
    parser.add_argument("--model",            default="odin_scannet200",
                        choices=list(ODIN_CHECKPOINTS.keys()), help="Which checkpoint to download")
    parser.add_argument("--verify",           action="store_true", help="Verify installation")
    parser.add_argument("--all",              action="store_true", help="Run all steps")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    env_info = verify_environment()

    if args.install or args.all:
        clone_odin()
        install_dependencies(env_info)

    if args.download_weights or args.all:
        download_weights(args.model)

    if args.verify or args.all:
        ok = verify_installation()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
