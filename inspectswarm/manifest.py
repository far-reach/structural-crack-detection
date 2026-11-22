"""Reproducibility helpers: seeding and a manifest written beside every result."""
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_sha() -> str:
    try:
        repo = Path(__file__).resolve().parents[1]
        sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _hardware() -> str:
    s = f"{platform.processor()} | {platform.system()} {platform.release()}"
    try:
        import torch

        if torch.cuda.is_available():
            s += f" | {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    return s


def manifest(config: dict, seed=None, extra: dict | None = None) -> dict:
    """Describe the run: code version, config hash, seed, packages, hardware."""
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("torch", "torchvision", "timm", "scipy", "sklearn", "onnxruntime"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            pass
    cfg = json.dumps(config, sort_keys=True, default=str)
    out = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": _git_sha(),
        "config": config,
        "config_sha1": hashlib.sha1(cfg.encode()).hexdigest()[:12],
        "seed": seed,
        "versions": versions,
        "hardware": _hardware(),
    }
    if extra:
        out.update(extra)
    return out


def save_result(path, result: dict, config: dict, seed=None, extra=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest(config, seed, extra), "result": result}
    path.write_text(json.dumps(payload, indent=1, default=float))
