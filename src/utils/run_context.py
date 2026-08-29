from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import dump_yaml


TRACKED_DEPENDENCIES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pyyaml",
    "pyarrow",
    "torch",
    "openml",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_hash(default: str = "nogit") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out
    except Exception:
        return default


def short_git_hash(default: str = "nogit") -> str:
    return git_hash(default=default)[:8]


def dependency_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in TRACKED_DEPENDENCIES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "missing"
    return out


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["torch_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        info["torch_cuda_available"] = False
        info["torch_device_count"] = 0
    return info


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    timestamp: str
    git_hash: str


def make_run_context(outputs_root: str | Path, run_prefix: str = "run") -> RunContext:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    g = short_git_hash()
    run_id = f"{run_prefix}_{ts}_{g}"
    run_dir = Path(outputs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunContext(run_id=run_id, run_dir=run_dir, timestamp=utc_now_iso(), git_hash=git_hash())


def write_meta(ctx: RunContext, config: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "run_id": ctx.run_id,
        "timestamp": ctx.timestamp,
        "git_hash": ctx.git_hash,
        "dependencies": dependency_versions(),
        "hardware": hardware_info(),
        "config_path": str(ctx.run_dir / "config.yaml"),
    }
    if extra:
        payload.update(extra)
    dump_yaml(config, ctx.run_dir / "config.yaml")
    with (ctx.run_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

