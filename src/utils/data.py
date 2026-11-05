from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def infer_delta_y(y_tilde: np.ndarray) -> float:
    y = np.asarray(y_tilde, dtype=float)
    uniq = np.unique(np.round(y, 12))
    if uniq.size < 2:
        return 1e-3
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1e-3
    # robust minimum effective step
    med = float(np.median(diffs))
    return max(med, 1e-8)


def quantize_y(y: np.ndarray, delta_y: float) -> np.ndarray:
    if not np.isfinite(delta_y) or delta_y <= 0:
        raise ValueError("delta_y must be finite and strictly positive")
    values = np.asarray(y, dtype=float)
    indices = np.floor(values / float(delta_y) + 0.5)
    return indices * float(delta_y)


def stratified_split_by_env(
    env: np.ndarray,
    seed: int,
    work_ratio: float = 0.6,
    lock_ratio: float = 0.2,
    cal_ratio: float = 0.2,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    env = np.asarray(env)
    split = np.array(["cal"] * env.shape[0], dtype=object)
    for e in np.unique(env):
        idx = np.where(env == e)[0]
        rng.shuffle(idx)
        n = idx.size
        n_work = int(np.floor(work_ratio * n))
        n_lock = int(np.floor(lock_ratio * n))
        n_cal = n - n_work - n_lock
        w = idx[:n_work]
        l = idx[n_work : n_work + n_lock]
        c = idx[n_work + n_lock : n_work + n_lock + n_cal]
        split[w] = "work"
        split[l] = "lock"
        split[c] = "cal"
    return split


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataset_artifacts(
    out_dir: str | Path,
    data_df: pd.DataFrame,
    metadata: dict[str, Any],
    readme_lines: list[str],
) -> None:
    p = ensure_dir(out_dir)
    data_df.to_csv(p / "data.csv", index=False)
    with (p / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with (p / "README.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines) + "\n")
    data_df[["sample_id", "split"]].to_parquet(p / "split_index.parquet", index=False)
