from __future__ import annotations

import numpy as np


def _safe_scale(values: np.ndarray, min_scale: float) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float(max(min_scale, 1.0))
    scale = float(np.std(arr))
    if not np.isfinite(scale) or scale < min_scale:
        mad = float(np.median(np.abs(arr - np.median(arr))))
        scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < min_scale:
        scale = 1.0
    return float(max(min_scale, scale))


def normalize_residuals_by_env(
    z: np.ndarray,
    env: np.ndarray,
    mode: str = "global",
    *,
    ref_z: np.ndarray | None = None,
    ref_env: np.ndarray | None = None,
    min_scale: float = 1e-6,
    shrinkage: float = 4.0,
) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    env_arr = np.asarray(env, dtype=int).reshape(-1)
    if z_arr.shape != env_arr.shape:
        raise ValueError("z and env must have the same shape.")

    mode_s = str(mode or "global").strip().lower()
    if mode_s in {"", "global", "none"}:
        return z_arr
    if mode_s not in {"env_std", "env_studentized"}:
        raise ValueError(f"Unknown residual normalization mode: {mode}")

    ref_z_arr = np.asarray(ref_z if ref_z is not None else z_arr, dtype=float).reshape(-1)
    ref_env_arr = np.asarray(ref_env if ref_env is not None else env_arr, dtype=int).reshape(-1)
    if ref_z_arr.shape != ref_env_arr.shape:
        raise ValueError("ref_z and ref_env must have the same shape.")

    global_scale = _safe_scale(ref_z_arr, min_scale=float(min_scale))
    shrink = float(max(0.0, shrinkage))
    env_scales: dict[int, float] = {}
    for env_id in np.unique(ref_env_arr):
        mask = ref_env_arr == int(env_id)
        local = _safe_scale(ref_z_arr[mask], min_scale=float(min_scale))
        n_local = int(np.sum(mask))
        if shrink > 0.0:
            weight = float(n_local) / float(n_local + shrink)
            scale = float(np.sqrt(weight * local * local + (1.0 - weight) * global_scale * global_scale))
        else:
            scale = float(local)
        env_scales[int(env_id)] = float(max(min_scale, scale))

    out = np.empty_like(z_arr, dtype=float)
    for env_id in np.unique(env_arr):
        mask = env_arr == int(env_id)
        scale = env_scales.get(int(env_id), global_scale)
        out[mask] = z_arr[mask] / float(max(min_scale, scale))
    return out
