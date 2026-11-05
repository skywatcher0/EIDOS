from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HSICResult:
    stat: float
    p_value: float
    n_perm: int
    test_type: str


def _rbf_kernel(x: np.ndarray, sigma: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    sq = (x - x.T) ** 2
    if sigma is None:
        med = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
        sigma = np.sqrt(max(med, 1e-9))
    return np.exp(-sq / (2 * sigma**2))


def _delta_kernel(env: np.ndarray) -> np.ndarray:
    env = np.asarray(env)
    return (env[:, None] == env[None, :]).astype(float)


def hsic_stat(z: np.ndarray, env: np.ndarray) -> float:
    k = _rbf_kernel(z)
    l = _delta_kernel(env)
    n = k.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    kc = h @ k @ h
    lc = h @ l @ h
    stat = np.trace(kc @ lc) / (n - 1) ** 2
    return float(max(stat, 0.0))


def hsic_permutation_test(
    z: np.ndarray,
    env: np.ndarray,
    n_perm: int = 200,
    seed: int = 0,
    permute_blocks: np.ndarray | None = None,
) -> HSICResult:
    rng = np.random.default_rng(seed)
    obs = hsic_stat(z, env)
    ge = 1
    env = np.asarray(env)
    blocks = None if permute_blocks is None else np.asarray(permute_blocks)
    if blocks is not None and blocks.shape[0] != env.shape[0]:
        raise ValueError("permute_blocks must have the same length as env.")
    for _ in range(n_perm):
        if blocks is None:
            perm = rng.permutation(env)
        else:
            perm = env.copy()
            for b in np.unique(blocks):
                idx = np.where(blocks == b)[0]
                if idx.size <= 1:
                    continue
                perm[idx] = rng.permutation(perm[idx])
        stat = hsic_stat(z, perm)
        if stat >= obs:
            ge += 1
    pval = ge / (n_perm + 1)
    return HSICResult(stat=obs, p_value=float(pval), n_perm=n_perm, test_type="hsic")
