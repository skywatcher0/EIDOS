from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InvarianceTestResult:
    stat: float
    p_value: float
    n_perm: int
    test_type: str


def _rff_features(
    z: np.ndarray,
    n_features: int = 128,
    gamma: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    if gamma is None:
        var = np.var(z) + 1e-9
        gamma = 1.0 / (2.0 * var)
    rng = np.random.default_rng(seed)
    w = rng.normal(loc=0.0, scale=np.sqrt(2.0 * gamma), size=(1, n_features))
    b = rng.uniform(0.0, 2.0 * np.pi, size=(1, n_features))
    phi = np.sqrt(2.0 / n_features) * np.cos(z @ w + b)
    return phi


def mmd_rff_stat(
    z: np.ndarray,
    env: np.ndarray,
    n_features: int = 128,
    seed: int = 0,
) -> float:
    phi = _rff_features(z=z, n_features=n_features, seed=seed)
    env = np.asarray(env)
    uniq = np.unique(env)
    if uniq.size <= 1:
        return 0.0
    means = {e: phi[env == e].mean(axis=0) for e in uniq}
    stat = 0.0
    for i, e1 in enumerate(uniq):
        for e2 in uniq[i + 1 :]:
            diff = means[e1] - means[e2]
            stat += float(diff @ diff)
    return stat


def mmd_rff_permutation_test(
    z: np.ndarray,
    env: np.ndarray,
    n_features: int = 128,
    n_perm: int = 200,
    seed: int = 0,
    permute_blocks: np.ndarray | None = None,
) -> InvarianceTestResult:
    rng = np.random.default_rng(seed)
    env = np.asarray(env)
    blocks = None if permute_blocks is None else np.asarray(permute_blocks)
    if blocks is not None and blocks.shape[0] != env.shape[0]:
        raise ValueError("permute_blocks must have the same length as env.")
    observed = mmd_rff_stat(z=z, env=env, n_features=n_features, seed=seed)
    ge = 1
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
        stat = mmd_rff_stat(z=z, env=perm, n_features=n_features, seed=seed + 1)
        if stat >= observed:
            ge += 1
    pval = ge / (n_perm + 1)
    return InvarianceTestResult(
        stat=float(observed),
        p_value=float(pval),
        n_perm=int(n_perm),
        test_type="mmd_rff",
    )
