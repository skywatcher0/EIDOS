from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PermutationResult:
    observed: float
    p_value: float
    n_perm: int


def permutation_test(
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    x: np.ndarray,
    labels: np.ndarray,
    n_perm: int = 500,
    seed: int = 0,
) -> PermutationResult:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    obs = float(stat_fn(x, labels))
    ge = 1
    for _ in range(n_perm):
        stat = float(stat_fn(x, rng.permutation(labels)))
        if stat >= obs:
            ge += 1
    return PermutationResult(observed=obs, p_value=ge / (n_perm + 1), n_perm=n_perm)

