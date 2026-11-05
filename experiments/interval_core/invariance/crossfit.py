from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class ResidualResult:
    z: np.ndarray
    env: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    fold: np.ndarray


def _stratified_folds(env: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folds = np.zeros(env.shape[0], dtype=int)
    for e in np.unique(env):
        idx = np.where(env == e)[0]
        rng.shuffle(idx)
        assignment = np.arange(idx.size) % k
        folds[idx] = assignment
    return folds


def crossfit_residuals(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    env_col: str,
    model_factory: Callable[[], object],
    k: int = 2,
    seed: int = 0,
) -> ResidualResult:
    x = df[feature_cols].to_numpy(float)
    y = df[target_col].to_numpy(float)
    env = df[env_col].to_numpy(int)
    folds = _stratified_folds(env=env, k=k, seed=seed)
    mu = np.zeros_like(y, dtype=float)
    sigma = np.zeros_like(y, dtype=float)

    for fold_id in range(k):
        test_idx = np.where(folds == fold_id)[0]
        train_idx = np.where(folds != fold_id)[0]
        model = model_factory()
        model.fit(x[train_idx], y[train_idx], env=env[train_idx])
        mu_i, sigma_i = model.predict_mu_sigma(x[test_idx])
        mu[test_idx] = mu_i
        sigma[test_idx] = sigma_i

    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    return ResidualResult(z=z, env=env, mu=mu, sigma=sigma, fold=folds)


def naive_residuals(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    env_col: str,
    model_factory: Callable[[], object],
) -> ResidualResult:
    x = df[feature_cols].to_numpy(float)
    y = df[target_col].to_numpy(float)
    env = df[env_col].to_numpy(int)
    model = model_factory()
    model.fit(x, y, env=env)
    mu, sigma = model.predict_mu_sigma(x)
    sigma = np.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    return ResidualResult(z=z, env=env, mu=mu, sigma=sigma, fold=np.zeros_like(env))

