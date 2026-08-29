from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class WeightedConformalResult:
    alpha: float
    coverage: float
    avg_length: float
    p10_length: float
    p90_length: float
    qhat_mean: float
    qhat_median: float
    qhat_max: float
    inf_fallback_rate: float


def estimate_density_ratio_logistic(
    x_source: np.ndarray,
    x_target: np.ndarray,
    seed: int = 0,
    ratio_clip: tuple[float, float] = (1e-3, 50.0),
) -> tuple[np.ndarray, np.ndarray]:
    x_source = np.asarray(x_source, dtype=float)
    x_target = np.asarray(x_target, dtype=float)
    if x_source.ndim != 2 or x_target.ndim != 2:
        raise ValueError("x_source and x_target must be 2D arrays.")
    x_all = np.vstack([x_source, x_target])
    y_all = np.concatenate([np.zeros(x_source.shape[0], dtype=int), np.ones(x_target.shape[0], dtype=int)])
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
            C=1.0,
            random_state=seed,
        ),
    )
    clf.fit(x_all, y_all)
    p_source = clf.predict_proba(x_source)[:, 1]
    p_target = clf.predict_proba(x_target)[:, 1]
    prior_ratio = float(x_source.shape[0] / max(1, x_target.shape[0]))

    def _ratio_from_p(p: np.ndarray) -> np.ndarray:
        odds = np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1.0)
        ratio = prior_ratio * odds
        return np.clip(ratio, float(ratio_clip[0]), float(ratio_clip[1]))

    return _ratio_from_p(p_source), _ratio_from_p(p_target)


def weighted_split_conformal_regression(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    x_cal: np.ndarray,
    y_test: np.ndarray,
    yhat_test: np.ndarray,
    x_test: np.ndarray,
    x_target_pool: np.ndarray,
    sigma_cal: np.ndarray | None = None,
    sigma_test: np.ndarray | None = None,
    alpha: float = 0.1,
    seed: int = 0,
    ratio_clip: tuple[float, float] = (1e-3, 50.0),
) -> WeightedConformalResult:
    y_cal = np.asarray(y_cal, dtype=float).reshape(-1)
    yhat_cal = np.asarray(yhat_cal, dtype=float).reshape(-1)
    x_cal = np.asarray(x_cal, dtype=float)
    y_test = np.asarray(y_test, dtype=float).reshape(-1)
    yhat_test = np.asarray(yhat_test, dtype=float).reshape(-1)
    x_test = np.asarray(x_test, dtype=float)
    x_target_pool = np.asarray(x_target_pool, dtype=float)
    if sigma_cal is None:
        sigma_cal_arr = np.ones_like(y_cal, dtype=float)
    else:
        sigma_cal_arr = np.asarray(sigma_cal, dtype=float).reshape(-1)
    if sigma_test is None:
        sigma_test_arr = np.ones_like(y_test, dtype=float)
    else:
        sigma_test_arr = np.asarray(sigma_test, dtype=float).reshape(-1)

    if y_cal.shape[0] == 0 or y_test.shape[0] == 0:
        raise ValueError("Calibration and test sets must be non-empty.")
    if sigma_cal_arr.shape != y_cal.shape or sigma_test_arr.shape != y_test.shape:
        raise ValueError("sigma_cal and sigma_test must match y_cal and y_test.")
    sigma_eps = 1e-6
    sigma_cal_arr = np.maximum(sigma_cal_arr, sigma_eps)
    sigma_test_arr = np.maximum(sigma_test_arr, sigma_eps)

    w_cal, _ = estimate_density_ratio_logistic(
        x_source=x_cal,
        x_target=x_target_pool,
        seed=seed,
        ratio_clip=ratio_clip,
    )
    _, w_test = estimate_density_ratio_logistic(
        x_source=x_cal,
        x_target=x_test,
        seed=seed + 1,
        ratio_clip=ratio_clip,
    )
    scores = np.abs(y_cal - yhat_cal) / sigma_cal_arr
    inf_fallback = 0
    qhats = np.zeros_like(y_test, dtype=float)

    scores_aug = np.concatenate([scores, [np.inf]])
    for i in range(y_test.shape[0]):
        weights_aug = np.concatenate([w_cal, [w_test[i]]])
        weights_aug = weights_aug / np.clip(np.sum(weights_aug), 1e-12, np.inf)
        order = np.argsort(scores_aug)
        sorted_scores = scores_aug[order]
        sorted_weights = weights_aug[order]
        cdf = np.cumsum(sorted_weights)
        hit = int(np.searchsorted(cdf, 1 - alpha, side="left"))
        qhat = float(sorted_scores[min(hit, sorted_scores.shape[0] - 1)])
        if not np.isfinite(qhat):
            qhat = float(np.max(scores))
            inf_fallback += 1
        qhats[i] = qhat

    radius = qhats * sigma_test_arr
    lo = yhat_test - radius
    hi = yhat_test + radius
    inside = (y_test >= lo) & (y_test <= hi)
    lengths = hi - lo
    return WeightedConformalResult(
        alpha=float(alpha),
        coverage=float(np.mean(inside)),
        avg_length=float(np.mean(lengths)),
        p10_length=float(np.quantile(lengths, 0.1)),
        p90_length=float(np.quantile(lengths, 0.9)),
        qhat_mean=float(np.mean(qhats)),
        qhat_median=float(np.median(qhats)),
        qhat_max=float(np.max(qhats)),
        inf_fallback_rate=float(inf_fallback / max(1, y_test.shape[0])),
    )
