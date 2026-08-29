from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalResult:
    alpha: float
    q_hat: float
    coverage: float
    avg_length: float
    p10_length: float
    p90_length: float
    lower: np.ndarray
    upper: np.ndarray
    guarantee_scope: str = "iid_exchangeable_only"
    finite_sample_bound_applicable: bool = True


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile with the exact edge rule.

    For ``k = ceil((n+1)(1-alpha))``, the order statistic is finite only when
    ``k <= n``.  The mathematically correct value for ``k = n+1`` is infinity,
    not the largest observed score.
    """

    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("calibration scores must be non-empty")
    if np.any(~np.isfinite(values)):
        raise ValueError("calibration scores must be finite")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    n = int(values.size)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(values, k - 1)[k - 1])


def split_conformal_regression(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    y_test: np.ndarray,
    yhat_test: np.ndarray,
    alpha: float = 0.1,
    exchangeable: bool = True,
) -> ConformalResult:
    y_cal = np.asarray(y_cal, dtype=float)
    yhat_cal = np.asarray(yhat_cal, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    yhat_test = np.asarray(yhat_test, dtype=float)

    scores = np.abs(y_cal - yhat_cal)
    q_hat = conformal_quantile(scores, alpha=alpha)

    lo = yhat_test - q_hat
    hi = yhat_test + q_hat
    inside = (y_test >= lo) & (y_test <= hi)
    lengths = hi - lo
    if np.isinf(q_hat):
        avg_length = p10_length = p90_length = float("inf")
    else:
        avg_length = float(np.mean(lengths))
        p10_length = float(np.quantile(lengths, 0.1))
        p90_length = float(np.quantile(lengths, 0.9))
    return ConformalResult(
        alpha=float(alpha),
        q_hat=q_hat,
        coverage=float(np.mean(inside)),
        avg_length=avg_length,
        p10_length=p10_length,
        p90_length=p90_length,
        lower=np.asarray(lo, dtype=float).copy(),
        upper=np.asarray(hi, dtype=float).copy(),
        guarantee_scope="iid_exchangeable_only" if exchangeable else "empirical_shift_diagnostic",
        finite_sample_bound_applicable=bool(exchangeable),
    )


def mondrian_conformal_regression(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    group_cal: np.ndarray,
    y_test: np.ndarray,
    yhat_test: np.ndarray,
    group_test: np.ndarray,
    alpha: float = 0.1,
    exchangeable: bool = True,
    unseen_group: str = "pooled",
) -> ConformalResult:
    """Group-conditional split conformal with an explicit unseen-group rule.

    The finite-sample Mondrian guarantee applies only when calibration and test
    observations are exchangeable within every requested group.  Under a
    shifted arm, or when a test group is absent from calibration, the returned
    intervals are an empirical diagnostic and ``finite_sample_bound_applicable``
    is false.  ``unseen_group`` may use the pooled calibration quantile or the
    numerically nearest observed group.
    """

    y_cal_arr = np.asarray(y_cal, dtype=float).reshape(-1)
    yhat_cal_arr = np.asarray(yhat_cal, dtype=float).reshape(-1)
    group_cal_arr = np.asarray(group_cal).reshape(-1)
    y_test_arr = np.asarray(y_test, dtype=float).reshape(-1)
    yhat_test_arr = np.asarray(yhat_test, dtype=float).reshape(-1)
    group_test_arr = np.asarray(group_test).reshape(-1)
    if not (
        y_cal_arr.shape == yhat_cal_arr.shape == group_cal_arr.shape
        and y_test_arr.shape == yhat_test_arr.shape == group_test_arr.shape
    ):
        raise ValueError("response, prediction, and group arrays must align")
    if y_cal_arr.size == 0 or y_test_arr.size == 0:
        raise ValueError("calibration and test roles must both be non-empty")
    if unseen_group not in {"pooled", "nearest"}:
        raise ValueError("unseen_group must be 'pooled' or 'nearest'")

    scores = np.abs(y_cal_arr - yhat_cal_arr)
    pooled_q = conformal_quantile(scores, alpha=alpha)
    available = np.unique(group_cal_arr)
    q_by_group = {
        group: conformal_quantile(scores[group_cal_arr == group], alpha=alpha)
        for group in available
    }
    unseen = set(np.unique(group_test_arr)) - set(available)
    q_test = np.empty(y_test_arr.shape, dtype=float)
    for index, group in enumerate(group_test_arr):
        if group in q_by_group:
            q_test[index] = q_by_group[group]
        elif unseen_group == "pooled":
            q_test[index] = pooled_q
        else:
            try:
                nearest = available[np.argmin(np.abs(available.astype(float) - float(group)))]
            except (TypeError, ValueError):
                q_test[index] = pooled_q
            else:
                q_test[index] = q_by_group[nearest]

    lower = yhat_test_arr - q_test
    upper = yhat_test_arr + q_test
    inside = (y_test_arr >= lower) & (y_test_arr <= upper)
    lengths = upper - lower
    if np.any(np.isinf(lengths)):
        avg_length = p10_length = p90_length = float("inf")
    else:
        avg_length = float(np.mean(lengths))
        p10_length = float(np.quantile(lengths, 0.1))
        p90_length = float(np.quantile(lengths, 0.9))
    guaranteed = bool(exchangeable and not unseen)
    return ConformalResult(
        alpha=float(alpha),
        q_hat=float(np.mean(q_test)),
        coverage=float(np.mean(inside)),
        avg_length=avg_length,
        p10_length=p10_length,
        p90_length=p90_length,
        lower=np.asarray(lower, dtype=float).copy(),
        upper=np.asarray(upper, dtype=float).copy(),
        guarantee_scope=(
            "mondrian_exchangeable"
            if guaranteed
            else "mondrian_empirical_shift_diagnostic"
        ),
        finite_sample_bound_applicable=guaranteed,
    )


def _linear_trend(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    *,
    nonnegative_slope: bool = False,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    x_new_arr = np.asarray(x_new, dtype=float)
    if x_arr.size == 1 or np.allclose(x_arr, x_arr[0]):
        return np.full(x_new_arr.shape, float(y_arr[-1]), dtype=float)
    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    if nonnegative_slope:
        slope = max(0.0, float(slope))
        intercept = float(np.mean(y_arr - slope * x_arr))
    return intercept + slope * x_new_arr


def proxy_trend_conformal_regression(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    env_cal: np.ndarray,
    y_test: np.ndarray,
    yhat_test: np.ndarray,
    env_test: np.ndarray,
    alpha: float = 0.1,
    seed: int = 0,
) -> ConformalResult:
    """Source-only interval transfer for ordered proxy environments.

    Calibration observations in every source environment are split internally.
    One half estimates a linear signed-residual trend; the other estimates
    environment-specific absolute-residual quantiles.  A non-decreasing
    log-quantile trend is extrapolated to the target proxy environment and
    inflated by the largest one-step source backtest error.  Test responses are
    used only to report empirical coverage.

    This is a shift diagnostic, not an exchangeable conformal guarantee.
    """

    y_cal_arr = np.asarray(y_cal, dtype=float).reshape(-1)
    yhat_cal_arr = np.asarray(yhat_cal, dtype=float).reshape(-1)
    env_cal_arr = np.asarray(env_cal).reshape(-1)
    y_test_arr = np.asarray(y_test, dtype=float).reshape(-1)
    yhat_test_arr = np.asarray(yhat_test, dtype=float).reshape(-1)
    env_test_arr = np.asarray(env_test).reshape(-1)
    if not (
        y_cal_arr.shape == yhat_cal_arr.shape == env_cal_arr.shape
        and y_test_arr.shape == yhat_test_arr.shape == env_test_arr.shape
    ):
        raise ValueError("response, prediction, and environment arrays must align")
    if y_cal_arr.size == 0 or y_test_arr.size == 0:
        raise ValueError("calibration and test roles must both be non-empty")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    source_envs = np.sort(np.unique(env_cal_arr.astype(float)))
    if source_envs.size < 3:
        fallback = mondrian_conformal_regression(
            y_cal=y_cal_arr,
            yhat_cal=yhat_cal_arr,
            group_cal=env_cal_arr,
            y_test=y_test_arr,
            yhat_test=yhat_test_arr,
            group_test=env_test_arr,
            alpha=alpha,
            exchangeable=False,
            unseen_group="nearest",
        )
        fallback.guarantee_scope = "source_only_proxy_trend_diagnostic"
        fallback.finite_sample_bound_applicable = False
        return fallback

    residual = y_cal_arr - yhat_cal_arr
    rng = np.random.default_rng(seed)
    location_rows: list[int] = []
    scale_rows: list[int] = []
    for env in source_envs:
        idx = np.flatnonzero(env_cal_arr.astype(float) == env)
        rng.shuffle(idx)
        cut = max(1, min(idx.size - 1, idx.size // 2)) if idx.size > 1 else 1
        location_rows.extend(idx[:cut].tolist())
        scale_rows.extend((idx[cut:] if cut < idx.size else idx).tolist())

    location_idx = np.asarray(location_rows, dtype=int)
    scale_idx = np.asarray(scale_rows, dtype=int)
    location_by_env = np.asarray(
        [
            np.median(residual[location_idx][env_cal_arr[location_idx].astype(float) == env])
            for env in source_envs
        ],
        dtype=float,
    )
    bias_cal = _linear_trend(
        source_envs,
        location_by_env,
        env_cal_arr[scale_idx].astype(float),
    )
    corrected_scores = np.abs(residual[scale_idx] - bias_cal)

    q_by_env: list[float] = []
    for env in source_envs:
        scores_env = corrected_scores[env_cal_arr[scale_idx].astype(float) == env]
        q_env = conformal_quantile(scores_env, alpha=alpha)
        if not np.isfinite(q_env):
            q_env = float(np.max(scores_env))
        q_by_env.append(max(float(q_env), 1e-12))
    q_env_arr = np.asarray(q_by_env, dtype=float)
    log_q = np.log(q_env_arr)

    backtest_inflation = 1.0
    for stop in range(2, source_envs.size):
        predicted = float(
            np.exp(
                _linear_trend(
                    source_envs[:stop],
                    log_q[:stop],
                    np.asarray([source_envs[stop]]),
                    nonnegative_slope=True,
                )[0]
            )
        )
        backtest_inflation = max(backtest_inflation, q_env_arr[stop] / max(predicted, 1e-12))

    target_env_float = env_test_arr.astype(float)
    target_bias = _linear_trend(source_envs, location_by_env, target_env_float)
    target_q = np.exp(
        _linear_trend(
            source_envs,
            log_q,
            target_env_float,
            nonnegative_slope=True,
        )
    ) * backtest_inflation
    pooled_q = conformal_quantile(corrected_scores, alpha=alpha)
    if not np.isfinite(pooled_q):
        pooled_q = float(np.max(corrected_scores))
    target_q = np.maximum(target_q, float(pooled_q))

    center = yhat_test_arr + target_bias
    trend_lower = center - target_q
    trend_upper = center + target_q
    baseline_q = conformal_quantile(np.abs(residual), alpha=alpha)
    if not np.isfinite(baseline_q):
        baseline_q = float(np.max(np.abs(residual)))
    nearest_floor = mondrian_conformal_regression(
        y_cal=y_cal_arr,
        yhat_cal=yhat_cal_arr,
        group_cal=env_cal_arr,
        y_test=y_test_arr,
        yhat_test=yhat_test_arr,
        group_test=env_test_arr,
        alpha=alpha,
        exchangeable=False,
        unseen_group="nearest",
    )
    # Preserve both source-calibrated baselines as nested safety floors.  A
    # misspecified proxy trend can widen, but cannot remove points covered by
    # either pooled split conformal or the declared nearest-group fallback.
    lower = np.minimum.reduce([trend_lower, yhat_test_arr - baseline_q, nearest_floor.lower])
    upper = np.maximum.reduce([trend_upper, yhat_test_arr + baseline_q, nearest_floor.upper])
    inside = (y_test_arr >= lower) & (y_test_arr <= upper)
    lengths = upper - lower
    return ConformalResult(
        alpha=float(alpha),
        q_hat=float(np.mean((upper - lower) / 2.0)),
        coverage=float(np.mean(inside)),
        avg_length=float(np.mean(lengths)),
        p10_length=float(np.quantile(lengths, 0.1)),
        p90_length=float(np.quantile(lengths, 0.9)),
        lower=np.asarray(lower, dtype=float).copy(),
        upper=np.asarray(upper, dtype=float).copy(),
        guarantee_scope="source_only_proxy_trend_diagnostic",
        finite_sample_bound_applicable=False,
    )
