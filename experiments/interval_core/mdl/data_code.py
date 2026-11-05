from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, t


NoiseType = Literal["gaussian", "student_t"]


@dataclass(frozen=True)
class UniformQuantizer:
    """A fixed half-open partition of the real line.

    Bin ``k`` is ``[origin + (k-1/2) delta, origin + (k+1/2) delta)``.
    Ties therefore belong to the bin on their right.  The partition is fixed
    before any observation is encoded; it is not centred on the observation.
    """

    delta: float
    origin: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("delta must be finite and strictly positive")
        if not np.isfinite(self.origin):
            raise ValueError("origin must be finite")

    def index(self, values: np.ndarray | float) -> np.ndarray:
        z = (np.asarray(values, dtype=float) - self.origin) / self.delta
        return np.floor(z + 0.5).astype(np.int64)

    def representative(self, indices: np.ndarray | int) -> np.ndarray:
        return self.origin + np.asarray(indices, dtype=float) * self.delta

    def quantize(self, values: np.ndarray | float) -> np.ndarray:
        return self.representative(self.index(values))

    def bounds(self, indices: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
        k = np.asarray(indices, dtype=float)
        lo = self.origin + (k - 0.5) * self.delta
        hi = self.origin + (k + 0.5) * self.delta
        return lo, hi


@dataclass
class DataCodeResult:
    dl_data_bits: float
    avg_bits: float
    min_bits: float
    max_bits: float
    negative_bits_count: int
    probs_min: float
    probs_max: float


def _distribution(noise: NoiseType, df: float):
    if noise == "gaussian":
        return norm
    if noise == "student_t":
        if not np.isfinite(df) or df <= 0:
            raise ValueError("df must be finite and positive")
        return t(df=df)
    raise ValueError(f"unsupported noise model: {noise}")


def _logdiffexp(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """Return log(exp(log_a)-exp(log_b)) for log_a >= log_b."""

    gap = np.minimum(np.asarray(log_b) - np.asarray(log_a), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(log_a) + np.log1p(-np.exp(gap))


def _log_interval_mass(
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    noise: NoiseType,
    df: float,
) -> np.ndarray:
    """Stable log probability for standardized intervals.

    Left-tail intervals use log-CDF differences, right-tail intervals use
    log-survival differences, and intervals crossing zero use direct CDFs.
    """

    dist = _distribution(noise, df)
    lo = np.asarray(z_lo, dtype=float)
    hi = np.asarray(z_hi, dtype=float)
    if np.any(hi <= lo):
        raise ValueError("every upper bin edge must exceed its lower edge")

    out = np.empty(np.broadcast_shapes(lo.shape, hi.shape), dtype=float)
    lo, hi = np.broadcast_arrays(lo, hi)

    left = hi <= 0.0
    right = lo >= 0.0
    middle = ~(left | right)

    if np.any(left):
        out[left] = _logdiffexp(dist.logcdf(hi[left]), dist.logcdf(lo[left]))
    if np.any(right):
        out[right] = _logdiffexp(dist.logsf(lo[right]), dist.logsf(hi[right]))
    if np.any(middle):
        mass = dist.cdf(hi[middle]) - dist.cdf(lo[middle])
        with np.errstate(divide="ignore"):
            out[middle] = np.log(mass)
    return out


def symmetric_geometric_mass(indices: np.ndarray | int, rho: float = 0.98) -> np.ndarray:
    """A normalized escape distribution on all integer quantizer bins."""

    if not np.isfinite(rho) or not 0.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between zero and one")
    k = np.asarray(indices, dtype=np.int64)
    normalizer = (1.0 - rho) / (1.0 + rho)
    return normalizer * np.power(rho, np.abs(k).astype(float))


def bin_log_prob_mass(
    y_tilde: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    delta_y: float,
    noise: NoiseType = "student_t",
    df: float = 3.0,
    robust_eps: float = 1e-4,
    origin: float = 0.0,
    escape_rho: float = 0.98,
) -> np.ndarray:
    """Log mass of the fixed quantizer bins containing ``y_tilde``.

    The optional robust component is a proper distribution over the same
    integer bins.  Consequently the returned masses define a normalized
    probability model and yield non-negative ideal code lengths.
    """

    if not np.isfinite(robust_eps) or not 0.0 <= robust_eps < 1.0:
        raise ValueError("robust_eps must lie in [0, 1)")

    quantizer = UniformQuantizer(delta=float(delta_y), origin=float(origin))
    indices = quantizer.index(y_tilde)
    lo, hi = quantizer.bounds(indices)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    lo, hi, mu_arr, sigma_safe = np.broadcast_arrays(lo, hi, mu_arr, sigma_safe)
    z_lo = (lo - mu_arr) / sigma_safe
    z_hi = (hi - mu_arr) / sigma_safe
    log_model = _log_interval_mass(z_lo, z_hi, noise=noise, df=df)
    log_model = np.where(np.isfinite(log_model), log_model, -np.inf)

    if robust_eps == 0.0:
        return log_model

    if not np.isfinite(escape_rho) or not 0.0 < escape_rho < 1.0:
        raise ValueError("escape_rho must lie strictly between zero and one")

    log_escape = (
        np.log((1.0 - escape_rho) / (1.0 + escape_rho))
        + np.abs(indices).astype(float) * np.log(escape_rho)
    )
    stacked = np.stack(
        [np.log1p(-robust_eps) + log_model, np.log(robust_eps) + log_escape],
        axis=0,
    )
    mixed = logsumexp(stacked, axis=0)
    return mixed


def bin_prob_mass(
    y_tilde: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    delta_y: float,
    noise: NoiseType = "student_t",
    df: float = 3.0,
    robust_eps: float = 1e-4,
    range_max: float | None = None,
    origin: float = 0.0,
    escape_rho: float = 0.98,
) -> np.ndarray:
    """Probability wrapper retained for callers that do not need log masses.

    ``range_max`` is accepted for backward compatibility but deliberately
    ignored: the former range-uniform background was not normalized on the
    unbounded quantizer alphabet.
    """

    del range_max
    return np.exp(
        bin_log_prob_mass(
            y_tilde=y_tilde,
            mu=mu,
            sigma=sigma,
            delta_y=delta_y,
            noise=noise,
            df=df,
            robust_eps=robust_eps,
            origin=origin,
            escape_rho=escape_rho,
        )
    )


def dl_data_bin_likelihood(
    y_tilde: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    delta_y: float,
    noise: NoiseType = "student_t",
    df: float = 3.0,
    robust_eps: float = 1e-4,
    origin: float = 0.0,
    escape_rho: float = 0.98,
) -> DataCodeResult:
    log_p = bin_log_prob_mass(
        y_tilde=y_tilde,
        mu=mu,
        sigma=sigma,
        delta_y=delta_y,
        noise=noise,
        df=df,
        robust_eps=robust_eps,
        origin=origin,
        escape_rho=escape_rho,
    )
    bits = -log_p / np.log(2.0)
    probs = np.exp(log_p)
    return DataCodeResult(
        dl_data_bits=float(bits.sum()),
        avg_bits=float(bits.mean()),
        min_bits=float(bits.min()),
        max_bits=float(bits.max()),
        negative_bits_count=int((bits < 0).sum()),
        probs_min=float(probs.min()),
        probs_max=float(probs.max()),
    )


def _pdf(x: np.ndarray, noise: NoiseType = "student_t", df: float = 3.0) -> np.ndarray:
    dist = _distribution(noise, df)
    return dist.pdf(x)


def dl_data_naive_pdf(
    y_tilde: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    noise: NoiseType = "student_t",
    df: float = 3.0,
) -> DataCodeResult:
    """Legacy density score; retained only as a named non-code ablation."""

    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    z = (np.asarray(y_tilde, dtype=float) - np.asarray(mu, dtype=float)) / sigma_safe
    p = _pdf(z, noise=noise, df=df) / sigma_safe
    p = np.clip(p, 1e-300, 1e300)
    bits = -np.log2(p)
    return DataCodeResult(
        dl_data_bits=float(bits.sum()),
        avg_bits=float(bits.mean()),
        min_bits=float(bits.min()),
        max_bits=float(bits.max()),
        negative_bits_count=int((bits < 0).sum()),
        probs_min=float(p.min()),
        probs_max=float(p.max()),
    )
