from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ROLE_ORDER = ("disc", "diag", "audit", "cal", "test")


def _validate_ratios(ratios: tuple[float, ...], expected: int) -> np.ndarray:
    values = np.asarray(ratios, dtype=float)
    if values.size != expected or np.any(values <= 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError(f"expected {expected} positive ratios summing to one")
    return values


def _allocate_counts(n: int, ratios: np.ndarray) -> np.ndarray:
    if n < ratios.size:
        raise ValueError(f"at least {ratios.size} observations are required per stratum")
    raw = ratios * n
    counts = np.floor(raw).astype(int)
    counts[counts == 0] = 1
    while counts.sum() > n:
        candidates = np.where(counts > 1)[0]
        j = int(candidates[np.argmax(counts[candidates] - raw[candidates])])
        counts[j] -= 1
    while counts.sum() < n:
        j = int(np.argmax(raw - counts))
        counts[j] += 1
    return counts


def five_role_split_by_env(
    env: np.ndarray,
    seed: int,
    ratios: tuple[float, float, float, float, float] = (0.4, 0.15, 0.15, 0.15, 0.15),
) -> np.ndarray:
    """Create disjoint discovery, diagnostic, audit, calibration and test roles."""

    weights = _validate_ratios(ratios, expected=5)
    labels = np.empty(np.asarray(env).shape[0], dtype=object)
    rng = np.random.default_rng(seed)
    env_arr = np.asarray(env)
    for value in np.unique(env_arr):
        indices = np.flatnonzero(env_arr == value)
        rng.shuffle(indices)
        counts = _allocate_counts(int(indices.size), weights)
        start = 0
        for role, count in zip(ROLE_ORDER, counts, strict=True):
            labels[indices[start : start + count]] = role
            start += int(count)
    return labels


def heldout_environment_split(
    env: np.ndarray,
    heldout_env: int | float | str,
    seed: int,
    source_ratios: tuple[float, float, float, float] = (0.5, 1 / 6, 1 / 6, 1 / 6),
) -> np.ndarray:
    """Reserve one environment for empirical OOD testing.

    Source observations receive the four pre-test roles.  The held-out
    environment is never used for discovery, selection, audit or calibration.
    This split does not create an exchangeability guarantee across environments.
    """

    env_arr = np.asarray(env)
    is_test = env_arr == heldout_env
    if not np.any(is_test) or np.all(is_test):
        raise ValueError("heldout_env must identify a non-empty proper subset")
    weights = _validate_ratios(source_ratios, expected=4)
    labels = np.full(env_arr.shape[0], "test", dtype=object)
    rng = np.random.default_rng(seed)
    for value in np.unique(env_arr[~is_test]):
        indices = np.flatnonzero((env_arr == value) & ~is_test)
        rng.shuffle(indices)
        counts = _allocate_counts(int(indices.size), weights)
        start = 0
        for role, count in zip(ROLE_ORDER[:4], counts, strict=True):
            labels[indices[start : start + count]] = role
            start += int(count)
    return labels


@dataclass(frozen=True)
class EvidencePartitions:
    frame: pd.DataFrame
    split_col: str = "split"

    def __post_init__(self) -> None:
        if self.split_col not in self.frame:
            raise ValueError(f"missing split column: {self.split_col}")
        observed = set(self.frame[self.split_col].astype(str).unique())
        unknown = observed - set(ROLE_ORDER)
        if unknown:
            raise ValueError(f"unknown evidence roles: {sorted(unknown)}")

    def discovery(self) -> pd.DataFrame:
        return self.frame[self.frame[self.split_col] == "disc"].copy()

    def diagnostic_selection(self) -> pd.DataFrame:
        return self.frame[self.frame[self.split_col] == "diag"].copy()

    def audit(self, selected_candidate_id: str) -> pd.DataFrame:
        if not selected_candidate_id:
            raise ValueError("audit access requires a frozen candidate identifier")
        return self.frame[self.frame[self.split_col] == "audit"].copy()

    def calibration(self, selected_candidate_id: str) -> pd.DataFrame:
        if not selected_candidate_id:
            raise ValueError("calibration access requires a frozen candidate identifier")
        return self.frame[self.frame[self.split_col] == "cal"].copy()

    def test(self, selected_candidate_id: str) -> pd.DataFrame:
        if not selected_candidate_id:
            raise ValueError("test access requires a frozen candidate identifier")
        return self.frame[self.frame[self.split_col] == "test"].copy()


@dataclass(frozen=True)
class AuditResult:
    mean_difference: float
    ci_low: float
    ci_high: float
    tolerance: float
    conclusion: str
    clusters: int
    bootstrap_draws: int


def audit_paired_difference(
    candidate_loss: np.ndarray,
    reference_loss: np.ndarray,
    cluster: np.ndarray,
    tolerance: float,
    seed: int = 0,
    draws: int = 5000,
    confidence: float = 0.95,
) -> AuditResult:
    """Cluster bootstrap for a frozen candidate on an untouched audit set."""

    candidate = np.asarray(candidate_loss, dtype=float).reshape(-1)
    reference = np.asarray(reference_loss, dtype=float).reshape(-1)
    groups = np.asarray(cluster).reshape(-1)
    if candidate.size == 0 or candidate.shape != reference.shape or candidate.shape != groups.shape:
        raise ValueError("candidate, reference and cluster must be non-empty aligned vectors")
    if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(reference)):
        raise ValueError("audit losses must be finite")
    if tolerance < 0 or draws < 100 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid tolerance, draw count or confidence")

    difference = candidate - reference
    unique = np.unique(groups)
    cluster_means = np.array([difference[groups == value].mean() for value in unique])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, unique.size, size=(draws, unique.size))
    boot = cluster_means[sampled].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(boot, [tail, 1.0 - tail])
    mean = float(cluster_means.mean())
    if ci_high < -tolerance:
        conclusion = "candidate_better_than_tolerance"
    elif ci_low > tolerance:
        conclusion = "candidate_worse_than_tolerance"
    elif ci_low >= -tolerance and ci_high <= tolerance:
        conclusion = "equivalent_within_tolerance"
    else:
        conclusion = "inconclusive"
    return AuditResult(
        mean_difference=mean,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        tolerance=float(tolerance),
        conclusion=conclusion,
        clusters=int(unique.size),
        bootstrap_draws=int(draws),
    )
