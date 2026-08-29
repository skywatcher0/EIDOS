from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class FisherEstimateDiagnostics:
    fisher_matrix: np.ndarray
    jacobian: np.ndarray
    step_sizes: np.ndarray
    grad_norm_mean: float
    grad_norm_std: float
    grad_norm_max: float
    sample_count: int


@dataclass
class FisherCodeResult:
    dl_param_bits: float
    logdet: float
    jitter_used: float
    rank: int
    fisher_matrix: np.ndarray
    eig_min: float
    eig_max: float
    cond_number: float
    effective_rank: int


def estimate_fisher_info(
    logpdf_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    diff_eps: float = 1e-4,
) -> np.ndarray:
    diag = estimate_fisher_info_with_diagnostics(
        logpdf_fn=logpdf_fn,
        params=params,
        diff_rel_eps=max(float(diff_eps), 1e-8),
        diff_abs_eps=max(float(diff_eps) * 0.1, 1e-9),
    )
    return diag.fisher_matrix


def estimate_fisher_info_with_diagnostics(
    logpdf_fn: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    diff_rel_eps: float = 1e-4,
    diff_abs_eps: float = 1e-6,
    min_step: float = 1e-7,
    max_step: float = 5e-2,
    clip_logpdf: float = 1e6,
) -> FisherEstimateDiagnostics:
    p = np.asarray(params, dtype=float)
    n_params = p.size
    base = np.asarray(logpdf_fn(p), dtype=float).reshape(-1)
    n = int(base.size)
    if n == 0:
        raise ValueError("logpdf_fn returned an empty vector.")

    rel_eps = max(float(diff_rel_eps), 1e-10)
    abs_eps = max(float(diff_abs_eps), 0.0)
    step_min = max(float(min_step), 1e-12)
    step_max = max(float(max_step), step_min)
    clip_bound = max(float(clip_logpdf), 10.0)

    jmat = np.zeros((n, n_params), dtype=float)
    step_sizes = np.zeros(n_params, dtype=float)
    for j in range(n_params):
        step = np.zeros_like(p)
        step_j = rel_eps * max(abs(float(p[j])), 1.0) + abs_eps
        step_j = float(np.clip(step_j, step_min, step_max))
        step_sizes[j] = step_j
        step[j] = step_j

        plus = np.asarray(logpdf_fn(p + step), dtype=float).reshape(-1)
        minus = np.asarray(logpdf_fn(p - step), dtype=float).reshape(-1)
        if plus.size != n or minus.size != n:
            raise ValueError("logpdf_fn output shape changed during finite-difference probing.")
        plus = np.clip(np.nan_to_num(plus, nan=0.0, posinf=clip_bound, neginf=-clip_bound), -clip_bound, clip_bound)
        minus = np.clip(
            np.nan_to_num(minus, nan=0.0, posinf=clip_bound, neginf=-clip_bound),
            -clip_bound,
            clip_bound,
        )
        grad = (plus - minus) / (2.0 * step_j)
        jmat[:, j] = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    fisher = (jmat.T @ jmat) / max(1, n)
    fisher = 0.5 * (fisher + fisher.T)
    grad_norms = np.linalg.norm(jmat, axis=1)
    return FisherEstimateDiagnostics(
        fisher_matrix=fisher,
        jacobian=jmat,
        step_sizes=step_sizes,
        grad_norm_mean=float(np.mean(grad_norms)),
        grad_norm_std=float(np.std(grad_norms, ddof=0)),
        grad_norm_max=float(np.max(grad_norms)),
        sample_count=n,
    )


def fisher_param_code_bits(
    fisher: np.ndarray,
    n_samples: int,
    lambda_prior: float = 1e-4,
    jitter: float = 1e-10,
    max_jitter_iters: int = 8,
) -> FisherCodeResult:
    k = fisher.shape[0]
    eye = np.eye(k, dtype=float)
    # Strong engineering guard: always include ridge prior to avoid singularity exploit.
    reg = fisher + lambda_prior * eye
    current_jitter = jitter

    logdet = None
    mat = reg + current_jitter * eye
    for _ in range(max_jitter_iters):
        mat = reg + current_jitter * eye
        sign, ld = np.linalg.slogdet(mat)
        if sign > 0 and np.isfinite(ld):
            logdet = float(ld)
            break
        current_jitter *= 10.0
    if logdet is None:
        # final hard fallback
        mat = reg + (current_jitter + 1e-2) * eye
        sign, ld = np.linalg.slogdet(mat)
        logdet = float(ld if np.isfinite(ld) else 0.0)

    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals = np.maximum(eigvals, 0.0)
    eig_max = float(np.max(eigvals)) if eigvals.size else 0.0
    eig_min = float(np.min(eigvals)) if eigvals.size else 0.0
    if eig_max <= 0.0:
        cond_number = float("inf")
        effective_rank = 0
    else:
        floor = max(eig_min, 1e-12)
        cond_number = float(eig_max / floor)
        effective_rank = int(np.sum(eigvals > eig_max * 1e-8))

    dl_uniform = 0.5 * k * np.log2(max(2, n_samples))
    dl_fisher = dl_uniform + 0.5 * (logdet / np.log(2))
    rank = int(np.linalg.matrix_rank(reg))
    return FisherCodeResult(
        dl_param_bits=float(dl_fisher),
        logdet=logdet,
        jitter_used=float(current_jitter),
        rank=rank,
        fisher_matrix=reg,
        eig_min=eig_min,
        eig_max=eig_max,
        cond_number=cond_number,
        effective_rank=effective_rank,
    )


def uniform_param_code_bits(k: int, n_samples: int) -> float:
    return float(0.5 * k * np.log2(max(2, n_samples)))
