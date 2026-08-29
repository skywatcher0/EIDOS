from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

@dataclass(frozen=True)
class QuantizedParameters:
    values: np.ndarray
    indices: np.ndarray
    steps: np.ndarray
    code_bits: float


def elias_delta_bits(n: int) -> int:
    """Return the Elias-delta code length for a positive integer."""

    value = int(n)
    if value != n or value < 1:
        raise ValueError("Elias-delta coding requires an integer n >= 1")
    k = value.bit_length() - 1
    m = (k + 1).bit_length() - 1
    return k + 2 * m + 1


def param_codelength_bits(q: Sequence[int] | np.ndarray) -> float:
    """Code signed grid indices with a sign bit and Elias-delta magnitude."""

    indices = np.asarray(q)
    if indices.ndim == 0:
        indices = indices.reshape(1)
    flat = indices.reshape(-1)
    if not np.issubdtype(flat.dtype, np.integer):
        as_float = flat.astype(float)
        if np.any(~np.isfinite(as_float)) or np.any(as_float != np.floor(as_float)):
            raise ValueError("parameter indices must be finite integers")
    return float(sum(1 + elias_delta_bits(abs(int(qi)) + 1) for qi in flat))


def quantize_parameters(
    params: np.ndarray,
    steps: np.ndarray | float,
) -> QuantizedParameters:
    """Quantize parameters on a fixed grid and code their integer indices.

    ``steps`` is part of the protocol and must be fixed before fitting.  The
    sign-plus-Elias-delta code is prefix-decodable and grows logarithmically
    with the magnitude of each transmitted grid index.
    """

    theta = np.asarray(params, dtype=float).reshape(-1)
    step = np.broadcast_to(np.asarray(steps, dtype=float), theta.shape).copy()
    if np.any(~np.isfinite(theta)):
        raise ValueError("parameters must be finite")
    if np.any(~np.isfinite(step)) or np.any(step <= 0):
        raise ValueError("every quantization step must be finite and positive")

    indices = np.floor(theta / step + 0.5).astype(np.int64)
    values = indices.astype(float) * step
    bits = param_codelength_bits(indices)
    return QuantizedParameters(values=values, indices=indices, steps=step, code_bits=bits)


def fisher_stress_steps(
    fisher: np.ndarray,
    base_step: float,
    ridge: float = 1e-6,
    min_multiplier: float = 0.125,
    max_multiplier: float = 8.0,
) -> np.ndarray:
    """Representation-dependent perturbation grid for a stress diagnostic.

    These steps are not used as a strict parameter code.  They are exposed so
    the Fisher-conditioned sensitivity analysis cannot be confused with the
    fixed-grid code above.
    """

    matrix = np.asarray(fisher, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("fisher must be a square matrix")
    if base_step <= 0 or ridge <= 0:
        raise ValueError("base_step and ridge must be positive")
    diag = np.maximum(np.diag(matrix), 0.0) + ridge
    raw = 1.0 / np.sqrt(diag)
    median = float(np.median(raw)) if raw.size else 1.0
    scaled = raw / max(median, np.finfo(float).tiny)
    multipliers = np.clip(scaled, min_multiplier, max_multiplier)
    return float(base_step) * multipliers
