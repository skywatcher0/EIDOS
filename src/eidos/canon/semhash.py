from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from eidos.dsl.domain_safety import safe_eval
from eidos.dsl.grammar import Expr


@dataclass
class SemHashResult:
    sem_hash: str
    vector: np.ndarray
    clip_ratio: float
    failure_reason: str | None = None


def _robust_scale(v: np.ndarray) -> np.ndarray:
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) + 1e-9
    return (v - med) / mad


def semantic_hash(
    expr: Expr,
    probes: Mapping[str, np.ndarray],
    tol: float = 5e-2,
    eps: float = 1e-8,
    exp_clip: float = 20.0,
) -> SemHashResult:
    out = safe_eval(expr, probes, eps=eps, exp_clip=exp_clip)
    vec = np.asarray(out.values, dtype=float)
    if np.isnan(vec).any() or np.isinf(vec).any():
        return SemHashResult(
            sem_hash="invalid",
            vector=vec,
            clip_ratio=out.clip_ratio,
            failure_reason=out.failure_reason or "domain_error",
        )
    z = _robust_scale(vec)
    q = np.round(z / tol).astype(np.int64)
    digest = hashlib.sha1(q.tobytes()).hexdigest()
    return SemHashResult(sem_hash=digest, vector=vec, clip_ratio=out.clip_ratio)


def dedup_by_semhash(
    exprs: Iterable[Expr],
    probes: Mapping[str, np.ndarray],
    tol: float = 5e-2,
) -> tuple[list[Expr], dict[str, list[str]], list[dict[str, str]]]:
    keep: list[Expr] = []
    buckets: dict[str, list[str]] = {}
    collisions: list[dict[str, str]] = []
    seen: dict[str, Expr] = {}
    for expr in exprs:
        r = semantic_hash(expr, probes=probes, tol=tol)
        eid = expr.to_string()
        buckets.setdefault(r.sem_hash, []).append(eid)
        if r.sem_hash not in seen:
            seen[r.sem_hash] = expr
            keep.append(expr)
        else:
            prev = seen[r.sem_hash]
            if prev.to_string() != eid:
                collisions.append({"hash": r.sem_hash, "a": prev.to_string(), "b": eid})
    return keep, buckets, collisions


def save_dedup_report(
    path: str,
    n_before: int,
    n_after: int,
    collisions: list[dict[str, str]],
) -> None:
    payload = {
        "n_before": n_before,
        "n_after": n_after,
        "collision_rate": 0.0 if n_before == 0 else (n_before - n_after) / n_before,
        "collisions": collisions,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

