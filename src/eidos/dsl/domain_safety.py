from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .grammar import Expr


@dataclass
class Interval:
    lo: float
    hi: float
    valid: bool = True
    definitely_invalid: bool = False
    maybe_invalid: bool = False
    reason: str | None = None


@dataclass
class RuntimeEvalResult:
    values: np.ndarray
    clip_ratio: float
    clip_events: dict[str, int]
    failure_reason: str | None = None


def _combine(a: Interval, b: Interval, op: str) -> Interval:
    if not a.valid:
        return a
    if not b.valid:
        return b

    if op == "add":
        return Interval(a.lo + b.lo, a.hi + b.hi)
    if op == "sub":
        return Interval(a.lo - b.hi, a.hi - b.lo)
    if op == "mul":
        vals = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
        return Interval(float(min(vals)), float(max(vals)))
    if op == "div":
        # Conservative rule: only reject if denominator is guaranteed zero.
        if abs(b.lo) < 1e-12 and abs(b.hi) < 1e-12:
            return Interval(0.0, 0.0, valid=False, definitely_invalid=True, reason="domain_error")
        # If zero may be crossed, keep candidate for runtime clipping.
        if b.lo <= 0.0 <= b.hi:
            return Interval(-np.inf, np.inf, maybe_invalid=True, reason="domain_error")
        vals = [a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi]
        return Interval(float(min(vals)), float(max(vals)))
    if op == "pow":
        # For general real powers, static interval is rough and conservative.
        if b.lo < 0 and a.lo <= 0:
            return Interval(-np.inf, np.inf, maybe_invalid=True, reason="domain_error")
        return Interval(-np.inf, np.inf, maybe_invalid=True)
    raise ValueError(f"unsupported interval op: {op}")


def interval_analysis(expr: Expr, var_intervals: Mapping[str, tuple[float, float]]) -> Interval:
    if expr.op == "const":
        v = float(expr.value)
        return Interval(v, v)
    if expr.op == "bool":
        v = 1.0 if bool(expr.value) else 0.0
        return Interval(v, v)
    if expr.op == "var":
        lo, hi = var_intervals.get(expr.name or "", (-np.inf, np.inf))
        return Interval(float(lo), float(hi))

    if expr.op in {"add", "sub", "mul", "div", "pow"}:
        a = interval_analysis(expr.args[0], var_intervals)
        b = interval_analysis(expr.args[1], var_intervals)
        return _combine(a, b, expr.op)

    child = interval_analysis(expr.args[0], var_intervals)
    if not child.valid:
        return child

    if expr.op == "neg":
        return Interval(-child.hi, -child.lo)
    if expr.op in {"sin", "cos"}:
        return Interval(-1.0, 1.0)
    if expr.op == "softplus":
        return Interval(max(0.0, child.lo), max(0.0, child.hi))
    if expr.op == "exp":
        if child.lo == -np.inf:
            return Interval(0.0, np.inf, maybe_invalid=True)
        lo_c = float(np.clip(child.lo, -50.0, 50.0))
        hi_c = float(np.clip(child.hi, -50.0, 50.0))
        return Interval(float(np.exp(lo_c)), float(np.exp(hi_c)), maybe_invalid=(child.lo < -50.0 or child.hi > 50.0))
    if expr.op == "log":
        if child.hi <= 0:
            return Interval(0.0, 0.0, valid=False, definitely_invalid=True, reason="domain_error")
        if child.lo <= 0:
            return Interval(-np.inf, np.log(child.hi), maybe_invalid=True, reason="domain_error")
        return Interval(float(np.log(child.lo)), float(np.log(child.hi)))

    if expr.op == "if":
        a = interval_analysis(expr.args[1], var_intervals)
        b = interval_analysis(expr.args[2], var_intervals)
        return Interval(min(a.lo, b.lo), max(a.hi, b.hi), maybe_invalid=a.maybe_invalid or b.maybe_invalid)

    if expr.op == "clip":
        x = interval_analysis(expr.args[0], var_intervals)
        lo = interval_analysis(expr.args[1], var_intervals)
        hi = interval_analysis(expr.args[2], var_intervals)
        return Interval(max(x.lo, lo.lo), min(x.hi, hi.hi), maybe_invalid=x.maybe_invalid or lo.maybe_invalid or hi.maybe_invalid)

    return Interval(-np.inf, np.inf, maybe_invalid=True)


def _safe_div(n: np.ndarray, d: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    mask = np.abs(d) < eps
    d_safe = np.where(mask, np.sign(d) * eps + (d == 0) * eps, d)
    return n / d_safe, int(mask.sum())


def safe_eval(
    expr: Expr,
    variables: Mapping[str, np.ndarray],
    eps: float = 1e-8,
    exp_clip: float = 20.0,
) -> RuntimeEvalResult:
    clip_events: dict[str, int] = {"log": 0, "exp": 0, "div": 0}
    n_total = 0

    def rec(node: Expr) -> np.ndarray:
        nonlocal n_total
        if node.op == "const":
            base = next(iter(variables.values()))
            return np.full_like(base, float(node.value), dtype=float)
        if node.op == "bool":
            base = next(iter(variables.values()))
            return np.full_like(base, float(bool(node.value)), dtype=float)
        if node.op == "var":
            return np.asarray(variables[node.name or ""], dtype=float)

        if node.op in {"add", "sub", "mul", "pow"}:
            a = rec(node.args[0])
            b = rec(node.args[1])
            n_total += a.size
            if node.op == "add":
                return a + b
            if node.op == "sub":
                return a - b
            if node.op == "mul":
                return a * b
            return np.power(a, b)

        if node.op == "div":
            a = rec(node.args[0])
            b = rec(node.args[1])
            out, n_clip = _safe_div(a, b, eps)
            clip_events["div"] += n_clip
            n_total += a.size
            return out

        x = rec(node.args[0])
        n_total += x.size
        if node.op == "neg":
            return -x
        if node.op == "sin":
            return np.sin(x)
        if node.op == "cos":
            return np.cos(x)
        if node.op == "log":
            mask = x <= eps
            clip_events["log"] += int(mask.sum())
            return np.log(np.maximum(x, eps))
        if node.op == "exp":
            x_clip = np.clip(x, -exp_clip, exp_clip)
            clip_events["exp"] += int((x_clip != x).sum())
            return np.exp(x_clip)
        if node.op == "softplus":
            x_clip = np.clip(x, -exp_clip, exp_clip)
            clip_events["exp"] += int((x_clip != x).sum())
            return np.log1p(np.exp(x_clip))
        if node.op == "if":
            cond = x > 0.0
            a = rec(node.args[1])
            b = rec(node.args[2])
            return np.where(cond, a, b)
        if node.op == "clip":
            lo = rec(node.args[1])
            hi = rec(node.args[2])
            x_clip = np.minimum(np.maximum(x, lo), hi)
            clip_events["exp"] += int((x_clip != x).sum())
            return x_clip
        raise ValueError(f"unknown node op: {node.op}")

    try:
        values = rec(expr)
    except Exception:
        return RuntimeEvalResult(
            values=np.full_like(next(iter(variables.values())), np.nan, dtype=float),
            clip_ratio=1.0,
            clip_events=clip_events,
            failure_reason="domain_error",
        )

    total_clips = sum(clip_events.values())
    denom = max(1, n_total)
    return RuntimeEvalResult(
        values=values,
        clip_ratio=float(total_clips / denom),
        clip_events=clip_events,
        failure_reason=None,
    )
