from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .grammar import Expr


UNIT_DIMENSIONLESS: dict[str, float] = {}


def _cleanup_unit(unit: Mapping[str, float]) -> dict[str, float]:
    out = {k: float(v) for k, v in unit.items() if abs(float(v)) > 1e-12}
    return dict(sorted(out.items()))


def unit_mul(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + float(v)
    return _cleanup_unit(out)


def unit_div(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) - float(v)
    return _cleanup_unit(out)


def unit_equal(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    return _cleanup_unit(a) == _cleanup_unit(b)


@dataclass
class TypeUnit:
    kind: str
    unit: dict[str, float]
    valid: bool
    reason: str | None = None


def infer_type_unit(
    expr: Expr,
    var_types: Mapping[str, str] | None = None,
    var_units: Mapping[str, Mapping[str, float]] | None = None,
) -> TypeUnit:
    var_types = var_types or {}
    var_units = var_units or {}

    if expr.op == "const":
        return TypeUnit(kind="Real", unit=UNIT_DIMENSIONLESS, valid=True)
    if expr.op == "bool":
        return TypeUnit(kind="Bool", unit=UNIT_DIMENSIONLESS, valid=True)
    if expr.op == "var":
        kind = var_types.get(expr.name or "", "Real")
        unit = dict(var_units.get(expr.name or "", UNIT_DIMENSIONLESS))
        return TypeUnit(kind=kind, unit=unit, valid=True)

    child = [infer_type_unit(a, var_types=var_types, var_units=var_units) for a in expr.args]
    invalid_child = next((c for c in child if not c.valid), None)
    if invalid_child is not None:
        return invalid_child

    if expr.op in {"sin", "cos", "exp", "log", "softplus", "neg"}:
        c = child[0]
        if c.kind != "Real":
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "type_mismatch")
        if expr.op in {"sin", "cos", "exp", "log", "softplus"} and not unit_equal(
            c.unit, UNIT_DIMENSIONLESS
        ):
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "unit_mismatch")
        return TypeUnit("Real", dict(c.unit), True)

    if expr.op in {"add", "sub"}:
        a, b = child
        if a.kind != "Real" or b.kind != "Real":
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "type_mismatch")
        if not unit_equal(a.unit, b.unit):
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "unit_mismatch")
        return TypeUnit("Real", dict(a.unit), True)

    if expr.op == "mul":
        a, b = child
        if a.kind != "Real" or b.kind != "Real":
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "type_mismatch")
        return TypeUnit("Real", unit_mul(a.unit, b.unit), True)

    if expr.op == "div":
        a, b = child
        if a.kind != "Real" or b.kind != "Real":
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "type_mismatch")
        return TypeUnit("Real", unit_div(a.unit, b.unit), True)

    if expr.op == "pow":
        a, b = child
        if a.kind != "Real" or b.kind != "Real":
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "type_mismatch")
        if not unit_equal(b.unit, UNIT_DIMENSIONLESS):
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "unit_mismatch")
        return TypeUnit("Real", UNIT_DIMENSIONLESS if unit_equal(a.unit, UNIT_DIMENSIONLESS) else a.unit, True)

    if expr.op == "if":
        cond, t_expr, f_expr = child
        if cond.kind != "Bool":
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "type_mismatch")
        if t_expr.kind != f_expr.kind:
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "type_mismatch")
        if not unit_equal(t_expr.unit, f_expr.unit):
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "unit_mismatch")
        return TypeUnit(t_expr.kind, dict(t_expr.unit), True)

    if expr.op == "clip":
        x, lo, hi = child
        if x.kind != "Real" or lo.kind != "Real" or hi.kind != "Real":
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "type_mismatch")
        if not (unit_equal(x.unit, lo.unit) and unit_equal(x.unit, hi.unit)):
            return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "unit_mismatch")
        return TypeUnit("Real", dict(x.unit), True)

    return TypeUnit("Real", UNIT_DIMENSIONLESS, False, "type_mismatch")
