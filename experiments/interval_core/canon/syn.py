from __future__ import annotations

from typing import Iterable

from experiments.interval_core.dsl.grammar import Const, Expr


COMMUTATIVE = {"add", "mul"}
ASSOCIATIVE = {"add", "mul"}


def _is_const(node: Expr) -> bool:
    return node.op == "const"


def _flatten(op: str, args: Iterable[Expr]) -> list[Expr]:
    out: list[Expr] = []
    for a in args:
        if a.op == op:
            out.extend(a.args)
        else:
            out.append(a)
    return out


def _fold_constants(op: str, args: list[Expr]) -> list[Expr]:
    if op == "add":
        csum = sum(float(a.value) for a in args if _is_const(a))
        rest = [a for a in args if not _is_const(a)]
        if abs(csum) > 1e-12 or not rest:
            rest.append(Const(csum))
        return rest
    if op == "mul":
        cprod = 1.0
        rest = []
        for a in args:
            if _is_const(a):
                cprod *= float(a.value)
            else:
                rest.append(a)
        if abs(cprod) < 1e-12:
            return [Const(0.0)]
        if abs(cprod - 1.0) > 1e-12 or not rest:
            rest.append(Const(cprod))
        return rest
    return args


def _make_nary(op: str, args: list[Expr]) -> Expr:
    if len(args) == 1:
        return args[0]
    node = Expr(op=op, args=(args[0], args[1]))
    for i in range(2, len(args)):
        node = Expr(op=op, args=(node, args[i]))
    return node


def canonicalize_syn(expr: Expr) -> Expr:
    if expr.op in {"const", "var", "bool"}:
        return expr

    cargs = [canonicalize_syn(a) for a in expr.args]

    if expr.op in ASSOCIATIVE:
        cargs = _flatten(expr.op, cargs)

    if expr.op in COMMUTATIVE:
        cargs = sorted(cargs, key=lambda e: e.to_string())

    if expr.op in {"add", "mul"}:
        cargs = _fold_constants(expr.op, cargs)
        cargs = sorted(cargs, key=lambda e: e.to_string()) if expr.op in COMMUTATIVE else cargs
        return _make_nary(expr.op, cargs)

    # local simplifications
    if expr.op == "sub":
        a, b = cargs
        if b.op == "const" and abs(float(b.value)) < 1e-12:
            return a
        return Expr(op="sub", args=(a, b))

    if expr.op == "div":
        a, b = cargs
        if b.op == "const" and abs(float(b.value) - 1.0) < 1e-12:
            return a
        return Expr(op="div", args=(a, b))

    if expr.op == "neg":
        a = cargs[0]
        if a.op == "neg":
            return a.args[0]
        return Expr(op="neg", args=(a,))

    return Expr(op=expr.op, args=tuple(cargs), value=expr.value, name=expr.name)

