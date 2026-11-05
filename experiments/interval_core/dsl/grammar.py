from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


UNARY_OPS = {"sin", "cos", "exp", "log", "softplus", "neg"}
BINARY_OPS = {"add", "sub", "mul", "div", "pow"}
TERNARY_OPS = {"if", "clip"}
LEAF_OPS = {"const", "var", "bool"}
ALL_OPS = UNARY_OPS | BINARY_OPS | TERNARY_OPS | LEAF_OPS


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    value: float | bool | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.op not in ALL_OPS:
            raise ValueError(f"Unknown op: {self.op}")

    def to_string(self) -> str:
        if self.op == "const":
            return f"{float(self.value):.6g}"
        if self.op == "bool":
            return "true" if bool(self.value) else "false"
        if self.op == "var":
            return str(self.name)
        if self.op in UNARY_OPS:
            return f"{self.op}({self.args[0].to_string()})"
        if self.op in BINARY_OPS:
            left, right = self.args
            return f"{self.op}({left.to_string()},{right.to_string()})"
        if self.op == "if":
            c, a, b = self.args
            return f"if({c.to_string()},{a.to_string()},{b.to_string()})"
        if self.op == "clip":
            x, lo, hi = self.args
            return f"clip({x.to_string()},{lo.to_string()},{hi.to_string()})"
        raise ValueError(f"Unhandled op: {self.op}")


def Const(value: float) -> Expr:
    return Expr(op="const", value=float(value))


def Bool(value: bool) -> Expr:
    return Expr(op="bool", value=bool(value))


def Var(name: str) -> Expr:
    return Expr(op="var", name=name)


def op1(name: str, x: Expr) -> Expr:
    return Expr(op=name, args=(x,))


def op2(name: str, x: Expr, y: Expr) -> Expr:
    return Expr(op=name, args=(x, y))


def op3(name: str, x: Expr, y: Expr, z: Expr) -> Expr:
    return Expr(op=name, args=(x, y, z))


def default_runtime_ops() -> dict[str, Callable[..., np.ndarray]]:
    return {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "pow": lambda a, b: np.power(a, b),
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "softplus": lambda x: np.log1p(np.exp(x)),
        "neg": lambda x: -x,
        "if": lambda cond, a, b: np.where(cond > 0, a, b),
        "clip": np.clip,
    }

