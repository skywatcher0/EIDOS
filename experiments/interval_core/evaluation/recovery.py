from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np


OPS = {"add", "sub", "mul", "div", "pow", "neg", "sin", "cos", "exp", "log", "sqrt", "clip"}


@dataclass(frozen=True)
class RecoveryDiagnostics:
    exact: bool
    semantic_nrmse: float
    semantic_max_scaled_error: float
    selected_variables: frozenset[str]
    selected_operators: frozenset[str]
    reason: str


def _truth(dataset_id: str, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    if dataset_id == "feynman_like_oscillator":
        return 2.2 * np.sin(1.1 * x1) + 0.8 * np.cos(x2) - 0.2 * x3
    if dataset_id == "feynman_like_lorentz":
        return (x1 * x2) / (1.0 + x3**2)
    if dataset_id == "feynman_like_relativistic":
        return x1 / np.sqrt(np.maximum(1.0 - (x2 / 3.2) ** 2, 0.06)) + 0.1 * x3
    if dataset_id == "feynman_like_kepler":
        return np.sqrt(np.maximum(x1, 0.1) / np.maximum(x2**3, 0.05))
    if dataset_id == "feynman_like_blackbody_proxy":
        return (x1**3) / (np.exp(np.clip(x2, 0.0, 5.0)) - 1.0 + 1e-2) + 0.05 * x3
    raise KeyError(f"unknown known-equation task: {dataset_id}")


def _eval_node(node: ast.AST, values: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, values)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return np.asarray(float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand, values)
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("unsupported serialized-expression node")
    name = node.func.id
    if name not in OPS:
        raise ValueError(f"unsupported operation: {name}")
    args = [_eval_node(arg, values) for arg in node.args]
    if name == "add":
        return args[0] + args[1]
    if name == "sub":
        return args[0] - args[1]
    if name == "mul":
        return args[0] * args[1]
    if name == "div":
        return args[0] / args[1]
    if name == "pow":
        return np.power(args[0], args[1])
    if name == "neg":
        return -args[0]
    if name == "sin":
        return np.sin(args[0])
    if name == "cos":
        return np.cos(args[0])
    if name == "exp":
        return np.exp(np.clip(args[0], -50.0, 50.0))
    if name == "log":
        return np.log(np.maximum(args[0], 1e-12))
    if name == "sqrt":
        return np.sqrt(np.maximum(args[0], 0.0))
    if name == "clip":
        return np.clip(args[0], args[1], args[2])
    raise AssertionError(name)


def _symbols(tree: ast.AST) -> tuple[frozenset[str], frozenset[str]]:
    variables: set[str] = set()
    operators: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            operators.add(node.func.id)
        elif isinstance(node, ast.Name) and node.id.startswith("x"):
            variables.add(node.id)
    return frozenset(variables), frozenset(operators)


def evaluate_known_equation_expression(
    dataset_id: str,
    expression: str,
    *,
    n_probe: int = 2048,
    nrmse_tolerance: float = 0.02,
    max_scaled_tolerance: float = 0.08,
) -> RecoveryDiagnostics:
    if not expression.strip():
        return RecoveryDiagnostics(False, float("inf"), float("inf"), frozenset(), frozenset(), "missing_expression")
    rng = np.random.default_rng(20260719)
    x1 = rng.uniform(0.2, 3.7, size=n_probe)
    x2 = rng.uniform(0.1, 2.8, size=n_probe)
    x3 = rng.normal(0.0, 1.3, size=n_probe)
    tree = ast.parse(expression, mode="eval")
    variables, operators = _symbols(tree)
    try:
        predicted = np.broadcast_to(
            _eval_node(tree, {"x1": x1, "x2": x2, "x3": x3}),
            x1.shape,
        ).astype(float)
        target = _truth(dataset_id, x1, x2, x3)
        if not np.isfinite(predicted).all():
            raise ValueError("nonfinite_prediction")
        scale = max(float(np.std(target)), 1e-12)
        error = predicted - target
        nrmse = float(np.sqrt(np.mean(error**2)) / scale)
        max_scaled = float(np.max(np.abs(error)) / scale)
    except Exception as exc:
        return RecoveryDiagnostics(False, float("inf"), float("inf"), variables, operators, f"evaluation_failed:{exc}")
    exact = bool(nrmse <= nrmse_tolerance and max_scaled <= max_scaled_tolerance)
    reason = "semantic_equivalence_within_declared_tolerance" if exact else "semantic_mismatch"
    return RecoveryDiagnostics(exact, nrmse, max_scaled, variables, operators, reason)
