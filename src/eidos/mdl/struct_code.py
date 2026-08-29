from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil, floor, log2

from src.eidos.dsl.grammar import ALL_OPS, Expr


STRICT_OP_ALPHABET = tuple(sorted(ALL_OPS))
STRICT_OP_WIDTH = int(ceil(log2(len(STRICT_OP_ALPHABET))))


@dataclass
class StructCodeResult:
    dl_struct_bits: float
    node_count: int
    op_hist: dict[str, int]


def elias_gamma_bits(value: int) -> int:
    """Length of the Elias-gamma code for a positive integer."""

    n = int(value)
    if n < 1:
        raise ValueError("Elias-gamma coding requires a positive integer")
    return 2 * floor(log2(n)) + 1


def _utf8_payload_bits(text: str) -> int:
    payload = text.encode("utf-8")
    return elias_gamma_bits(len(payload) + 1) + 8 * len(payload)


def _walk_strict(expr: Expr, hist: Counter[str]) -> tuple[int, int]:
    if expr.op not in STRICT_OP_ALPHABET:
        raise ValueError(f"operator is outside the fixed grammar: {expr.op}")
    hist[expr.op] += 1
    nodes = 1
    payload_bits = 0
    if expr.op == "const":
        payload_bits += 64
    elif expr.op == "var":
        if expr.name is None:
            raise ValueError("variable leaves require a name")
        payload_bits += _utf8_payload_bits(expr.name)
    elif expr.op == "bool":
        payload_bits += 1
    for child in expr.args:
        child_nodes, child_payload = _walk_strict(child, hist)
        nodes += child_nodes
        payload_bits += child_payload
    return nodes, payload_bits


def struct_code_bits(expr: Expr, op_bits: dict[str, float] | None = None) -> StructCodeResult:
    """Kraft-valid prefix length for a finite expression grammar.

    The code concatenates an Elias-gamma node count, fixed-width preorder
    operator tokens, and self-delimiting leaf payloads.  Operator arities make
    the tree uniquely decodable.  ``op_bits`` is rejected because arbitrary
    per-operator costs need not satisfy Kraft's inequality.
    """

    if op_bits is not None:
        raise ValueError("custom operator costs are not permitted in strict coding mode")
    hist: Counter[str] = Counter()
    node_count, payload_bits = _walk_strict(expr, hist)
    bits = elias_gamma_bits(node_count) + STRICT_OP_WIDTH * node_count + payload_bits
    return StructCodeResult(
        dl_struct_bits=float(bits),
        node_count=node_count,
        op_hist=dict(hist),
    )


def grammar_extension_code_bits(concepts: list[Expr]) -> float:
    """Prefix length for transmitting a finite list of concept definitions."""

    count_bits = elias_gamma_bits(len(concepts) + 1)
    return float(count_bits + sum(struct_code_bits(expr).dl_struct_bits for expr in concepts))


def grammar_code_bits(new_concepts: int, concept_cost: float = 24.0) -> float:
    """Legacy bookkeeping penalty retained for archived experiment replay.

    It is intentionally not described as a code in the revised manuscript.
    New strict analyses must pass definitions to ``grammar_extension_code_bits``.
    """

    if new_concepts <= 0:
        return 0.0
    return float(new_concepts * concept_cost + elias_gamma_bits(new_concepts))
