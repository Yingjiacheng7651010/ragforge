"""Agent data model and the safe calculator tool."""

import ast
import math
import operator
from dataclasses import dataclass, field
from typing import Any

_TOOLS = ("retrieve", "calculator", "lookup_table")

_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "pow": pow,
}


@dataclass
class Step:
    """One planned step; ``result`` is filled in by the executor."""

    id: str
    tool: str
    query: str
    result: str | None = field(default=None)


def safe_calc(expression: str) -> float:
    """Evaluate a math expression with an AST whitelist (no arbitrary code)."""
    return _eval_node(ast.parse(expression, mode="eval").body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return float(_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        args = [_eval_node(arg) for arg in node.args]
        return float(_FUNCS[node.func.id](*args))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")
