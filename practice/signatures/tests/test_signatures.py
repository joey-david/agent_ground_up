from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).resolve()
TARGET = HERE.parents[1]
REFERENCE = HERE.parents[2] / "implementation"
MODULES = [
    "tools.py",
    "memory.py",
    "experience.py",
    "skills.py",
    "agent.py",
    "tasks.py",
    "evaluate.py",
    "archive.py",
    "improve.py",
    "loss.py",
]
REQUIRED = "<required>"


def annotation(node: ast.expr | None) -> str | None:
    return None if node is None else ast.unparse(node)


def default(node: ast.expr | None) -> str:
    return REQUIRED if node is None else ast.unparse(node)


def function_contract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple:
    positional = [*node.args.posonlyargs, *node.args.args]
    positional_defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(node.args.defaults)
    ) + list(node.args.defaults)
    positional_contract = tuple(
        (arg.arg, annotation(arg.annotation), default(value))
        for arg, value in zip(positional, positional_defaults, strict=True)
    )
    kwonly = tuple(
        (arg.arg, annotation(arg.annotation), default(value))
        for arg, value in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
    )
    vararg = (
        None
        if node.args.vararg is None
        else (node.args.vararg.arg, annotation(node.args.vararg.annotation))
    )
    kwarg = (
        None
        if node.args.kwarg is None
        else (node.args.kwarg.arg, annotation(node.args.kwarg.annotation))
    )
    return positional_contract, kwonly, vararg, kwarg, annotation(node.returns)


def class_contract(node: ast.ClassDef) -> tuple:
    bases = tuple(ast.unparse(base) for base in node.bases)
    fields = tuple(
        (
            child.target.id,
            annotation(child.annotation),
            default(child.value),
        )
        for child in node.body
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
    )
    methods = tuple(
        (child.name, function_contract(child))
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return bases, fields, methods


def contracts(path: Path) -> dict[str, tuple]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, tuple] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[f"function:{node.name}"] = function_contract(node)
        elif isinstance(node, ast.ClassDef):
            result[f"class:{node.name}"] = class_contract(node)
    return result


def test_all_python_interfaces_types_and_defaults_match() -> None:
    for filename in MODULES:
        expected = contracts(REFERENCE / "kernel" / filename)
        actual = contracts(TARGET / "kernel" / filename)
        assert actual == expected, f"interface mismatch in {filename}"
