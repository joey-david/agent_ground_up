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


def annotation(node: ast.expr | None) -> str | None:
    return None if node is None else ast.unparse(node)


def function_contract(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple:
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = [False] * (len(positional) - len(node.args.defaults)) + [True] * len(node.args.defaults)
    positional_contract = tuple(
        (arg.arg, annotation(arg.annotation), has_default)
        for arg, has_default in zip(positional, defaults, strict=True)
    )
    kwonly = tuple(
        (arg.arg, annotation(arg.annotation), default is not None)
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
    )
    vararg = None if node.args.vararg is None else (node.args.vararg.arg, annotation(node.args.vararg.annotation))
    kwarg = None if node.args.kwarg is None else (node.args.kwarg.arg, annotation(node.args.kwarg.annotation))
    return positional_contract, kwonly, vararg, kwarg, annotation(node.returns)


def contracts(path: Path) -> dict[str, tuple]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, tuple] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = function_contract(node)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f"{node.name}.{child.name}"] = function_contract(child)
    return result


def test_all_python_signatures_and_types_match() -> None:
    for filename in MODULES:
        expected = contracts(REFERENCE / "kernel" / filename)
        actual = contracts(TARGET / "kernel" / filename)
        assert actual == expected, f"signature mismatch in {filename}"
