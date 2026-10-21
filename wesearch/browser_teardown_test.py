"""Every Chrome a test starts must die with the module that started it.

The pool keeps browsers WARM on purpose -- a reused Chrome answers in 0.130s
against 4.196s for a per-request launch -- so nothing closes one at the end of
a test. That is right for the pool and wrong for a test run, which otherwise
leaves every browser it opened resident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import ast

import pytest

from tools.precommit.tiers import TIERS


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent


def test_the_conftest_imports_at_module_scope() -> None:
    """Fixtures import at the top, like every other module.

    STYLE.md admits an inline import in exactly two places: a CLI
    ``_parse_args`` helper, and a MEASURED ``lazy_import`` win. A fixture body
    is neither, and the ``PLC0415`` suppression that hid one here was justified
    by a guess about import cost rather than a measurement -- which is also what
    the comment rule rejects.
    """
    conftest = _CWD / "conftest.py"
    tree = ast.parse(conftest.read_text(encoding="utf-8"), str(conftest))
    inline = [
        f"line {node.lineno}: {ast.unparse(node)}"
        for parent in ast.walk(tree)
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(parent)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    assert inline == [], f"{conftest.name} imports inside a function: {inline}"


def test_the_package_conftest_binds_browser_teardown() -> None:
    """An autouse module fixture must stop launched Chromes, not a per-test ``finally``.

    A ``finally`` protects the one test that writes it. Binding teardown at the
    package conftest is what stops a NEW test file reintroducing the leak by
    omission.

    Parsed, never grepped. The conftest's own docstring names the scope and
    what it stops while explaining them, so every substring this asserts is
    present in prose alone: gutting the fixture body to a bare ``yield`` left
    substring checks passing against a fixture that reaps nothing.
    """
    fixture = _fixture_def(_CWD / "conftest.py", name="stop_launched_chromes")

    decorator = next(
        node
        for node in fixture.decorator_list
        if isinstance(node, ast.Call) and _decorator_name(node) == "pytest.fixture"
    )
    keywords = {
        keyword.arg: keyword.value.value
        for keyword in decorator.keywords
        if isinstance(keyword.value, ast.Constant)
    }

    assert keywords.get("scope") == "module", (
        "browser teardown must be MODULE scoped: session scope reaps only "
        "after every later test has already run beside the browsers"
    )
    assert keywords.get("autouse") is True, (
        "browser teardown must be autouse, so it covers every test file "
        "without each one opting in"
    )
    assert _stops_launched(fixture), (
        "wesearch/conftest.py never calls BrowserPool().shutdown(stop_launched=True); "
        "the shared Chrome is detached and outlives a disconnect, so every "
        "isolated test profile leaves one running"
    )


def _fixture_def(path: Path, *, name: str) -> ast.FunctionDef:
    """Return the named function's definition in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} defines no {name!r}")


def _decorator_name(call: ast.Call) -> str:
    """Return a decorator's dotted name, e.g. ``pytest.fixture``."""
    return ast.unparse(call.func)


def _stops_launched(tree: ast.AST) -> bool:
    """Whether ``tree`` calls ``BrowserPool().shutdown(stop_launched=True)``."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shutdown"
        and isinstance(node.func.value, ast.Call)
        and ast.unparse(node.func.value.func) == "BrowserPool"
        and any(
            keyword.arg == "stop_launched"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
        for node in ast.walk(tree)
    )


@pytest.mark.compute_large_fixture
def test_no_test_file_relies_on_its_own_browser_teardown() -> None:
    """With the fixture in place, a per-file Chrome teardown is a trap.

    It reads as sufficient and is not: the next file to launch a browser and
    omit the call leaks exactly as before. One owner, so there is a single
    place that can be wrong.
    """
    # Parsed, not grepped: this file and a comment elsewhere both mention the
    # call by name, and a substring search reports those as violations. Only a
    # real Call node is one.
    # ``BrowserPool``'s own unit tests call the method as their subject, not as
    # teardown.
    owner = _CWD / "fetch" / "transport" / "zendriver_test.py"
    offenders = [
        str(path.relative_to(_CWD))
        for path in sorted(_CWD.rglob("*_test.py"))
        if path not in {_THIS, owner}
        and _stops_launched(ast.parse(path.read_text(encoding="utf-8"), str(path)))
    ]

    assert offenders == [], (
        f"these files stop launched Chromes themselves; the autouse fixture in "
        f"conftest.py owns that now: {offenders}"
    )


def test_a_file_marking_an_xdist_group_keeps_the_scheduler_that_honors_it() -> None:
    """``xdist_group`` is inert unless the run schedules by group.

    ``pyproject.toml`` sets ``--dist=worksteal`` for every run, under which
    ``xdist_group`` is IGNORED -- so a file that marks one is relying on the
    integration tier's ``--dist=loadgroup`` to override it. Losing that flag
    breaks nothing visibly: the tests still pass, spread across workers, and
    the grouping the marker asked for silently stops happening.

    Asserted only for the files that actually mark a group. Claiming it protects
    every browser test would be false -- ``parity_integration_test`` marks none,
    so ``loadgroup`` falls back to per-test scope there and its tests may land
    on any worker.
    """
    marked = [
        path.name
        for path in sorted(_CWD.rglob("*_test.py"))
        if "xdist_group" in path.read_text(encoding="utf-8") and path != _THIS
    ]
    assert marked, "no test file marks an xdist group; this guard is vacuous"
    extra = TIERS["pytest-integration-global"].extra

    assert "--dist=loadgroup" in extra, (
        f"{marked} mark an xdist group, but the integration tier no longer "
        f"passes --dist=loadgroup, so the marker is inert: {extra}"
    )


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
