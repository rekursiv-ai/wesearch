"""Every Chrome a test starts must die with the module that started it.

The pool keeps browsers WARM on purpose -- a reused Chrome answers in 0.130s
against 4.196s for a per-request launch -- so nothing closes one at the end of
a test. That is right for the pool and wrong for a test run, which otherwise
leaves every browser it opened resident.
"""

from __future__ import annotations

from pathlib import Path

import ast

import pytest
import yaml

from wesearch.lib.custom_json import DictCodec, ListCodec, StrCodec


PACKAGE_ROOT = Path(__file__).resolve().parent
CONFTEST = PACKAGE_ROOT / "conftest.py"


def _own_repo_config() -> Path | None:
    """Return this checkout's ``.pre-commit-config.yaml``, or ``None``.

    Searched upward rather than counted: this file is copied into the exported
    package, where ``wesearch/`` sits at the checkout root instead of under
    ``loop/``, so a fixed ``parents[2]`` resolved to ``/tmp`` and the export
    died reading a config that exists only in the monorepo.

    Bounded by the enclosing REPOSITORY, and that bound is the point. An
    unbounded walk reaches ``/`` and adopts the first config it finds anywhere
    above -- verified: an unrelated one two levels up was picked up, which in
    an installed package would be a stranger's, and the assertions below would
    then be made against hooks wesearch does not own.
    """
    for candidate in [PACKAGE_ROOT, *PACKAGE_ROOT.parents]:
        config = candidate / ".pre-commit-config.yaml"
        if config.is_file():
            return config
        if (candidate / ".git").exists():
            return None
    return None


_CONFIG = _own_repo_config()


def test_the_conftest_imports_at_module_scope() -> None:
    """Fixtures import at the top, like every other module.

    STYLE.md admits an inline import in exactly two places: a CLI
    ``_parse_args`` helper, and a MEASURED ``lazy_import`` win. A fixture body
    is neither, and the ``PLC0415`` suppression that hid one here was justified
    by a guess about import cost rather than a measurement -- which is also what
    the comment rule rejects.
    """
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"), str(CONFTEST))
    inline = [
        f"line {node.lineno}: {ast.unparse(node)}"
        for parent in ast.walk(tree)
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(parent)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    assert inline == [], f"{CONFTEST.name} imports inside a function: {inline}"


def test_the_package_conftest_binds_browser_teardown() -> None:
    """An autouse module fixture must close the pool, not a per-test ``finally``.

    A ``finally`` protects the one test that writes it. Binding teardown at the
    package conftest is what stops a NEW test file reintroducing the leak by
    omission.

    Parsed, never grepped. The conftest's own docstring names ``autouse``, the
    scope, and ``shutdown_browsers`` while explaining them, so every substring
    this asserts is present in prose alone: gutting the fixture body to a bare
    ``yield`` left three substring checks passing against a fixture that reaps
    nothing.
    """
    fixture = _fixture_def(CONFTEST, name="close_pooled_browsers")

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
    assert _calls(fixture, "shutdown_browsers"), (
        "wesearch/conftest.py names shutdown_browsers but never calls it; a "
        "test file that launches Chrome now leaks for the whole run"
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


def _calls(tree: ast.AST, name: str) -> bool:
    """Whether ``tree`` contains a call to the bare function ``name``."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        for node in ast.walk(tree)
    )


@pytest.mark.compute_large_fixture
def test_no_test_file_relies_on_its_own_browser_teardown() -> None:
    """With the fixture in place, a per-file ``shutdown_browsers`` is a trap.

    It reads as sufficient and is not: the next file to launch a browser and
    omit the call leaks exactly as before. One owner, so there is a single
    place that can be wrong.
    """
    # Parsed, not grepped: this file and a comment elsewhere both mention the
    # call by name, and a substring search reports those as violations. Only a
    # real Call node is one.
    offenders = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in sorted(PACKAGE_ROOT.rglob("*_test.py"))
        if path != Path(__file__)
        and _calls(
            ast.parse(path.read_text(encoding="utf-8"), str(path)), "shutdown_browsers"
        )
    ]

    assert offenders == [], (
        f"these files close browsers themselves; the autouse fixture in "
        f"conftest.py owns that now: {offenders}"
    )


def test_a_file_marking_an_xdist_group_keeps_the_scheduler_that_honors_it() -> None:
    """``xdist_group`` is inert unless the run schedules by group.

    ``pyproject.toml`` sets ``--dist=worksteal`` for every run, under which
    ``xdist_group`` is IGNORED -- so a file that marks one is relying on the
    integration hook's ``--dist=loadgroup`` to override it. Losing that flag
    breaks nothing visibly: the tests still pass, spread across workers, and
    the grouping the marker asked for silently stops happening.

    Asserted only for the files that actually mark a group. Claiming it protects
    every browser test would be false -- ``parity_integration_test`` marks none,
    so ``loadgroup`` falls back to per-test scope there and its tests may land
    on any worker.
    """
    if _CONFIG is None:
        pytest.skip("no .pre-commit-config.yaml above this package")
    marked = [
        path.name
        for path in sorted(PACKAGE_ROOT.rglob("*_test.py"))
        if "xdist_group" in path.read_text(encoding="utf-8") and path != Path(__file__)
    ]
    assert marked, "no test file marks an xdist group; this guard is vacuous"

    config = DictCodec.coerce(yaml.safe_load(_CONFIG.read_text(encoding="utf-8")))
    integration = [
        hook
        for repo in ListCodec.mappings(config.get("repos", []))
        for hook in ListCodec.mappings(repo.get("hooks", []))
        if StrCodec.coerce(hook.get("id")) == "pytest-integration-global"
    ]
    assert integration, "no pytest-integration-global hook"
    script = "\n".join(ListCodec.coerce(integration[0].get("args", []), str))

    assert "--dist=loadgroup" in script, (
        f"{marked} mark an xdist group, but the integration hook no longer "
        f"passes --dist=loadgroup, so the marker is inert: {script}"
    )


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
