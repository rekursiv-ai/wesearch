"""The wesearch wheel, built and imported outside the checkout."""

from __future__ import annotations

import pytest

from wesearch.lib.testing.wheel import built_wheel


pytestmark = pytest.mark.cli_uv


def test_the_wheel_imports_outside_the_checkout() -> None:
    wheel = built_wheel("wesearch", anchor=__file__)

    assert "site-packages" in wheel.run("import wesearch; print(wesearch.__file__)")


def test_the_mcp_extra_serves_the_mcp_server() -> None:
    wheel = built_wheel("wesearch", anchor=__file__)

    wheel.run("import wesearch.mcp.server", extras=("mcp",))


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
