"""Defaults shared by every fetch transport."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import inspect

import pytest

from wesearch.fetch.transport.curl import fetch_curl
from wesearch.fetch.transport.stdlib import fetch_stdlib
from wesearch.fetch.transport.zendriver import fetch_zendriver
from wesearch.fetch.types import Fetch


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("transport", [fetch_curl, fetch_stdlib, fetch_zendriver])
def test_every_fetch_argument_defaults_like_the_protocol(transport: Fetch) -> None:
    def defaults(function: Callable[..., object]) -> dict[str, object]:
        return {
            name: cast(object, parameter.default)
            for name, parameter in inspect.signature(function).parameters.items()
            if name not in {"self", "url"}
        }

    shared = defaults(Fetch.__call__)
    assert inspect.Parameter.empty not in shared.values()
    assert defaults(transport) == shared


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
