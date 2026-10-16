"""Package-level pytest fixtures for wesearch.

Binds the autouse XDG isolation fixture at the package root: wesearch keeps
cookies, the Chrome profile, and caches under ``wesearch.lib.userdirs``, and an
unisolated test would write into the developer's real directories. Each test
also gets its own default session, so no egress or connection leaks between tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wesearch.fetch.session import close_default_session
from wesearch.fetch.transport.zendriver import BrowserPool
from wesearch.lib.testing.resources import (
    isolate_user_dirs,
    pytest_collection_modifyitems,
    pytest_configure,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


# Re-exported: an autouse fixture reaches only its conftest's directory.
__all__ = [
    "fresh_default_session",
    "isolate_user_dirs",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "stop_launched_chromes",
]


@pytest.fixture(scope="module", autouse=True)
def stop_launched_chromes() -> Iterator[None]:
    """Stop every Chrome a test module launched.

    Module scope: 22 Chromes left running until session end starved a sibling
    suite of memory.
    """
    yield
    BrowserPool().shutdown(stop_launched=True)


@pytest.fixture(autouse=True)
def fresh_default_session() -> Iterator[None]:
    """Give each test its own default session, closed when the test ends."""
    close_default_session()
    yield
    close_default_session()
