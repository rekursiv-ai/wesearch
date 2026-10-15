"""Package-level pytest fixtures for wesearch.

Exists to bind the autouse XDG isolation fixture at the package root. wesearch
resolves its cache (fetched PDFs), state (zendriver profile), and config
through ``wesearch.lib.userdirs``; an unisolated test writes into the developer's
-- and, after export, the installer's -- real directories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wesearch.fetch.transport.curl import close_curl_sessions_except
from wesearch.fetch.transport.zendriver import (
    close_launched_browsers,
    shutdown_browsers,
)
from wesearch.lib.testing.resources import (
    isolate_user_dirs,
    pytest_collection_modifyitems,
    pytest_configure,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


# Re-exported, not merely imported: an autouse fixture reaches only the
# directory of the conftest that names it, so binding it here is what widens it
# to the whole package.
__all__ = [
    "close_pooled_browsers",
    "close_pooled_curl_sessions",
    "isolate_user_dirs",
    "pytest_collection_modifyitems",
    "pytest_configure",
]


@pytest.fixture(scope="module", autouse=True)
def close_pooled_browsers() -> Iterator[None]:
    """Disconnect from, and reap, every Chrome a test module started.

    Production leaves the shared Chrome running for the next process; a test
    run must not, or each isolated profile leaves one behind. MODULE scope, not
    session: session scope reaps them only after every later test has run
    beside them, and 22 resident browsers starved a sibling suite of memory.
    """
    yield
    shutdown_browsers()
    close_launched_browsers()


@pytest.fixture(scope="module", autouse=True)
def close_pooled_curl_sessions() -> Iterator[None]:
    """Close every pooled curl session when its test module finishes.

    curl_cffi stores native handles in thread-local state. Leaving those
    handles alive after a module lets a later PGlite subprocess fork inherit
    them; Python 3.14 can then finalize another thread's handle in the child
    before exec and crash inside ``curl_easy_cleanup``.
    """
    yield
    close_curl_sessions_except(None)
