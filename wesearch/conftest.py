"""Package-level pytest fixtures for wesearch.

Exists to bind the autouse XDG isolation fixture at the package root. wesearch
resolves its cache (fetched PDFs), state (zendriver profile), and config
through ``wesearch.lib.userdirs``; an unisolated test writes into the developer's
-- and, after export, the installer's -- real directories.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from wesearch.fetch.transport.zendriver import shutdown_browsers
from wesearch.lib.testing.userdirs_fixture import (
    isolate_user_dirs,
    pytest_configure,
)


# Re-exported, not merely imported: an autouse fixture reaches only the
# directory of the conftest that names it, so binding it here is what widens it
# to the whole package.
__all__ = ["close_pooled_browsers", "isolate_user_dirs", "pytest_configure"]


@pytest.fixture(scope="module", autouse=True)
def close_pooled_browsers() -> Iterator[None]:
    """Close every pooled Chrome when the module that opened one finishes.

    The pool keeps browsers warm on purpose, so nothing else closes one.
    MODULE scope, not session: session scope reaps them only after every later
    test has run beside them, and 22 resident browsers starved a sibling suite
    of memory. ``zendriver._pool`` covers process exit via ``atexit``.
    """
    yield
    shutdown_browsers()
