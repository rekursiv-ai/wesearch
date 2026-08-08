"""Package-level pytest fixtures for wesearch.

Exists to bind the autouse XDG isolation fixture at the package root. wesearch
resolves its cache (fetched PDFs), state (zendriver profile), and config
through ``wesearch.lib.userdirs``; an unisolated test writes into the developer's
-- and, after export, the installer's -- real directories.
"""

from __future__ import annotations

from wesearch.lib.testing.userdirs_fixture import (
    isolate_user_dirs,
    pytest_configure,
)


# Re-exported, not merely imported: an autouse fixture reaches only the
# directory of the conftest that names it, so binding it here is what widens it
# to the whole package.
__all__ = ["isolate_user_dirs", "pytest_configure"]
