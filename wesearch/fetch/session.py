"""The process's shared :class:`FetchSession`, for callers that pass none."""

from __future__ import annotations

import atexit
import os
import threading

from wesearch.fetch.types import FetchSession


__all__ = ["close_default_session", "default_session"]

# Process-wide by design: a fresh session per call re-probes the egress and
# re-attaches to Chrome, so ``session=None`` shares this one, as urllib3's
# default pool and boto3's default session do.
_default: list[FetchSession] = []
_hooked: list[bool] = []
_lock = threading.Lock()


def default_session() -> FetchSession:
    """Return the process's shared session, made on first use.

    Returns:
      session: The same session on every call in this process; closed at exit.

    """
    with _lock:
        if not _hooked:
            # Once, at first use rather than import: importing must not
            # register process hooks.
            _hooked.append(True)
            atexit.register(close_default_session)
            if hasattr(os, "register_at_fork"):
                os.register_at_fork(after_in_child=_forget_in_child)
        if not _default:
            _default.append(FetchSession())
        return _default[0]


def close_default_session() -> None:
    """Close the shared session; the next :func:`default_session` makes a new one."""
    with _lock:
        sessions = list(_default)
        _default.clear()
    for session in sessions:
        session.close()


def _forget_in_child() -> None:
    """Drop the parent's session in a forked child, whose sockets and Chrome it shares."""
    global _lock  # noqa: PLW0603 -- the parent's lock may be held across the fork.
    _lock = threading.Lock()
    _default.clear()
