"""Tests for wesearch.fetch.session."""

from __future__ import annotations

from unittest.mock import patch

import atexit
import os
import threading

import pytest

from wesearch.fetch import session
from wesearch.fetch.session import default_session
from wesearch.fetch.types import FetchSession


def test_the_default_session_is_one_per_process() -> None:
    first = default_session()
    assert default_session() is first


def test_concurrent_first_calls_make_one_session() -> None:
    seen: list[FetchSession] = []
    start = threading.Barrier(8)

    def grab() -> None:
        start.wait()
        seen.append(default_session())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len({id(s) for s in seen}) == 1


def test_exit_closes_the_default_session() -> None:
    made = default_session()
    with patch.object(FetchSession, "close", autospec=True) as close:
        session.close_default_session()
    close.assert_called_once_with(made)
    assert default_session() is not made


def test_process_hooks_register_once(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list[object] = []
    monkeypatch.setattr(session, "_hooked", [])
    monkeypatch.setattr(atexit, "register", registered.append)
    default_session()
    session.close_default_session()
    default_session()
    assert registered == [session.close_default_session]


def test_forgetting_in_a_child_drops_the_session_and_the_lock() -> None:
    parent = default_session()
    held = session._lock
    held.acquire()
    try:
        session._forget_in_child()
        assert session._lock is not held
        assert default_session() is not parent
    finally:
        held.release()


def test_closing_with_no_default_does_nothing() -> None:
    session.close_default_session()
    with patch.object(FetchSession, "close", autospec=True) as close:
        session.close_default_session()
    close.assert_not_called()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_forked_child_never_reuses_the_parents_session() -> None:
    parent = default_session()
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover -- runs in the child, outside the tracer.
        os.close(read)
        os.write(write, b"fresh" if default_session() is not parent else b"same")
        os._exit(0)
    os.close(write)
    try:
        assert os.read(read, 5) == b"fresh"
    finally:
        os.close(read)
        os.waitpid(pid, 0)
    assert default_session() is parent


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
