"""Unit tests for driving a headless Chrome in chrome.capture."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from wesearch.chrome.capture import (
    _kill_group,
    chrome_available,
    die_with_parent,
    drive_chrome,
)


def _popen_mock(*, timeout: bool = False) -> MagicMock:
    """A ``Popen`` stub whose ``communicate`` times out on the first call.

    The second call must succeed: the timeout path reaps after killing, and a
    stub that raises forever would hide a missing reap behind an exception the
    test itself supplied.
    """
    process = MagicMock()
    process.pid = 4321
    process.communicate.side_effect = (
        [subprocess.TimeoutExpired(cmd="chrome", timeout=40.0), (b"", b"")]
        if timeout
        else [(b"", b"")]
    )
    return process


def _fresh_popen_mock(*args: object, **kwargs: object) -> MagicMock:
    """A new stub per ``Popen`` call, for tests that drive Chrome twice."""
    del args, kwargs
    return _popen_mock()


class TestDriveChrome:
    def test_timeout_is_reported_not_raised(self) -> None:
        # A Chrome that hangs AFTER navigating has already put the request on
        # the wire; the server-side record is complete. Propagating the timeout
        # failed the parity suite over a browser shutdown nobody is testing.
        with (
            patch("subprocess.Popen", return_value=_popen_mock(timeout=True)),
            patch("os.killpg"),
            patch(
                "wesearch.chrome.capture._chrome_binary",
                return_value="google-chrome-stable",
            ),
        ):
            assert drive_chrome("https://localhost:1/") is True

    def test_a_timed_out_chrome_is_killed_as_a_group(self) -> None:
        """The whole process group dies, not just the browser's direct child.

        Chrome forks a zygote and one renderer per tab. Killing only the process
        we spawned reparents the rest to init, where they live on at ~70 MB
        each.
        """
        process = _popen_mock(timeout=True)
        with (
            patch("subprocess.Popen", return_value=process) as popen,
            patch("os.killpg") as killpg,
            patch(
                "wesearch.chrome.capture._chrome_binary",
                return_value="google-chrome-stable",
            ),
        ):
            drive_chrome("https://localhost:1/")

        # Its own group leader, which is what lets one killpg reach every
        # process it forked.
        assert popen.call_args.kwargs["preexec_fn"] is die_with_parent
        killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        # Reaped after the kill: an unwaited child stays a zombie holding the
        # pipes this function opened.
        assert process.communicate.call_count == 2

    def test_a_kill_racing_chromes_own_exit_is_not_an_error(self) -> None:
        """Chrome exiting between the timeout and the kill is normal, not a fault.

        ``killpg`` raises ``ProcessLookupError`` for a group that is already
        gone. Letting it propagate would convert the benign race into a test
        failure in the parity suite.
        """
        with (
            patch("subprocess.Popen", return_value=_popen_mock(timeout=True)),
            patch("os.killpg", side_effect=ProcessLookupError),
            patch(
                "wesearch.chrome.capture._chrome_binary",
                return_value="google-chrome-stable",
            ),
        ):
            assert drive_chrome("https://localhost:1/") is True

    def test_clean_exit_reports_no_timeout(self) -> None:
        with (
            patch("subprocess.Popen", return_value=_popen_mock()),
            patch("os.killpg") as killpg,
            patch(
                "wesearch.chrome.capture._chrome_binary",
                return_value="google-chrome-stable",
            ),
        ):
            assert drive_chrome("https://localhost:1/") is False
        killpg.assert_not_called()

    def test_missing_binary_raises(self) -> None:
        with (
            patch("wesearch.chrome.capture._chrome_binary", return_value=None),
            pytest.raises(RuntimeError, match="No Chrome binary"),
        ):
            drive_chrome("https://localhost:1/")

    def test_certificate_flag_only_when_requested(self) -> None:
        with (
            patch("subprocess.Popen", side_effect=_fresh_popen_mock) as run,
            patch(
                "wesearch.chrome.capture._chrome_binary",
                return_value="google-chrome-stable",
            ),
        ):
            drive_chrome("https://localhost:1/")
            assert "--ignore-certificate-errors" not in run.call_args.args[0]
            drive_chrome("https://localhost:1/", ignore_certificate_errors=True)
            assert "--ignore-certificate-errors" in run.call_args.args[0]

    def test_sandbox_flag_only_when_requested(self) -> None:
        # --no-sandbox drops Chrome's containment boundary. It is needed only
        # where the harness runs as root (CI); a caller pointing this at a real
        # URL must not silently get an unsandboxed browser.
        with (
            patch("subprocess.Popen", side_effect=_fresh_popen_mock) as run,
            patch(
                "wesearch.chrome.capture._chrome_binary",
                return_value="google-chrome-stable",
            ),
        ):
            drive_chrome("https://example.com/")
            assert "--no-sandbox" not in run.call_args.args[0]
            drive_chrome("https://example.com/", disable_sandbox=True)
            assert "--no-sandbox" in run.call_args.args[0]


@pytest.mark.slow
def test_a_killed_group_takes_the_forked_grandchild_with_it() -> None:
    """One ``killpg`` reaps a process AND whatever it forked.

    The mocked tests above assert the call is made; this asserts the call does
    what it is relied on to do. A real process forks a real child, then the
    group is killed through the same helper the timeout path uses.
    """
    process = _spawn_probe()
    try:
        assert process.stdout is not None
        grandchild = int(process.stdout.readline())
        _kill_group(process)

        assert _died_within(grandchild, seconds=10.0), (
            f"grandchild {grandchild} survived the group kill"
        )
    finally:
        _reap(process)


@pytest.mark.slow
def test_a_child_dies_when_its_parent_is_sigkilled() -> None:
    """A SIGKILLed parent still takes its browser with it.

    The backstop for the case no cleanup code reaches: ``atexit`` covers an
    ordinary exit, which is what actually leaked, and SIGKILL runs nothing.

    ``--proofed`` because the signal is armed on the FORKING THREAD. The probe's
    unproofed arm forks and arms inside the child, whose thread then ends; only
    the arm that reaches ``exec`` through ``die_with_parent`` -- how every
    browser starts -- keeps it armed against a thread that outlives it.
    """
    parent = _spawn_probe("--proofed")
    try:
        assert parent.stdout is not None
        child_pid = int(parent.stdout.readline())
        os.kill(parent.pid, signal.SIGKILL)

        assert _died_within(child_pid, seconds=10.0), (
            f"child {child_pid} survived SIGKILL of its parent"
        )
    finally:
        _reap(parent)


def _spawn_probe(*args: str) -> subprocess.Popen[bytes]:
    """Start :mod:`orphan_probe` orphan-proofed; its stdout carries a child PID."""
    return subprocess.Popen(  # noqa: S603 -- fixed argv, interpreter from sys.
        [sys.executable, str(Path(__file__).with_name("orphan_probe.py")), *args],
        stdout=subprocess.PIPE,
        preexec_fn=die_with_parent,  # noqa: PLW1509 -- bare syscalls only; takes no lock a forked thread could hold.
    )


def _reap(process: subprocess.Popen[bytes]) -> None:
    """Kill ``process`` and close the pipe this test opened on it."""
    process.kill()
    process.wait()
    if process.stdout is not None:
        process.stdout.close()


def _died_within(pid: int, *, seconds: float) -> bool:
    """Whether ``pid`` stops being a live process within ``seconds``.

    Polled: a kill is asynchronous, so reading once right after signalling
    reports the pre-kill state and would pass an implementation that kills
    nothing only by luck.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _alive(pid: int) -> bool:
    """Whether ``pid`` names a live, unreaped process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie answers signal 0, so signalling alone cannot distinguish one from
    # a live process; only a non-Z state counts as leaked.
    try:
        status = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return status.rsplit(")", 1)[-1].split()[0] != "Z"


class TestChromeAvailable:
    def test_finds_the_debian_chromium_browser_binary(self) -> None:
        # chromium-browser is the binary name Debian/Ubuntu install, so a host
        # carrying only that one skipped the whole parity suite as "no Chrome".
        def only_chromium_browser(name: str) -> str | None:
            return "/usr/bin/chromium-browser" if name == "chromium-browser" else None

        with patch.object(shutil, "which", only_chromium_browser):
            assert chrome_available()


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
