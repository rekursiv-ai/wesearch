"""Unit tests for driving a headless Chrome in chrome.capture."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast
from unittest.mock import MagicMock, patch

import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from wesearch.chrome import capture
from wesearch.chrome.capture import (
    _chrome_binary,
    _kill_group,
    _load_libc,
    chrome_available,
    die_with_parent,
    drive_chrome,
)


_CWD: Final = Path(__file__).resolve().parent


# The second call must succeed: the timeout path reaps after killing, and a stub that
# raises forever would hide a missing reap behind an exception the test itself supplied.
def _popen_mock(*, timeout: bool = False) -> MagicMock:
    """Return a ``Popen`` stub whose ``communicate`` times out on the first call."""
    process = MagicMock()
    process.pid = 4321
    process.communicate.side_effect = (
        [subprocess.TimeoutExpired(cmd="chrome", timeout=40.0), (b"", b"")]
        if timeout
        else [(b"", b"")]
    )
    return process


def _fresh_popen_mock(*args: object, **kwargs: object) -> MagicMock:
    """Return a new stub per ``Popen`` call, for tests that drive Chrome twice."""
    del args, kwargs
    return _popen_mock()


class TestKillGroup:
    def test_ignores_missing_or_inaccessible_process_groups(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", ""], stdout=subprocess.PIPE)
        try:
            for error in (ProcessLookupError(), PermissionError()):
                with patch("os.killpg", side_effect=error) as killpg:
                    _kill_group(process)
                killpg.assert_called_once_with(process.pid, signal.SIGKILL)
        finally:
            process.wait()


class TestDieWithParent:
    def test_arms_process_group_and_parent_death_signal(self) -> None:
        libc = MagicMock()
        with (
            patch.object(capture, "_libc", libc),
            patch.object(os, "setpgid") as setpgid,
        ):
            die_with_parent()
        setpgid.assert_called_once_with(0, 0)
        libc.prctl.assert_called_once_with(1, signal.SIGKILL, 0, 0, 0)

    def test_ignores_process_group_and_libc_errors(self) -> None:
        libc = MagicMock()
        with (
            patch.object(capture, "_libc", libc),
            patch.object(os, "setpgid", side_effect=OSError),
        ):
            libc.prctl.side_effect = OSError
            die_with_parent()
        libc.prctl.assert_called_once_with(1, signal.SIGKILL, 0, 0, 0)

    def test_skips_parent_death_signal_when_libc_is_unavailable(self) -> None:
        with patch.object(capture, "_libc", None), patch.object(os, "setpgid"):
            die_with_parent()


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
            pytest.raises(RuntimeError, match=r"^No Chrome binary found on PATH\.$"),
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
            argv = cast(list[str], run.call_args.args[0])
            assert "--ignore-certificate-errors" not in argv
            drive_chrome("https://localhost:1/", ignore_certificate_errors=True)
            argv = cast(list[str], run.call_args.args[0])
            assert "--ignore-certificate-errors" in argv

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
            argv = cast(list[str], run.call_args.args[0])
            assert "--no-sandbox" not in argv
            drive_chrome("https://example.com/", disable_sandbox=True)
            argv = cast(list[str], run.call_args.args[0])
            assert "--no-sandbox" in argv

    def test_default_timeouts_streams_and_profile_prefix_are_exact(self) -> None:
        process = _popen_mock()
        with (
            patch("subprocess.Popen", return_value=process) as run,
            patch("wesearch.chrome.capture._chrome_binary", return_value="chrome"),
        ):
            assert drive_chrome("https://example.test/") is False
        run.assert_called_once()
        assert run.call_args.kwargs["stdout"] is subprocess.PIPE
        assert run.call_args.kwargs["stderr"] is subprocess.PIPE
        argv = cast(list[str], run.call_args.args[0])
        assert argv[4].startswith("--user-data-dir=")
        assert Path(argv[4].split("=", 1)[1]).name.startswith("chrome-capture-")
        process.communicate.assert_called_once_with(timeout=40.0)

    def test_default_reap_timeout_and_suppression_are_exact(self) -> None:
        process = MagicMock()
        process.pid = 4321
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="chrome", timeout=40.0),
            subprocess.TimeoutExpired(cmd="chrome", timeout=10.0),
        ]
        with (
            patch("subprocess.Popen", return_value=process),
            patch("os.killpg"),
            patch("wesearch.chrome.capture._chrome_binary", return_value="chrome"),
        ):
            assert drive_chrome("https://example.test/") is True
        assert process.communicate.call_args_list[1].kwargs == {"timeout": 10.0}

    def test_all_chrome_flags_and_timeouts_are_forwarded(self) -> None:
        process = _popen_mock(timeout=True)
        with (
            patch("subprocess.Popen", return_value=process) as run,
            patch("os.killpg"),
            patch(
                "wesearch.chrome.capture._chrome_binary",
                return_value="chrome",
            ),
        ):
            assert drive_chrome(
                "https://example.test/",
                timeout_sec=2,
                reap_timeout_sec=3,
                ignore_certificate_errors=True,
                disable_sandbox=True,
            )
        argv = cast(list[str], run.call_args.args[0])
        assert argv == [
            "chrome",
            "--headless=new",
            "--disable-gpu",
            "--incognito",
            argv[4],
            "--password-store=basic",
            "--no-sandbox",
            "--ignore-certificate-errors",
            "--dump-dom",
            "https://example.test/",
        ]
        assert process.communicate.call_args_list[0].kwargs == {"timeout": 2}
        assert process.communicate.call_args_list[1].kwargs == {"timeout": 3}


@pytest.mark.cli_git
def test_a_killed_group_takes_the_forked_grandchild_with_it() -> None:
    """One ``killpg`` reaps a process AND whatever it forked.

    The mocked tests above assert the call is made; this asserts the call does
    what it is relied on to do. A real process forks a real child, then the
    group is killed through the same helper the timeout path uses.
    """
    process = _spawn_probe()
    try:
        assert process.stdout is not None
        stdout = process.stdout
        grandchild = int(stdout.readline())
        _kill_group(process)

        assert _died_within(grandchild, seconds=10.0), (
            f"grandchild {grandchild} survived the group kill"
        )
    finally:
        _reap(process)


@pytest.mark.cli_git
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
        stdout = parent.stdout
        child_pid = int(stdout.readline())
        os.kill(parent.pid, signal.SIGKILL)

        assert _died_within(child_pid, seconds=10.0), (
            f"child {child_pid} survived SIGKILL of its parent"
        )
    finally:
        _reap(parent)


def _spawn_probe(*args: str) -> subprocess.Popen[bytes]:
    """Start :mod:`orphan_probe` orphan-proofed; its stdout carries a child PID."""
    return subprocess.Popen(  # noqa: S603 -- fixed argv, interpreter from sys.
        [sys.executable, str(_CWD / "orphan_probe.py"), *args],
        stdout=subprocess.PIPE,
        preexec_fn=die_with_parent,  # noqa: PLW1509 -- bare syscalls only; takes no lock a forked thread could hold.
    )


def _reap(process: subprocess.Popen[bytes]) -> None:
    """Kill ``process`` and close the pipe this test opened on it."""
    process.kill()
    process.wait()
    if process.stdout is not None:
        process.stdout.close()


# Polled: a kill is asynchronous, so reading once right after signalling reports the
# pre-kill state and would pass an implementation that kills nothing only by luck.
def _died_within(pid: int, *, seconds: float) -> bool:
    """Whether ``pid`` stops being a live process within ``seconds``."""
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
    @pytest.mark.parametrize(
        "name",
        [
            "google-chrome-stable",
            "google-chrome",
            "chromium-browser",
            "chromium",
            "chrome",
        ],
    )
    def test_binary_search_order_and_names(self, name: str) -> None:
        def only(name_to_find: str) -> str | None:
            return "/bin/native" if name_to_find == name else None

        with patch.object(shutil, "which", only):
            assert _chrome_binary() == name

    def test_no_binary_returns_none(self) -> None:
        with patch.object(shutil, "which", return_value=None):
            assert _chrome_binary() is None

    def test_load_libc_returns_none_off_linux(self) -> None:
        with patch.object(sys, "platform", "darwin"):
            assert _load_libc() is None

    def test_load_libc_handles_missing_library(self) -> None:
        with (
            patch.object(sys, "platform", "linux"),
            patch("ctypes.CDLL", side_effect=OSError),
        ):
            assert _load_libc() is None

    def test_load_libc_uses_exact_linux_loader_arguments(self) -> None:
        libc = object()
        with (
            patch.object(sys, "platform", "linux-gnu"),
            patch("ctypes.CDLL", return_value=libc) as cdll,
        ):
            assert _load_libc() is libc
        cdll.assert_called_once_with("libc.so.6", use_errno=True)

    def test_finds_the_debian_chromium_browser_binary(self) -> None:
        # chromium-browser is the binary name Debian/Ubuntu install, so a host
        # carrying only that one skipped the whole parity suite as "no Chrome".
        def only_chromium_browser(name: str) -> str | None:
            return "/usr/bin/chromium-browser" if name == "chromium-browser" else None

        with patch.object(shutil, "which", only_chromium_browser):
            assert chrome_available()

    def test_a_snap_confined_chromium_does_not_count(self, tmp_path: Path) -> None:
        # Ubuntu's ``chromium`` package is a snap wrapper, which cannot use the
        # temp profile ``drive_chrome`` hands it; see ``_chrome_binary``.
        snap = tmp_path / "snap"
        snap.write_bytes(b"\x7fELF")
        entry = tmp_path / "chromium"
        entry.symlink_to(snap)

        def only_snap_chromium(name: str) -> str | None:
            return str(entry) if name == "chromium" else None

        with patch.object(shutil, "which", only_snap_chromium):
            assert not chrome_available()

    def test_a_usr_bin_wrapper_around_the_snap_does_not_count(
        self,
        tmp_path: Path,
    ) -> None:
        # The transitional deb puts a shell script under /usr/bin, so the
        # launcher's own path says nothing; its body execs the snap.
        wrapper = tmp_path / "chromium-browser"
        wrapper.write_text('#!/bin/sh\nexec /snap/bin/chromium "$@"\n')

        def only_wrapper(name: str) -> str | None:
            return str(wrapper) if name == "chromium-browser" else None

        with patch.object(shutil, "which", only_wrapper):
            assert not chrome_available()

    def test_snap_wrapper_marker_at_read_limit_is_detected(
        self,
        tmp_path: Path,
    ) -> None:
        wrapper = tmp_path / "chromium"
        wrapper.write_bytes(b"x" * (4_096 - len(b"/snap/bin/") + 1) + b"/snap/bin/")
        assert not capture._snap_confined(str(wrapper))

    def test_snap_run_marker_is_case_sensitive(self, tmp_path: Path) -> None:
        wrapper = tmp_path / "chromium"
        wrapper.write_bytes(b"x" * 100 + b"snap run")
        assert capture._snap_confined(str(wrapper))
        wrapper.write_bytes(b"x" * 100 + b"SNAP RUN")
        assert not capture._snap_confined(str(wrapper))

    def test_snap_path_check_uses_exact_prefix(self, tmp_path: Path) -> None:
        resolved = MagicMock()
        resolved.name = "chromium"
        resolved.is_relative_to.return_value = False
        resolved.read_bytes.return_value = b"native"
        with patch.object(Path, "resolve", return_value=resolved):
            assert not capture._snap_confined(str(tmp_path / "chromium"))
        resolved.is_relative_to.assert_called_once_with("/snap")

    def test_a_native_binary_counts(self, tmp_path: Path) -> None:
        # Positive control for the byte scan: an ordinary executable is kept.
        binary = tmp_path / "google-chrome"
        binary.write_bytes(b"\x7fELF" + bytes(64))

        def only_native(name: str) -> str | None:
            return str(binary) if name == "google-chrome" else None

        with patch.object(shutil, "which", only_native):
            assert chrome_available()


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
