"""Drive a real headless Chrome at a URL, for parity testing.

A parity test needs a real Chrome to issue a real request so it can assert
:mod:`wesearch.fetch` sends the same thing. This module launches that
Chrome; the request itself is read off the server that received it
(:class:`wesearch.chrome.echo.EchoOracle` records ordered header names as
they arrive), so nothing here has to interpret Chrome's own bookkeeping.

Requires a ``google-chrome`` / ``google-chrome-stable`` binary; callers gate on
:func:`chrome_available` and skip when absent (Chrome is not a hard dependency).
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import signal
import subprocess
import sys
import tempfile


__all__ = [
    "chrome_available",
    "die_with_parent",
    "drive_chrome",
]


def _load_libc() -> ctypes.CDLL | None:
    """Resolve libc once, at import, or ``None`` off Linux.

    Here and not inside :func:`die_with_parent`, which runs between ``fork``
    and ``exec``: ``CDLL`` is a ``dlopen``, and the loader lock it takes is
    never released in the child if another thread held it at the fork.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        return ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return None


# config-globals: ignore -- a process-wide libc handle, not a tunable.
_libc = _load_libc()


def chrome_available() -> bool:
    """Whether a Chrome binary is on ``PATH``."""
    return _chrome_binary() is not None


def drive_chrome(
    url: str,
    *,
    timeout_sec: float = 40.0,
    reap_timeout_sec: float = 10.0,
    ignore_certificate_errors: bool = False,
    disable_sandbox: bool = False,
) -> bool:
    """Load ``url`` in a headless Chrome, returning once it has exited or hung.

    Args:
      url: The URL to navigate to.
      timeout_sec: How long to wait for Chrome before killing it.
      reap_timeout_sec: How long to wait for the killed browser's pipes to
        close. Bounded because a descendant that left the process group survives
        the kill still holding them; see the call site.
      ignore_certificate_errors: Accept an untrusted TLS certificate. Required
        only to reach a loopback oracle serving a self-signed cert; never enable
        against a real host.
      disable_sandbox: Drop Chrome's process sandbox. Required only where the
        harness runs as root (a CI container), which is the one environment
        where Chrome refuses to start otherwise; never enable against a real
        host, where the sandbox is the containment boundary for hostile pages.

    Returns:
      timed_out: Whether Chrome had to be killed at ``timeout_sec``. Reported
        rather than raised so a caller holding the server-side record can tell
        a hang that already navigated (harmless) from one that never did.

    Raises:
      RuntimeError: When no Chrome binary is available.

    """
    binary = _chrome_binary()
    if binary is None:
        raise RuntimeError("No Chrome binary found on PATH.")
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="chrome-capture-") as profile:
        # `Popen` + explicit group kill, not `subprocess.run`: its timeout path
        # kills only the direct child, and Chrome forks a zygote and a renderer
        # per tab, so the rest reparent to init at ~70 MB each.
        process = subprocess.Popen(  # noqa: S603 -- fixed argv, binary from PATH.
            [
                binary,
                "--headless=new",
                "--disable-gpu",
                "--incognito",
                f"--user-data-dir={profile}",
                # Without this, Chrome asks the D-Bus Secret Service for its
                # password-store key, and a registered-but-locked keyring --
                # any headless server someone has logged into -- never
                # answers. Every page load then hangs forever while
                # about:blank still renders. Safe because this profile is
                # incognito and discarded per call; the persistent one that
                # can hold a seated login belongs to the zendriver backend,
                # which launches its own Chrome.
                "--password-store=basic",
                *(["--no-sandbox"] if disable_sandbox else []),
                *(["--ignore-certificate-errors"] if ignore_certificate_errors else []),
                "--dump-dom",
                url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=die_with_parent,  # noqa: PLW1509 -- bare syscalls only; takes no lock a forked thread could hold.
        )
        try:
            process.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(process)
            # Bounded: a descendant that left the group survives the kill
            # holding the inherited pipes, and an unbounded wait for EOF hangs
            # here forever instead of reporting the timeout.
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.communicate(timeout=reap_timeout_sec)
    return timed_out


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL a process and every child it forked.

    :func:`die_with_parent` made it a group leader, so one ``killpg`` reaches
    the zygote and renderers. A missing group is the normal race between the
    timeout firing and Chrome finishing, not an error.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


def die_with_parent() -> None:
    """Make the calling child its own group leader, killed when its parent dies.

    Runs in the forked child before ``exec``, so it must never raise and must
    call nothing that takes a lock -- the ``PLW1509`` hazard. That holds only
    because ``_libc`` is already resolved; see :func:`_load_libc`.

    ``PR_SET_PDEATHSIG`` is THREAD-scoped: the kernel fires it when the forking
    thread exits, not the process. Correct here, where the caller blocks on the
    child, and WRONG for a pool that outlives its launching thread -- measured,
    a real Chrome died with ``rc=-9`` when only that thread ended.
    """
    with contextlib.suppress(OSError):
        os.setpgid(0, 0)
    if _libc is None:
        return
    with contextlib.suppress(OSError):
        # PR_SET_PDEATHSIG == 1. Spelled out because the constant lives in
        # `prctl`, a package we do not depend on.
        _libc.prctl(1, signal.SIGKILL, 0, 0, 0)


def _chrome_binary() -> str | None:
    """The first available Chrome binary name, or ``None``."""
    for name in (
        "google-chrome-stable",
        "google-chrome",
        "chromium-browser",
        "chromium",
        "chrome",
    ):
        if shutil.which(name) is not None:
            return name
    return None
