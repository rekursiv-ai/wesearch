"""Drive a real headless Chrome at a URL, for parity testing.

The request is recorded by the receiving server (e.g. the loopback echo
oracle in the parity tests), not here. Callers gate on
:func:`chrome_available` and skip when no usable Chrome is installed; a
snap-confined Chromium does not count.
"""

from __future__ import annotations

from pathlib import Path

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


# Here and not inside :func:`die_with_parent`, which runs between ``fork`` and ``exec``:
# ``CDLL`` is a ``dlopen``, and the loader lock it takes is never released in the child
# if another thread held it at the fork.
def _load_libc() -> ctypes.CDLL | None:
    """Resolve libc once, at import, or ``None`` off Linux."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        return ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return None


_libc = _load_libc()


def chrome_available() -> bool:
    """Whether a Chrome binary that honors ``--user-data-dir`` is on ``PATH``."""
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
        close.
      ignore_certificate_errors: Accept an untrusted TLS certificate, e.g. a
        loopback oracle's self-signed one. Never enable against a real host.
      disable_sandbox: Drop Chrome's process sandbox, which it requires when
        run as root. Never enable against a real host.

    Returns:
      timed_out: Whether Chrome had to be killed at ``timeout_sec``.

    Raises:
      RuntimeError: When no Chrome binary is available.

    """
    binary = _chrome_binary()
    if binary is None:
        raise RuntimeError("No Chrome binary found on PATH.")
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="chrome-capture-") as profile:
        # `Popen` + group kill, not `subprocess.run`: its timeout kills only the
        # direct child, orphaning Chrome's zygote and renderers.
        process = subprocess.Popen(  # noqa: S603 -- fixed argv, binary from PATH.
            [
                binary,
                "--headless=new",
                "--disable-gpu",
                "--incognito",
                f"--user-data-dir={profile}",
                # Otherwise Chrome asks D-Bus for its keyring, and a locked
                # keyring never answers, hanging every page load. Safe: the
                # profile is discarded per call.
                "--password-store=basic",
                # Field-trial features change Accept and Accept-Language per build.
                (
                    "--disable-features=JXLImageFormat,ReduceAcceptLanguage,"
                    "ReduceAcceptLanguageHTTP"
                ),
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
            # holding the pipes, so waiting for EOF could hang forever.
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.communicate(timeout=reap_timeout_sec)
    return timed_out


def die_with_parent() -> None:
    """Make the calling child its own group leader, killed when its parent dies.

    Use as ``preexec_fn``: it never raises and takes no lock.

    ``PR_SET_PDEATHSIG`` fires when the forking THREAD exits, not the process,
    so use it only where the launching thread outlives the child.
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
    """Return the first usable Chrome binary name, or ``None``."""
    for name in (
        "google-chrome-stable",
        "google-chrome",
        "chromium-browser",
        "chromium",
        "chrome",
    ):
        found = shutil.which(name)
        # A snap-confined Chromium ignores a ``--user-data-dir`` outside its
        # sandbox and falls back to its persistent profile, so runs are not
        # discarded and concurrent launches abort on its ``SingletonLock``.
        if found is None or _snap_confined(found):
            continue
        return name
    return None


# The byte scan catches Ubuntu's ``/usr/bin/chromium-browser`` shell wrapper,
# which execs the snap but passes a ``/snap/`` path check.
def _snap_confined(launcher: str) -> bool:
    """Whether ``launcher`` is the snap entry point or a wrapper that execs it."""
    resolved = Path(launcher).resolve()
    if resolved.name == "snap" or resolved.is_relative_to("/snap"):
        return True
    try:
        head = resolved.read_bytes()[:4096]
    except OSError:
        return False
    return b"snap run" in head or b"/snap/bin/" in head


# A missing group is the race of Chrome exiting as the timeout fires, not an error.
def _kill_group(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL the process group led by ``process``."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
