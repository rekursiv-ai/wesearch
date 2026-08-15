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

import shutil
import subprocess
import tempfile


__all__ = [
    "chrome_available",
    "drive_chrome",
]


def chrome_available() -> bool:
    """Whether a Chrome binary is on ``PATH``."""
    return _chrome_binary() is not None


def drive_chrome(
    url: str,
    *,
    timeout_sec: float = 40.0,
    ignore_certificate_errors: bool = False,
) -> bool:
    """Load ``url`` in a headless Chrome, returning once it has exited or hung.

    Args:
      url: The URL to navigate to.
      timeout_sec: How long to wait for Chrome before killing it.
      ignore_certificate_errors: Accept an untrusted TLS certificate. Required
        only to reach a loopback oracle serving a self-signed cert; never enable
        against a real host.

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
        # Chrome's exit is not what a caller observes -- the request reached the
        # server before the page rendered. Two headless Chromes cold-starting at
        # once on a 2-core CI runner routinely blow the timeout AFTER
        # navigating, so raising here would fail a suite over a browser
        # shutdown nobody is testing.
        try:
            subprocess.run(  # noqa: S603 -- fixed argv, binary resolved from PATH.
                [
                    binary,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--incognito",
                    f"--user-data-dir={profile}",
                    *(
                        ["--ignore-certificate-errors"]
                        if ignore_certificate_errors
                        else []
                    ),
                    "--dump-dom",
                    url,
                ],
                capture_output=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
    return timed_out


def _chrome_binary() -> str | None:
    """The first available Chrome binary name, or ``None``."""
    for name in ("google-chrome-stable", "google-chrome", "chromium", "chrome"):
        if shutil.which(name) is not None:
            return name
    return None
