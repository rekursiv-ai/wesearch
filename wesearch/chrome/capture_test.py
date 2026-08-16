"""Unit tests for driving a headless Chrome in chrome.capture."""

from __future__ import annotations

from unittest.mock import patch

import shutil
import subprocess

import pytest

from wesearch.chrome.capture import chrome_available, drive_chrome


class TestDriveChrome:
    def test_timeout_is_reported_not_raised(self) -> None:
        # A Chrome that hangs AFTER navigating has already put the request on
        # the wire; the server-side record is complete. Propagating the timeout
        # failed the parity suite over a browser shutdown nobody is testing.
        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="chrome", timeout=40.0),
            ),
            patch(
                "wesearch.chrome.capture._chrome_binary",
                return_value="google-chrome-stable",
            ),
        ):
            assert drive_chrome("https://localhost:1/") is True

    def test_clean_exit_reports_no_timeout(self) -> None:
        with (
            patch("subprocess.run"),
            patch(
                "wesearch.chrome.capture._chrome_binary",
                return_value="google-chrome-stable",
            ),
        ):
            assert drive_chrome("https://localhost:1/") is False

    def test_missing_binary_raises(self) -> None:
        with (
            patch("wesearch.chrome.capture._chrome_binary", return_value=None),
            pytest.raises(RuntimeError, match="No Chrome binary"),
        ):
            drive_chrome("https://localhost:1/")

    def test_certificate_flag_only_when_requested(self) -> None:
        with (
            patch("subprocess.run") as run,
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
            patch("subprocess.run") as run,
            patch(
                "wesearch.chrome.capture._chrome_binary",
                return_value="google-chrome-stable",
            ),
        ):
            drive_chrome("https://example.com/")
            assert "--no-sandbox" not in run.call_args.args[0]
            drive_chrome("https://example.com/", disable_sandbox=True)
            assert "--no-sandbox" in run.call_args.args[0]


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
