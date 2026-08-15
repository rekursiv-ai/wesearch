"""Unit tests for driving a headless Chrome in chrome.capture."""

from __future__ import annotations

from unittest.mock import patch

import subprocess

import pytest

from wesearch.chrome.capture import drive_chrome


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


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
