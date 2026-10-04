"""Tests for wesearch.chrome.headers."""

from __future__ import annotations

from typing import Literal

import base64
import hashlib

import pytest

from wesearch.chrome.headers import (
    chrome_client_hints,
    chrome_headers_for_google,
    chrome_navigation_headers,
    chrome_user_agent,
    impersonate_version_platform,
    is_google_property,
)


class TestChromeClientHints:
    def test_default_platform_is_macos(self) -> None:
        assert chrome_client_hints(major=146) == chrome_client_hints(
            major=146,
            platform="macOS",
        )

    def test_desktop_extended_hints(self) -> None:
        assert chrome_client_hints(major=146, platform="macOS") == {
            "sec-ch-ua-arch": '"x86"',
            "sec-ch-ua-platform-version": '""',
            "sec-ch-ua-model": '""',
            "sec-ch-ua-bitness": '"64"',
            "sec-ch-ua-wow64": "?0",
            "sec-ch-ua-full-version-list": (
                '"Chromium";v="146.0.0.0", "Not-A.Brand";v="24.0.0.0", '
                '"Google Chrome";v="146.0.0.0"'
            ),
        }

    def test_android_hints_are_mobile_shaped(self) -> None:
        # A phone reports empty arch/bitness and a device model, not x86/64.
        assert chrome_client_hints(major=131, platform="Android") == {
            "sec-ch-ua-arch": '""',
            "sec-ch-ua-platform-version": '""',
            "sec-ch-ua-model": '"K"',
            "sec-ch-ua-bitness": '""',
            "sec-ch-ua-wow64": "?0",
            "sec-ch-ua-full-version-list": (
                '"Chromium";v="131.0.0.0", "Not-A.Brand";v="24.0.0.0", '
                '"Google Chrome";v="131.0.0.0"'
            ),
        }

    def test_does_not_include_the_basic_hints(self) -> None:
        # sec-ch-ua / -mobile / -platform are curl_cffi's job (its impersonate
        # emits them); this helper adds ONLY the extended set curl omits.
        h = chrome_client_hints(major=146, platform="macOS")
        assert "sec-ch-ua" not in h
        assert "sec-ch-ua-mobile" not in h
        assert "sec-ch-ua-platform" not in h

    def test_full_version_flows_into_full_version_list(self) -> None:
        h = chrome_client_hints(
            major=146,
            platform="macOS",
            full_version="146.0.7379.0",
        )
        assert "146.0.7379.0" in h["sec-ch-ua-full-version-list"]

    def test_unsupported_platform_rejected(self) -> None:
        with pytest.raises(ValueError, match="platform"):
            chrome_client_hints(major=146, platform="BeOS")  # ty: ignore[invalid-argument-type] -- The test passes an invalid platform to verify validation.  # pyright: ignore[reportArgumentType] -- The test passes an invalid platform to verify validation.


class TestGoogleChromeHeaders:
    def test_default_platform_is_macos(self) -> None:
        assert chrome_headers_for_google(major=146) == chrome_headers_for_google(
            major=146,
            platform="macOS",
        )

    def test_has_google_only_headers_not_client_hints(self) -> None:
        # chrome_headers_for_google is the Google-ONLY set; the extended client
        # hints are fetch's Accept-CH job, so they must NOT appear here.
        h = chrome_headers_for_google(major=146, platform="macOS")
        assert list(h) == [
            "x-browser-channel",
            "x-browser-year",
            "x-browser-validation",
            "x-client-data",
        ]
        assert h["x-browser-channel"] == "stable"
        assert h["x-browser-year"] == "2026"
        assert h["x-client-data"] == "CInbygE="
        assert "sec-ch-ua-arch" not in h
        assert "sec-ch-ua-full-version-list" not in h

    def test_custom_client_data_is_forwarded_exactly(self) -> None:
        assert (
            chrome_headers_for_google(
                major=146,
                platform="Linux",
                x_client_data="custom-token",
            )["x-client-data"]
            == "custom-token"
        )

    def test_validation_is_base64_sha1_of_key_plus_ua(self) -> None:
        # x-browser-validation = base64(sha1(linux_api_key + linux UA)).
        h = chrome_headers_for_google(major=146, platform="Linux")
        ua = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
        )
        key = "AIzaSyBqJZh-7pA44blAaAkH6490hUFOwX0KCYM"
        want = base64.b64encode(hashlib.sha1((key + ua).encode()).digest()).decode()  # noqa: S324 -- The fixture uses the protocol's legacy digest representation.
        assert h["x-browser-validation"] == want

    def test_validation_differs_by_platform(self) -> None:
        mac = chrome_headers_for_google(major=146, platform="macOS")
        lin = chrome_headers_for_google(major=146, platform="Linux")
        assert mac["x-browser-validation"] != lin["x-browser-validation"]

    def test_android_validation_uses_mobile_ua(self) -> None:
        # Android's UA ends "Mobile Safari"; its validation token must reflect
        # that (differs from the desktop token even at the same major).
        droid = chrome_headers_for_google(major=131, platform="Android")
        desk = chrome_headers_for_google(major=131, platform="Linux")
        assert droid["x-browser-validation"] != desk["x-browser-validation"]


class TestImpersonateVersionPlatform:
    @pytest.mark.parametrize(
        ("platform", "tail"),
        [
            ("Windows", "Safari/537.36"),
            ("Linux", "Safari/537.36"),
            ("macOS", "Safari/537.36"),
            ("Android", "Mobile Safari/537.36"),
        ],
    )
    def test_user_agent_is_exact_for_each_platform(
        self,
        platform: Literal["Windows", "Linux", "macOS", "Android"],
        tail: str,
    ) -> None:
        assert chrome_user_agent(146, platform) == (
            f"Mozilla/5.0 ({ {'Windows': 'Windows NT 10.0; Win64; x64', 'Linux': 'X11; Linux x86_64', 'macOS': 'Macintosh; Intel Mac OS X 10_15_7', 'Android': 'Linux; Android 10; K'}[platform] }) AppleWebKit/537.36 "
            f"(KHTML, like Gecko) Chrome/146.0.0.0 {tail}"
        )

    def test_user_agent_rejects_platform_with_exact_message(self) -> None:
        with pytest.raises(ValueError, match=r"^Unsupported platform 'BeOS'\.$"):
            chrome_user_agent(146, "BeOS")

    def test_chrome_desktop(self) -> None:
        assert impersonate_version_platform("chrome") == (146, "macOS")

    def test_chrome_android(self) -> None:
        assert impersonate_version_platform("chrome_android") == (131, "Android")

    def test_unknown_falls_back_to_desktop(self) -> None:
        assert impersonate_version_platform("safari") == (146, "macOS")

    def test_navigation_headers_default_platform_is_macos(self) -> None:
        assert chrome_navigation_headers(major=146) == chrome_navigation_headers(
            major=146,
            platform="macOS",
        )

    def test_navigation_headers_are_exact_for_get(self) -> None:
        headers = chrome_navigation_headers(
            major=146,
            platform="macOS",
            http2=True,
        )
        assert list(headers) == [
            "sec-ch-ua",
            "sec-ch-ua-mobile",
            "sec-ch-ua-platform",
            "Upgrade-Insecure-Requests",
            "User-Agent",
            "Accept",
            "Sec-Fetch-Site",
            "Sec-Fetch-Mode",
            "Sec-Fetch-User",
            "Sec-Fetch-Dest",
            "Accept-Encoding",
            "Accept-Language",
            "Priority",
        ]
        assert headers == {
            "sec-ch-ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": chrome_user_agent(146),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            ),
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Priority": "u=0, i",
        }

    def test_navigation_headers_are_exact_for_head(self) -> None:
        headers = chrome_navigation_headers(
            major=131,
            platform="Android",
            method="HEAD",
            http2=False,
        )
        assert headers["User-Agent"] == chrome_user_agent(131, "Android")
        assert headers["Upgrade-Insecure-Requests"] == "1"
        assert headers["Sec-Fetch-Mode"] == "navigate"
        assert "Content-Type" not in headers
        assert "Origin" not in headers
        assert "Priority" not in headers

    def test_navigation_headers_non_navigation_defaults_are_empty(self) -> None:
        headers = chrome_navigation_headers(major=146, platform="macOS", method="PATCH")
        assert "Content-Type" not in headers
        assert "Origin" not in headers
        assert "Priority" not in headers

    def test_navigation_headers_are_exact_for_post(self) -> None:
        headers = chrome_navigation_headers(
            major=131,
            platform="Android",
            method="POST",
            content_type="application/json",
            origin="https://example.test",
            http2=False,
        )
        assert list(headers) == [
            "sec-ch-ua",
            "sec-ch-ua-mobile",
            "sec-ch-ua-platform",
            "User-Agent",
            "Accept",
            "Content-Type",
            "Origin",
            "Sec-Fetch-Site",
            "Sec-Fetch-Mode",
            "Sec-Fetch-Dest",
            "Accept-Encoding",
            "Accept-Language",
        ]
        assert headers == {
            "sec-ch-ua": '"Chromium";v="131", "Not-A.Brand";v="24", "Google Chrome";v="131"',
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": '"Android"',
            "User-Agent": chrome_user_agent(131, "Android"),
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Origin": "https://example.test",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
        }
        assert "Priority" not in headers

    @pytest.mark.parametrize(
        "host",
        ["google.com", "SCHOLAR.GOOGLE.COM.", "a.gstatic.com", "x.googleapis.com"],
    )
    def test_google_property_domains(self, host: str) -> None:
        assert is_google_property(host)

    @pytest.mark.parametrize(
        "host",
        ["google.com.evil", "evilgoogle.com", "example.com", "google.comX"],
    )
    def test_non_google_domains(self, host: str) -> None:
        assert not is_google_property(host)

    def test_trailing_non_dot_character_is_not_stripped(self) -> None:
        assert is_google_property("google.comX") is False
        assert is_google_property("google.comX.") is False
        assert is_google_property("google.com..") is True


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
