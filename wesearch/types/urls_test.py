"""Tests for ``wesearch.types.urls``."""

from __future__ import annotations

import base64

from wesearch.types.urls import without_userinfo


def test_no_userinfo() -> None:
    assert without_userinfo("https://example.com/p?q=1") == (
        "https://example.com/p?q=1",
        "",
    )


def test_user_pass_stripped_and_encoded() -> None:
    url, auth = without_userinfo("https://u:p@example.com:8443/x")
    assert url == "https://example.com:8443/x"
    assert auth == "Basic " + base64.b64encode(b"u:p").decode()


def test_pct_decoded_credentials() -> None:
    url, auth = without_userinfo("https://u%40x:p%3Aw@example.com/")
    assert url == "https://example.com/"
    assert auth == "Basic " + base64.b64encode(b"u@x:p:w").decode()


def test_user_only() -> None:
    url, auth = without_userinfo("https://u@example.com/")
    assert url == "https://example.com/"
    assert auth == "Basic " + base64.b64encode(b"u:").decode()


def test_password_only_and_at_sign_in_password() -> None:
    assert without_userinfo("https://:pass@example.com/path")[1] == "Basic OnBhc3M="
    assert without_userinfo("https://u:p@ss@example.com/path")[0] == (
        "https://example.com/path"
    )


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
