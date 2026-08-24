"""Tests for cross-site challenge detection."""

from __future__ import annotations

import pytest

from wesearch.fetch.challenge import classify_challenge, classify_http_error
from wesearch.types.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    FetchError,
    PuzzleChallengeError,
)


@pytest.mark.parametrize("marker", ["g-recaptcha", "h-captcha", "data-sitekey"])
def test_error_body_captcha_widget_is_puzzle(marker: str) -> None:
    assert classify_challenge(f'<div class="{marker}"></div>') is PuzzleChallengeError


def test_captcha_provider_named_in_js_config_is_content() -> None:
    # REV: a page that merely NAMES a captcha backend in its JS config (every
    # Wikipedia article carries this in mw.config) must not be misread as a
    # served challenge. The literal ``hcaptcha`` appears only inside a JSON
    # string, never as a rendered widget element.
    body = (
        "<html><head><title>Reciprocal rank fusion - Wikipedia</title></head>"
        '<body><script>RLCONF={"wgconfirmeditcaptchaneededforgenericedit":'
        '"hcaptcha","wgconfirmeditforceshowcaptcha":false};</script>'
        "<p>Reciprocal rank fusion is a method...</p></body></html>"
    )
    assert classify_challenge(body) is None


def test_captcha_marker_in_script_src_is_content() -> None:
    # A script reference to a captcha library (present on countless pages with a
    # dormant, non-blocking widget) is not a served challenge.
    body = (
        "<html><head><title>Contact Us</title>"
        '<script src="https://js.hcaptcha.com/1/api.js"></script></head>'
        "<body><h1>Contact form below</h1></body></html>"
    )
    assert classify_challenge(body) is None


def test_recaptcha_mentioned_in_prose_is_content() -> None:
    body = (
        "<html><head><title>How reCAPTCHA works</title></head>"
        "<body>This article explains recaptcha and hcaptcha internals.</body></html>"
    )
    assert classify_challenge(body) is None


def test_widget_marker_is_found_past_an_earlier_mention() -> None:
    # Only the FIRST occurrence in a tag was examined, so a marker appearing in
    # an unrelated attribute masked the real class that followed it.
    body = '<div data-provider="hcaptcha" class="hcaptcha"></div>'
    assert classify_challenge(body) is PuzzleChallengeError


def test_a_marker_on_the_script_tag_itself_is_structural() -> None:
    # ``trk_jschal_js`` is a default Cloudflare marker and lives on the SCRIPT
    # TAG, not in a div. Removing whole script ELEMENTS to hide their text also
    # removed the opening tag's attributes, so the browser settle loop -- which
    # calls this with ``on_success_body=True`` -- read the interstitial as the
    # finished page and returned the wall.
    body = (
        '<script id="trk_jschal_js" '
        'src="/cdn-cgi/challenge-platform/h/b/orchestrate/jsch/v1"></script>'
    )
    assert classify_challenge(body, on_success_body=True) is CloudflareChallengeError


def test_widget_markup_inside_a_script_is_content() -> None:
    # The tag scanner reads raw HTML, so a template literal or a string in JS
    # looks exactly like served markup. A script's contents are not the DOM.
    body = '<script>const t = `<div class="g-recaptcha"></div>`;</script>'
    assert classify_challenge(body) is None


def test_data_marker_in_a_url_is_content() -> None:
    # Any marker beginning ``data-`` was accepted anywhere in a tag, so a link
    # to documentation about the attribute read as a served widget.
    assert classify_challenge('<a href="/docs/data-sitekey">API docs</a>') is None
    assert classify_challenge('<div data-sitekey="x"></div>') is PuzzleChallengeError


def test_success_body_captcha_widget_is_content() -> None:
    body = '<form><div class="g-recaptcha" data-sitekey="x"></div></form>'
    assert classify_challenge(body, on_success_body=True) is None


def test_cloudflare_challenge_markup() -> None:
    body = '<script>window._cf_chl_opt={};</script><div class="challenge-platform">'
    assert classify_challenge(body) is CloudflareChallengeError


def test_turnstile_wins_over_generic_widget() -> None:
    body = '<div class="cf-turnstile" data-sitekey="0x4AAA"></div>'
    assert classify_challenge(body) is CloudflareChallengeError


@pytest.mark.parametrize(
    "markup",
    [
        '<div id="cf-challenge-running"></div>',
        '<div id="challenge-spinner"></div>',
        '<div id="cf-please-wait"></div>',
        '<div id="turnstile-wrapper"></div>',
        "<input name='cf-turnstile-response' value='x'>",
    ],
)
def test_cloudflare_structural_markup_on_success(markup: str) -> None:
    assert classify_challenge(markup, on_success_body=True) is CloudflareChallengeError


def test_cloudflare_title_on_success() -> None:
    assert (
        classify_challenge("<title>Just a moment...</title>", on_success_body=True)
        is CloudflareChallengeError
    )


def test_title_starting_with_challenge_phrase_is_content() -> None:
    body = "<title>Access Denied: A History of Firewalls</title>"
    assert classify_challenge(body, on_success_body=True) is None


def test_challenge_words_in_prose_are_content() -> None:
    body = (
        "<html><head><title>Blog</title></head><body>"
        "Please wait just a moment while the turnstile loads.</body></html>"
    )
    assert classify_challenge(body, on_success_body=True) is None


def test_page_documenting_cloudflare_selectors_is_content() -> None:
    body = (
        "<html><head><title>FlareSolverr | DeepWiki</title></head><body>"
        "Detection checks '#cf-challenge-running' and '#turnstile-wrapper'."
        "</body></html>"
    )
    assert classify_challenge(body, on_success_body=True) is None


def test_cloudflare_ambient_beacon_only_counts_on_error() -> None:
    body = "<script src='/cdn-cgi/challenge-platform/scripts/jsd/main.js'></script>"
    assert classify_challenge(body, on_success_body=True) is None
    assert classify_challenge(body) is CloudflareChallengeError


def test_marker_groups_are_retunable() -> None:
    assert classify_challenge("<div class='widgetguard-v2'></div>") is None
    assert (
        classify_challenge(
            "<div class='widgetguard-v2'></div>",
            cloudflare=("widgetguard-v2",),
        )
        is CloudflareChallengeError
    )


def test_provider_markers_are_not_shared_challenges() -> None:
    for body in (
        "<form id='challenge-form'></form>",
        "<form id='gs_captcha_f'></form>",
        "https://www.google.com/sorry/index",
        "<meta content='0;url=/httpservice/retry/enablejs'>",
    ):
        assert classify_challenge(body) is None


def test_http_error_uses_body_challenge() -> None:
    error = classify_http_error(
        "https://x.com", 403, {}, b'<div class="g-recaptcha"></div>'
    )
    assert isinstance(error, PuzzleChallengeError)


def test_http_error_uses_cloudflare_mitigation_header() -> None:
    # ``cf-mitigated: challenge`` is Cloudflare's own signal, documented as
    # present on every challenge page type and carrying no other value.
    # https://developers.cloudflare.com/cloudflare-challenges/challenge-types/challenge-pages/detect-response/
    error = classify_http_error(
        "https://x.com",
        403,
        {"server": "cloudflare", "cf-ray": "a1", "cf-mitigated": "challenge"},
        b"Temporarily unavailable",
    )
    assert isinstance(error, CloudflareChallengeError)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (403, b'{"error":"invalid API token"}'),
        (403, b'{"message":"forbidden"}'),
        (503, b"Service Temporarily Unavailable"),
        (503, b'{"error":"upstream at capacity"}'),
    ],
)
def test_cloudflare_fronted_origin_error_is_not_a_challenge(
    status: int, body: bytes
) -> None:
    """A response Cloudflare PROXIES is the origin's, not Cloudflare's.

    Every Cloudflare-fronted response carries ``server: cloudflare`` and a
    ``cf-ray``, so those headers prove only that Cloudflare is in front -- never
    that it authored the body. Treating them as proof made an expired API token
    a bot wall, which sends the caller to the browser and PERMANENTLY pins the
    domain to it (``fetch.py`` remembers the domain, and the routing list has no
    expiry). That is the wedge the 1015 rate-limit fix already removed once.
    """
    error = classify_http_error(
        "https://api.example/v1/x",
        status,
        {"server": "cloudflare", "cf-ray": "a1b2c3d4e5f6-SJC"},
        body,
    )
    assert type(error) is FetchError
    assert not isinstance(error, BotDetectionError)


def test_http_error_preserves_plain_failure() -> None:
    error = classify_http_error("https://x.com", 404, {"server": "nginx"}, b"Not found")
    assert type(error) is FetchError
    assert not isinstance(error, BotDetectionError)


def test_cloudflare_rate_limit_is_not_a_challenge() -> None:
    # Verbatim from a live rate-limited SearXNG instance: Cloudflare error 1015
    # is a RATE LIMIT the site owner configured, not an automated-access
    # challenge. Classifying it as one made callers "recover" by re-fetching
    # through a browser, which renders a JSON API into HTML and breaks parsing.
    error = classify_http_error(
        "https://search.example/search?format=json",
        429,
        {
            "server": "cloudflare",
            "cf-ray": "a2e5f1ad0fe14aad-SJC",
            "retry-after": "10",
            "content-type": "text/plain; charset=UTF-8",
        },
        b"error code: 1015",
    )
    assert type(error) is FetchError
    assert not isinstance(error, BotDetectionError)


def test_cloudflare_challenge_at_429_is_still_a_challenge() -> None:
    # Dropping 429 from the mitigation statuses must not blind the body check:
    # an interstitial served WITH a 429 is proven by its own markup.
    error = classify_http_error(
        "https://x.com",
        429,
        {"server": "cloudflare"},
        b"<html><head><title>Just a moment...</title></head></html>",
    )
    assert isinstance(error, CloudflareChallengeError)


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
