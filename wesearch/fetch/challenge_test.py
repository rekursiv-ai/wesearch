"""Tests for cross-site challenge detection."""

from __future__ import annotations

import pytest

from wesearch.fetch.challenge import (
    _is_widget_occurrence,
    _occurrences,
    _tags,
    _text,
    classify_challenge,
    classify_http_error,
    is_shell,
    is_unmarked_block,
    redirects_soon,
    visible_chars,
)
from wesearch.types.errors import (
    AwsWafChallengeError,
    BotDetectionError,
    CloudflareChallengeError,
    DataDomeChallengeError,
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
    ],
)
def test_cloudflare_structural_markup_on_success(markup: str) -> None:
    assert classify_challenge(markup, on_success_body=True) is CloudflareChallengeError


@pytest.mark.parametrize(
    "markup",
    [
        '<div class="cf-turnstile" data-sitekey="0x4aaaaaae1j7knkm6din4x9">',
        "<input name='cf-turnstile-response' value='x'>",
    ],
)
def test_success_body_turnstile_widget_is_content(markup: str) -> None:
    # Verbatim from a TechSpot article, which embeds Turnstile in its comment
    # form: reading the widget as a wall failed every fetch of the site, and
    # held the browser leg in its settle loop for half its budget.
    body = f"<html><head><title>Article | TechSpot</title></head><body>{markup}"
    assert classify_challenge(body, on_success_body=True) is None
    assert classify_challenge(body) is CloudflareChallengeError


_ARTICLE = "<p>" + "Alphabet reported quarterly revenue of $127.6bn. " * 60 + "</p>"


@pytest.mark.parametrize(
    "markup",
    [
        '<div id="cf-challenge-running"></div>',
        '<div class="cf-turnstile" data-sitekey="x"></div>',
        '<div class="g-recaptcha" data-sitekey="x"></div>',
        "<script src='/cdn-cgi/challenge-platform/scripts/jsd/main.js'></script>",
    ],
)
def test_markers_on_a_page_with_content_are_not_a_wall(markup: str) -> None:
    # Each marker list grew one false wall at a time; a page with an article is
    # its own proof that the block did not happen, whatever widget it embeds.
    body = f"<html><head><title>News</title></head><body>{_ARTICLE}{markup}"
    assert classify_challenge(body) is None
    assert classify_challenge(body, on_success_body=True) is None
    assert classify_challenge(f"<html><body>{markup}") is not None


def test_text_hidden_from_the_reader_does_not_make_a_wall_content() -> None:
    hidden = f"<script>{_ARTICLE}</script><style>{_ARTICLE}</style><!--{_ARTICLE}-->"
    body = f"<html><body>{hidden}<div id='cf-challenge-running'></div>"
    assert classify_challenge(body, on_success_body=True) is CloudflareChallengeError


def test_cloudflare_interstitial_prose_fits_the_wall_budget() -> None:
    # The visible text of Cloudflare's managed-challenge page, with its Ray ID.
    body = (
        "<html><head><title>Just a moment...</title></head><body>"
        "<h1>www.example.com</h1><h2>Verify you are human by completing the "
        "action below.</h2><div id='cf-challenge-running'></div><p>www.example.com "
        "needs to review the security of your connection before proceeding.</p>"
        "<noscript>Enable JavaScript and cookies to continue</noscript>"
        "<footer>Ray ID: 8f2a1b3c4d5e6f70 Performance &amp; security by "
        "Cloudflare</footer></body></html>"
    )
    # A fivefold margin: Cloudflare can grow its prose without the rule missing it.
    assert visible_chars(body) * 5 < 2_000
    assert classify_challenge(body.replace("Just a moment...", "x")) is (
        CloudflareChallengeError
    )


def test_title_proves_a_wall_whatever_the_page_carries() -> None:
    body = f"<html><head><title>Just a moment...</title></head><body>{_ARTICLE}"
    assert classify_challenge(body, on_success_body=True) is CloudflareChallengeError


# The markup of the pages that broke a fetch on 2026-10-30, cut to what decides it.

# TechSpot embeds Turnstile in a newsletter form; every article fetch failed as a
# Cloudflare wall.
_ARTICLE_WITH_TURNSTILE = (
    f"<!DOCTYPE html><html><head><title>News | TechSpot</title></head><body>"
    f'{_ARTICLE}<form><div class="cf-turnstile" data-sitekey="0x4AAAAAAE1J7knkm6DIN4X9">'
    "</div></form></body></html>"
)
# Reddit's network security refusing this IP: a bare HTML 403, no marker.
_REDDIT_403 = (
    "<body class=theme-beta><div><h1>You've been blocked by network security.</h1>"
    "<p>If you think you've been blocked by mistake, file a ticket below.</p></div>"
    "</body>"
)
# AWS WAF's challenge: AWS docs served it with a 405, Semantic Scholar and IMDb
# with a 202.
_AWS_WAF = (
    "<!DOCTYPE html><html><head><title></title>"
    '<script src="https://x.token.awswaf.com/challenge.js"></script></head><body>'
    "<script>AwsWafIntegration.getToken().then(() => location.reload())</script>"
    "<noscript>JavaScript is disabled. In order to continue, we need to verify that "
    "you're not a robot.</noscript></body></html>"
)
# DataDome's interstitial: Reuters and WSJ answered curl and Chrome with a 401.
_DATADOME = (
    "<html><head><title>reuters.com</title></head><body>"
    "<p>Please enable JS and disable any ad blocker</p>"
    '<script src="https://ct.captcha-delivery.com/i.js"></script></body></html>'
)
# PyTorch's docs: an unversioned page redirects by script and meta refresh.
_REDIRECT_STUB = (
    "<!DOCTYPE html><title>Redirecting&hellip;</title>"
    '<script>location.replace("../2.15/x.html")</script>'
    '<meta http-equiv="refresh" content="0; url=../2.15/x.html">'
)
# Kaggle: an empty mount point its scripts fill.
_APP_SHELL = (
    "<!DOCTYPE html><html><head><title>Kaggle</title>"
    '<script src="/static/app.js"></script></head><body><div id="root"></div>'
    "</body></html>"
)


def test_an_article_embedding_turnstile_is_content() -> None:
    assert classify_challenge(_ARTICLE_WITH_TURNSTILE, on_success_body=True) is None
    assert classify_challenge(_ARTICLE_WITH_TURNSTILE) is None


def test_a_firewalls_bare_403_is_an_unmarked_block() -> None:
    # No marker names it, so only its shape -- a bare HTML 403 -- earns another try.
    error = classify_http_error(
        "https://oauth.example/",
        403,
        {"content-type": "text/html; charset=UTF-8"},
        _REDDIT_403.encode(),
    )
    assert type(error) is FetchError
    assert is_unmarked_block(error, requested="https://oauth.example/")


@pytest.mark.parametrize(
    ("body", "on_success", "expected"),
    [
        (_AWS_WAF, False, AwsWafChallengeError),
        (_AWS_WAF, True, AwsWafChallengeError),
        (_DATADOME, False, DataDomeChallengeError),
    ],
)
def test_a_firewall_interstitial_is_classified(
    body: str,
    on_success: bool,
    expected: type[BotDetectionError],
) -> None:
    assert classify_challenge(body, on_success_body=on_success) is expected


@pytest.mark.parametrize(
    "body",
    [
        # A Brave result snippet naming DataDome's challenge host.
        b'{"web":{"results":[{"description":"blocked by captcha-delivery.com"}]}}',
        # A feed item quoting Cloudflare's markup in escaped HTML.
        (
            b"<?xml version='1.0'?><rss><channel><item><description>"
            b'&lt;div id="cf-challenge-running"&gt;</description></item></channel></rss>'
        ),
        b"%PDF-1.7 AwsWafIntegration.getToken()",
        # Feeds without a prolog, or with a comment before the root.
        b'<rss version="2.0"><channel><title>Just a moment...</title></channel></rss>',
        (
            b'<feed xmlns="http://www.w3.org/2005/Atom"><title>cf-challenge</title>'
            b'<entry><content>&lt;div id="cf-challenge-running"&gt;</content></entry></feed>'
        ),
        (
            b"<?xml version='1.0'?>\n<!-- generator: x -->\n<!-- v2 -->\n"
            b"<rss><channel><item><title>Just a moment...</title></item></channel></rss>"
        ),
        b"<!-- cached --><rdf:RDF><item>AwsWafIntegration</item></rdf:RDF>",
    ],
)
def test_a_non_html_success_body_is_never_a_wall(body: bytes) -> None:
    # The curl body of every automatic request is classified, APIs and feeds
    # too; a false wall would send the API to Chrome and pin its domain there.
    assert classify_challenge(body, on_success_body=True) is None


def test_html_is_recognized_past_a_byte_order_mark_and_without_a_doctype() -> None:
    wall = "\ufeff<head><title>Just a moment...</title></head>".encode()
    assert classify_challenge(wall, on_success_body=True) is CloudflareChallengeError


def test_xhtml_behind_an_xml_prolog_is_html() -> None:
    wall = (
        b'<?xml version="1.0" encoding="utf-8"?>\n<!-- served as xhtml -->\n'
        b'<html xmlns="http://www.w3.org/1999/xhtml"><head>'
        b"<title>Just a moment...</title></head></html>"
    )
    assert classify_challenge(wall, on_success_body=True) is CloudflareChallengeError


def test_a_firewall_script_on_a_page_of_content_is_a_beacon() -> None:
    body = f"<script>AwsWafIntegration.getToken()</script>{_ARTICLE}"
    assert classify_challenge(body, on_success_body=True) is None


@pytest.mark.parametrize(
    ("body", "shell"),
    [(_REDIRECT_STUB, True), (_APP_SHELL, True), (_ARTICLE_WITH_TURNSTILE, False)],
)
def test_a_page_without_content_is_a_shell(body: str, shell: bool) -> None:
    assert is_shell(body.encode()) is shell


@pytest.mark.parametrize(
    "body",
    [b'{"a": 1}', b"<?xml version='1.0'?><rss/>", b"plain text"],
)
def test_only_html_can_be_a_shell(body: bytes) -> None:
    assert not is_shell(body)


@pytest.mark.parametrize(
    ("page", "soon"),
    [
        (
            '<meta http-equiv="refresh" content="0;url=/httpservice/retry/enablejs">',
            True,
        ),
        ("<META HTTP-EQUIV=Refresh CONTENT='3; URL=../2.15/x.html'>", True),
        ('<meta http-equiv="refresh" content="60;url=/next">', False),
        ('<meta http-equiv="refresh" content="300">', False),
        ("<p>refresh content=0;url=/x</p>", False),
    ],
)
def test_only_an_imminent_meta_refresh_redirects_soon(page: str, soon: bool) -> None:
    assert redirects_soon(page) is soon


def test_a_redirect_stub_redirects_soon() -> None:
    # Chrome read PyTorch's stub before its refresh fired.
    assert redirects_soon(_REDIRECT_STUB)


@pytest.mark.parametrize("body", [b"", b" \r\n\t"])
def test_an_empty_success_is_a_shell(body: bytes) -> None:
    # AliExpress serves curl an empty 200; the browser renders the store.
    assert is_shell(body)


@pytest.mark.parametrize(
    ("body", "shell"),
    [
        # A short page with nothing to run is the whole page; rendering it only
        # adds a browser that can fail where curl succeeded.
        (b"<!doctype html><html><body><p>OK</p></body></html>", False),
        (b"<html><body>Page moved.</body></html>", False),
        # A script or an imminent refresh is what produces the real page.
        (b'<html><body><div id=root></div><script src="/a.js"></script>', True),
        (b'<html><meta http-equiv="refresh" content="0;url=/next"></html>', True),
        # A refresh far off is not a redirect worth rendering for.
        (b'<html><meta http-equiv="refresh" content="300;url=/n"></html>', False),
    ],
)
def test_a_shell_is_a_page_something_will_replace(body: bytes, shell: bool) -> None:
    assert is_shell(body) is shell


def test_visible_chars_skip_hidden_elements_in_any_case() -> None:
    assert visible_chars(b"<SCRIPT>x x</SCRIPT><p>ab cd</p><!-- e -->") == 4


@pytest.mark.parametrize(
    ("landed", "unmarked"),
    [
        # operations.nfl.com, 2026-10-30: curl is redirected to a 404 route the
        # browser never sees; Chrome gets the page with a 200.
        ("https://operations.nfl.com/v2/operations?_rsc", True),
        ("https://operations.nfl.com/", False),
    ],
)
def test_a_404_after_a_redirect_is_unmarked_but_one_at_the_url_is_not(
    landed: str,
    unmarked: bool,
) -> None:
    error = FetchError(landed, 404, {"content-type": "text/html"}, b"")
    assert is_unmarked_block(error, requested="https://operations.nfl.com/") is unmarked


@pytest.mark.parametrize(
    ("requested", "landed"),
    [
        # ``fetch`` strips the credentials and appends ``params`` before sending,
        # so the URL a plain 404 names differs from the caller's in those parts.
        ("https://u:p@search.example/x", "https://search.example/x"),
        ("https://search.example/x", "https://search.example/x?page=2"),
        ("https://Search.Example/x#top", "https://search.example/x"),
    ],
)
def test_a_404_at_the_url_asked_for_is_not_unmarked(
    requested: str,
    landed: str,
) -> None:
    error = FetchError(landed, 404, {"content-type": "text/html"}, b"")
    assert not is_unmarked_block(error, requested=requested)


@pytest.mark.parametrize(
    ("status", "content_type", "unmarked"),
    [
        (403, "text/html", True),
        (403, "TEXT/HTML; charset=utf-8", True),
        (401, "text/html", True),
        (405, "text/html", True),
        (503, "text/html", True),
        (403, "application/json", False),
        (404, "text/html", False),
        (429, "text/html", False),
        (500, "text/html", False),
    ],
)
def test_unmarked_block_is_an_html_refusal_only(
    status: int,
    content_type: str,
    unmarked: bool,
) -> None:
    error = FetchError("https://x.com", status, {"Content-Type": content_type}, b"")
    assert is_unmarked_block(error, requested="https://x.com") is unmarked
    assert not is_unmarked_block(
        FetchError("https://x.com", 403, {}, b""),
        requested="https://x.com",
    )


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
        "https://x.com",
        403,
        {},
        b'<div class="g-recaptcha"></div>',
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
    status: int,
    body: bytes,
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
    assert error.url == "https://x.com"
    assert error.status == 404
    assert error.headers == {"server": "nginx"}
    assert error.body == b"Not found"


def test_http_error_challenge_preserves_all_error_fields() -> None:
    error = classify_http_error(
        "https://x.com",
        403,
        {"cf-mitigated": "challenge"},
        b"body",
    )
    assert isinstance(error, CloudflareChallengeError)
    assert error.url == "https://x.com"
    assert error.status == 403
    assert error.headers == {"cf-mitigated": "challenge"}
    assert error.body == b"body"


def test_mitigation_header_requires_exact_trimmed_case_insensitive_value() -> None:
    assert isinstance(
        classify_http_error(
            "https://x.com",
            403,
            {"CF-MITIGATED": " Challenge "},
            b"x",
        ),
        CloudflareChallengeError,
    )
    assert type(classify_http_error("https://x.com", 403, {}, b"x")) is FetchError
    assert (
        type(
            classify_http_error(
                "https://x.com",
                403,
                {"cf-mitigated": "challenged"},
                b"x",
            ),
        )
        is FetchError
    )


def test_text_decodes_invalid_bytes_with_replacement_and_lowercases() -> None:
    assert _text(bytes([65, 66, 67, 255])) == "abc�"
    assert _text("ABC") == "abc"


def test_tags_preserve_opening_attributes_but_remove_script_body() -> None:
    assert _tags('<script id="trk_jschal_js">fake <div></script><div id="x">') == [
        '<script id="trk_jschal_js">',
        "</script>",
        '<div id="x">',
    ]


def test_occurrences_returns_every_overlapping_start_in_order() -> None:
    assert _occurrences("aaaa", "aa") == [0, 1, 2]
    assert _occurrences('<div data-sitekey="x">', "data-sitekey") == [5]


@pytest.mark.parametrize(
    ("tag", "marker", "expected"),
    [
        ('<div class="h-captcha">', "h-captcha", True),
        ('<div id="h-captcha">', "h-captcha", True),
        ('<div data-sitekey="x">', "data-sitekey", True),
        ('<a href="/data-sitekey">', "data-sitekey", False),
        ('<div data-sitekeyish="x">', "data-sitekey", False),
        ('<div class="x">', "data-sitekey", False),
    ],
)
def test_widget_occurrence_requires_identity_attribute_position(
    tag: str,
    marker: str,
    expected: bool,
) -> None:
    index = tag.find(marker)
    assert _is_widget_occurrence(tag, marker, index) is expected


def test_data_attribute_boundary_characters_are_all_supported() -> None:
    marker = "data-sitekey"
    assert _is_widget_occurrence(f" {marker}=x", marker, 1)
    assert _is_widget_occurrence(f" {marker} x", marker, 1)
    assert _is_widget_occurrence(f" {marker}", marker, 1)
    assert _is_widget_occurrence(f" {marker}>", marker, 1)
    assert not _is_widget_occurrence(f"{marker}=x", marker, 0)
    assert _is_widget_occurrence(f"{marker} ", marker, 0) is False
    assert not _is_widget_occurrence(f"x{marker}=x", marker, 1)
    assert not _is_widget_occurrence(f" {marker}?", marker, 1)


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
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
