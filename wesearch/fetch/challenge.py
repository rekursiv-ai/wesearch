"""Recognize Cloudflare challenges and CAPTCHA widgets in a response.

Matches page structure and titles, never prose, after FlareSolverr's model.

FlareSolverr -- MIT License, Copyright (c) 2025 Diego Heras (ngosang).
https://github.com/FlareSolverr/FlareSolverr
"""

from __future__ import annotations

from urllib.parse import urlparse

import re

from wesearch.types.errors import (
    AwsWafChallengeError,
    BotDetectionError,
    CloudflareChallengeError,
    DataDomeChallengeError,
    FetchError,
    PuzzleChallengeError,
)


__all__ = [
    "classify_challenge",
    "classify_http_error",
    "is_shell",
    "is_unmarked_block",
    "is_wall_title",
    "redirects_soon",
    "visible_chars",
]


def is_wall_title(
    title: str | None,
    *,
    cloudflare_titles: tuple[str, ...] = (
        "just a moment...",
        "attention required! | cloudflare",
    ),
) -> bool:
    """Whether ``title`` is the title of Cloudflare's interstitial.

    A renderer that converts a page (the reader proxy's ``Title:`` line) keeps
    its title, so the title proves a wall in any format.

    Args:
      title: A page title, any case, or ``None`` when the page has none.
      cloudflare_titles: Cloudflare interstitial titles, lower case.

    Returns:
      walled: Whether the page is the interstitial.

    """
    return title is not None and title.strip().lower() in cloudflare_titles


def classify_challenge(
    content: str | bytes,
    *,
    on_success_body: bool = False,
    cloudflare: tuple[str, ...] = (
        "cf_chl",
        "cf-challenge",
        "cf-challenge-running",
        "cf-please-wait",
        "challenge-spinner",
        "trk_jschal_js",
        "turnstile-wrapper",
    ),
    # Turnstile's embeddable widget: any site can put it on an ordinary page (a
    # comment form), so like reCAPTCHA it proves a wall only on an error body.
    turnstile_widget: tuple[str, ...] = ("cf-turnstile", "cf-turnstile-response"),
    cloudflare_ambient: tuple[str, ...] = ("/cdn-cgi/challenge-platform/",),
    puzzle_widget: tuple[str, ...] = (
        "g-recaptcha",
        "recaptcha",
        "h-captcha",
        "hcaptcha",
        "data-sitekey",
    ),
    # A firewall's interstitial is identified by the script that runs it, so these
    # are read in script bodies too; the visible-text bound keeps a page that only
    # loads the vendor's beacon from counting.
    wall_scripts: tuple[tuple[str, type[BotDetectionError]], ...] = (
        ("awswafintegration", AwsWafChallengeError),
        ("captcha-delivery.com", DataDomeChallengeError),
    ),
    max_wall_text_chars: int = 2_000,
) -> type[BotDetectionError] | None:
    """Return the shared challenge class ``content`` proves, if any.

    Args:
      content: Response body.
      on_success_body: Whether the body arrived with a success status. Generic
        CAPTCHA widgets and Cloudflare's ambient beacon are inconclusive on a
        successful page, so only structural interstitial evidence is accepted.
      cloudflare: Cloudflare challenge-only markup markers.
      turnstile_widget: Cloudflare Turnstile widget markers.
      cloudflare_ambient: Cloudflare markers also present on ordinary proxied pages.
      puzzle_widget: Generic CAPTCHA widget markers.
      wall_scripts: Markers of a firewall's interstitial script, each with the
        error it proves.
      max_wall_text_chars: Most visible non-space characters a wall carries. A
        page with more is the content itself, whatever widgets it embeds.

    Returns:
      error_type: The proven challenge type, or ``None``.

    """
    # A JSON, feed, or PDF success body is the content itself: one that merely
    # mentions a vendor's markers must not send an API to the browser.
    if on_success_body and not _is_html(content):
        return None
    text = _text(content)
    if is_wall_title(_page_title(text)):
        return CloudflareChallengeError
    # Every wall is an interstitial with a sentence or two; markers on a page that
    # has an article are a widget or a beacon the site embeds, never the block.
    if visible_chars(text) > max_wall_text_chars:
        return None
    if any(marker in tag for tag in _tags(text) for marker in cloudflare):
        return CloudflareChallengeError
    for marker, error_type in wall_scripts:
        if marker in text:
            return error_type
    if on_success_body:
        return None
    if any(
        marker in text
        for marker in (*cloudflare, *turnstile_widget, *cloudflare_ambient)
    ):
        return CloudflareChallengeError
    if any(
        _is_widget_occurrence(tag, marker, index)
        for tag in _tags(text)
        for marker in puzzle_widget
        for index in _occurrences(tag, marker)
    ):
        return PuzzleChallengeError
    return None


def is_unmarked_block(
    error: FetchError,
    *,
    requested: str,
    refusal_statuses: frozenset[int] = frozenset({401, 403, 405, 503}),
) -> bool:
    """Whether ``error`` may be a firewall's block that carries no challenge marker.

    A fingerprinting firewall refuses a flagged client with a bare HTML error
    page: Reddit and eBay a 403, Reuters and WSJ a 401, AWS docs a 405, Amazon
    a 503. Or it redirects the client to a route that 404s (operations.nfl.com):
    a 404 proves only that the URL it answers is absent, not the one asked for.
    Only another client or IP tells either from the site's own refusal. An
    API's error is JSON, a bad key that neither clears.

    Args:
      error: The failure to judge.
      requested: The URL the caller asked for, before any redirect.
      refusal_statuses: Statuses a firewall answers a flagged client with.

    Returns:
      unmarked: Whether another transport or IP is worth one try.

    """
    refused = error.status in refusal_statuses or (
        error.status == 404 and _resource(error.url) != _resource(requested)
    )
    return refused and any(
        key.lower() == "content-type" and "text/html" in value.lower()
        for key, value in error.headers.items()
    )


def is_shell(body: bytes, *, min_text_chars: int = 500) -> bool:
    """Whether ``body`` is a page something will replace, so a browser must render it.

    An empty 200 (AliExpress), an imminent meta refresh (PyTorch's docs), or a
    short HTML page with a script to fill it (Kaggle, X, claude.com). A short
    page with nothing to run is the page itself, however brief.

    Args:
      body: A successful response body.
      min_text_chars: Fewest visible non-space characters a page of content has.

    Returns:
      shell: Whether rendering it in a browser may yield the content.

    """
    if not body.strip():
        return True
    if not _is_html(body):
        return False
    return redirects_soon(body) or (
        visible_chars(body) < min_text_chars and b"<script" in body.lower()
    )


def classify_http_error(
    url: str,
    status: int,
    headers: dict[str, str],
    body: bytes,
) -> FetchError:
    """Return the error for an HTTP failure, a challenge error when one is proven.

    Proof is challenge markup or ``cf-mitigated: challenge``; ``server:
    cloudflare`` is not, since Cloudflare also fronts the origin's own errors.

    References:
      https://developers.cloudflare.com/cloudflare-challenges/challenge-types/challenge-pages/detect-response/
        ``cf-mitigated`` is present on all Challenge Page types, and
        ``challenge`` is its only valid value.

    Args:
      url: Requested URL.
      status: HTTP status code.
      headers: Response headers.
      body: Decompressed response body.

    Returns:
      error: A challenge error when detected, otherwise ``FetchError``.

    """
    error_type = classify_challenge(body)
    if error_type is None and _is_cloudflare_mitigated(headers):
        error_type = CloudflareChallengeError
    if error_type is None:
        return FetchError(url, status, headers, body)
    return error_type(url=url, status=status, headers=headers, body=body)


def redirects_soon(
    content: str | bytes,
    *,
    max_delay_sec: int = 5,
    meta_refresh: re.Pattern[str] = re.compile(
        r"""<meta\b[^>]*http-equiv\s*=\s*["']?refresh["']?[^>]*"""
        r"""content\s*=\s*["']?\s*(\d+)\s*[;,]\s*url\s*=""",
        re.IGNORECASE,
    ),
) -> bool:
    """Whether ``content`` is a page a meta refresh is about to replace.

    Google's enablejs page and PyTorch's versioned docs redirect this way; a
    browser that reads the page before the refresh fires returns the stub.

    Args:
      content: An HTML page.
      max_delay_sec: Longest refresh delay worth waiting for.
      meta_refresh: A ``<meta http-equiv="refresh" content="N; url=...">`` tag.

    Returns:
      soon: Whether a refresh to another URL fires within ``max_delay_sec``.

    """
    text = content.decode(errors="replace") if isinstance(content, bytes) else content
    match = meta_refresh.search(text)
    return match is not None and int(match[1]) <= max_delay_sec


def visible_chars(
    content: str | bytes,
    *,
    hidden: re.Pattern[str] = re.compile(
        r"<(script|style|noscript|template)\b.*?</\1\s*>|<!--.*?-->",
        re.DOTALL | re.IGNORECASE,
    ),
    tag: re.Pattern[str] = re.compile(r"<[^>]+>"),
) -> int:
    """Count the non-space characters a reader of the rendered page sees.

    Args:
      content: An HTML page.
      hidden: Elements whose text is never shown.
      tag: One tag.

    Returns:
      count: Visible characters, whitespace excluded.

    """
    text = content.decode(errors="replace") if isinstance(content, bytes) else content
    return sum(len(word) for word in tag.sub(" ", hidden.sub(" ", text)).split())


def _tags(
    text: str,
    *,
    tag: re.Pattern[str] = re.compile(r"<[^>]+>", re.DOTALL),
    # Script and style bodies hold markup-looking strings; their tags stay, since
    # Cloudflare's ``trk_jschal_js`` marker is an attribute of a script tag.
    script_body: re.Pattern[str] = re.compile(
        r"(<(script|style)\b[^>]*>).*?(</\2>)",
        re.DOTALL | re.IGNORECASE,
    ),
) -> list[str]:
    """Return the rendered tags of ``text``, with script/style bodies removed."""
    return [
        match.group(0)
        for match in tag.finditer(script_body.sub(_replace_script_body, text))
    ]


def _replace_script_body(match: re.Match[str]) -> str:
    """Keep script/style tags while removing their contents."""
    opening = match.group(1)
    closing = match.group(3)
    assert isinstance(opening, str)
    assert isinstance(closing, str)
    return opening + closing


# A widget is a marker in a class or id value, or a ``data-`` marker as an attribute
# name; one in a URL or a JSON string only mentions a CAPTCHA provider.
def _is_widget_occurrence(
    tag: str,
    marker: str,
    index: int,
    *,
    attr_context: re.Pattern[str] = re.compile(r'(?:class|id)\s*=\s*["\'][^"\']*$'),
) -> bool:
    """Whether ``marker`` at ``index`` names this tag's element, not its text."""
    if attr_context.search(tag[:index]) is not None:
        return True
    if not marker.startswith("data-"):
        return False
    after = tag[index + len(marker) : index + len(marker) + 1]
    return (index > 0 and tag[index - 1].isspace()) and after in ("=", "", " ", ">")


def _occurrences(tag: str, marker: str) -> list[int]:
    """Every start index of ``marker`` within ``tag``."""
    found: list[int] = []
    index = tag.find(marker)
    while index >= 0:
        found.append(index)
        index = tag.find(marker, index + 1)
    return found


def _is_cloudflare_mitigated(headers: dict[str, str]) -> bool:
    """Whether Cloudflare declares it served a challenge page for this response."""
    lower_headers = {key.lower(): value.lower() for key, value in headers.items()}
    value = lower_headers.get("cf-mitigated")
    return value is not None and value.strip() == "challenge"


# Negative on purpose: walls open with a comment (Amazon, DuckDuckGo), a bare
# <body> (Reddit), or a BOM, so only a body that is provably something else --
# JSON, a feed, a PDF, plain text -- counts as not HTML.
def _is_html(
    content: str | bytes,
    *,
    # A feed's root may follow a prolog, comments, both, or neither.
    data_start: re.Pattern[str] = re.compile(
        r"\s*(?:[{\[]|%PDF-|(?:<\?xml[^>]*>\s*)?(?:<!--.*?-->\s*)*"
        r"<(?:rss|feed|rdf:RDF)\b)",
        re.IGNORECASE | re.DOTALL,
    ),
) -> bool:
    """Whether ``content`` may be an HTML page: markup, and not JSON, a feed, or a PDF."""
    head = content[:2048]
    text = (head.decode(errors="replace") if isinstance(head, bytes) else head).lstrip(
        "\ufeff",
    )
    return text.lstrip().startswith("<") and data_start.match(text) is None


def _resource(url: str) -> tuple[str, str]:
    """Return the host and path ``url`` names, without credentials or query."""
    parsed = urlparse(url)
    return (parsed.hostname or "", parsed.path or "/")


def _text(content: str | bytes) -> str:
    decoded = (
        content.decode(errors="replace") if isinstance(content, bytes) else content
    )
    return decoded.lower()


def _page_title(
    text: str,
    *,
    title: re.Pattern[str] = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL),
) -> str | None:
    match = title.search(text)
    if match is None:
        return None
    found = match[1]
    assert isinstance(found, str)
    return found.strip()
