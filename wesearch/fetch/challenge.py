"""Cross-site HTTP challenge detection for ``wesearch.fetch``.

This module recognizes only challenge technology shared across providers:
Cloudflare interstitials and generic CAPTCHA widgets. Provider page states belong
beside their parsers.

Cloudflare markers and titles follow FlareSolverr's structural detection model:
challenge-only markup or an interstitial title, never prose keywords.

FlareSolverr -- MIT License, Copyright (c) 2025 Diego Heras (ngosang).
https://github.com/FlareSolverr/FlareSolverr
"""

from __future__ import annotations

import re

from wesearch.types.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    FetchError,
    PuzzleChallengeError,
)


__all__ = ["classify_challenge", "classify_http_error"]

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
# Script and style bodies are TEXT, not markup: a template literal or a string
# constant holding ``<div class="g-recaptcha">`` is indistinguishable from
# served markup to a tag regex, and read as a challenge on ordinary pages.
_SCRIPT_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.DOTALL)


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
        "cf-turnstile",
        "cf-turnstile-response",
    ),
    cloudflare_ambient: tuple[str, ...] = ("/cdn-cgi/challenge-platform/",),
    cloudflare_titles: tuple[str, ...] = (
        "just a moment...",
        "attention required! | cloudflare",
    ),
    puzzle_widget: tuple[str, ...] = (
        "g-recaptcha",
        "recaptcha",
        "h-captcha",
        "hcaptcha",
        "data-sitekey",
    ),
) -> type[BotDetectionError] | None:
    """Return the shared challenge class ``content`` proves, if any.

    Args:
      content: Response body.
      on_success_body: Whether the body arrived with a success status. Generic
        CAPTCHA widgets and Cloudflare's ambient beacon are inconclusive on a
        successful page, so only structural interstitial evidence is accepted.
      cloudflare: Cloudflare challenge-only markup markers.
      cloudflare_ambient: Cloudflare markers also present on ordinary proxied pages.
      cloudflare_titles: Cloudflare interstitial page titles.
      puzzle_widget: Generic CAPTCHA widget markers.

    Returns:
      error_type: The proven challenge type, or ``None``.

    """
    text = _text(content)
    title = _page_title(text)
    if title in cloudflare_titles:
        return CloudflareChallengeError
    if _has_markup_marker(text, cloudflare):
        return CloudflareChallengeError
    if on_success_body:
        return None
    if any(marker in text for marker in (*cloudflare, *cloudflare_ambient)):
        return CloudflareChallengeError
    if _has_widget_marker(text, puzzle_widget):
        return PuzzleChallengeError
    return None


def classify_http_error(
    url: str,
    status: int,
    headers: dict[str, str],
    body: bytes,
) -> FetchError:
    """Build the most specific error proven by an HTTP failure.

    Only evidence that Cloudflare AUTHORED the response counts: its challenge
    markup, or the ``cf-mitigated: challenge`` header it sets on every challenge
    page type. Presence alone does not -- ``server: cloudflare`` and ``cf-ray``
    ride every proxied response, including the origin's own 403s and 503s, so
    reading them as mitigation made an expired API token a bot wall. That sends
    the caller to the browser and pins the domain there permanently, since the
    learned routing list has no expiry.

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


def _text(content: str | bytes) -> str:
    decoded = (
        content.decode("utf-8", "replace") if isinstance(content, bytes) else content
    )
    return decoded.lower()


def _page_title(text: str) -> str | None:
    match = _TITLE_RE.search(text)
    if match is None:
        return None
    title = match[1]
    assert isinstance(title, str)
    return title.strip()


def _tags(text: str) -> list[str]:
    """The rendered tags of ``text``, with script/style bodies removed."""
    # ``finditer`` + ``group(0)``, not ``findall``: typeshed types the latter's
    # result as ``list[Any]``, which erases the element type downstream.
    return [match.group(0) for match in _TAG_RE.finditer(_SCRIPT_RE.sub("", text))]


def _has_markup_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in tag for tag in _tags(text) for marker in markers)


# A served CAPTCHA widget is a marker used as an element identity -- a class or
# id value (``class="h-captcha"``) or the ``data-sitekey`` attribute -- inside a
# rendered tag. A marker inside a ``src``/``href`` URL (``js.hcaptcha.com``) or a
# JSON string (Wikipedia's ``mw.config`` names its edit-captcha backend) is a
# mention, not a challenge; matching those raised a false PuzzleChallengeError on
# ordinary pages. This restricts the match to the widget's structural anatomy.
_ATTR_CONTEXT_RE = re.compile(r'(?:class|id)\s*=\s*["\'][^"\']*$')


def _has_widget_marker(text: str, markers: tuple[str, ...]) -> bool:
    """Whether a puzzle-widget marker appears as an element class/id/attribute.

    Restricts each marker to a class/id attribute value or a bare attribute name
    (``data-sitekey``) inside an HTML tag, excluding URLs and JSON strings that
    merely name a captcha provider.

    EVERY occurrence in a tag is examined, not the first: an earlier mention in
    an unrelated attribute (``data-provider="hcaptcha" class="hcaptcha"``) is
    not the widget, and stopping there masked the real class beside it.
    """
    return any(
        _is_widget_occurrence(tag, marker, index)
        for tag in _tags(text)
        for marker in markers
        for index in _occurrences(tag, marker)
    )


def _occurrences(tag: str, marker: str) -> list[int]:
    """Every start index of ``marker`` within ``tag``."""
    found: list[int] = []
    index = tag.find(marker)
    while index >= 0:
        found.append(index)
        index = tag.find(marker, index + 1)
    return found


def _is_widget_occurrence(tag: str, marker: str, index: int) -> bool:
    """Whether ``marker`` at ``index`` names this tag's element, not its text.

    Two positions count. A marker inside a class/id attribute VALUE identifies
    the element. A ``data-`` marker additionally counts as an attribute NAME --
    but only in that position: accepted anywhere in the tag, a link to
    ``/docs/data-sitekey`` read as a served widget.
    """
    if _ATTR_CONTEXT_RE.search(tag[:index]) is not None:
        return True
    if not marker.startswith("data-"):
        return False
    after = tag[index + len(marker) : index + len(marker) + 1]
    return (index > 0 and tag[index - 1].isspace()) and after in ("=", "", " ", ">")


def _is_cloudflare_mitigated(headers: dict[str, str]) -> bool:
    """Whether Cloudflare declares it served a challenge page for this response."""
    lower_headers = {key.lower(): value.lower() for key, value in headers.items()}
    return lower_headers.get("cf-mitigated", "").strip() == "challenge"
