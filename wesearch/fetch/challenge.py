"""Recognize Cloudflare challenges and CAPTCHA widgets in a response.

Matches page structure and titles, never prose, after FlareSolverr's model.

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
    if any(marker in tag for tag in _tags(text) for marker in cloudflare):
        return CloudflareChallengeError
    if on_success_body:
        return None
    if any(marker in text for marker in (*cloudflare, *cloudflare_ambient)):
        return CloudflareChallengeError
    if any(
        _is_widget_occurrence(tag, marker, index)
        for tag in _tags(text)
        for marker in puzzle_widget
        for index in _occurrences(tag, marker)
    ):
        return PuzzleChallengeError
    return None


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
