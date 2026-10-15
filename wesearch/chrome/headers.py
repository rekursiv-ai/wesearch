"""Chrome request-header parity: the headers curl_cffi impersonation omits.

curl_cffi replays Chrome's TLS/HTTP-2 fingerprint and the THREE basic client
hints (``sec-ch-ua``, ``-mobile``, ``-platform``), but not the extended
low-entropy hints a real Chrome sends to every site, nor the Google-only
integrity headers it sends to Google properties. A request that carries a
Chrome User-Agent + TLS fingerprint yet omits these is provably not Chrome, a
bot signal no cookie or pacing change addresses.

Two layers, split by where a real Chrome sends them:

- :func:`chrome_client_hints` -- the extended UA client-hint set
  (``sec-ch-ua-arch`` etc.). Chrome sends these to EVERY site, so
  :mod:`wesearch.fetch` adds them on every curl request, matching the
  impersonated browser (desktop or Android). No extra requests.
- :func:`chrome_headers_for_google` -- the ``x-browser-*`` + ``x-client-data``
  headers Chrome sends ONLY to Google-owned hosts. A caller scraping a Google
  property (e.g. Scholar) layers these on; they must NOT go to arbitrary sites.
  ``x-browser-validation`` is the reverse-engineered integrity token:
  ``base64(sha1(platform_api_key + UA))``.

No Chrome version is written here: callers pass the major their transport
presents, and the brand list is derived from it. The parity test checks the rest
against a real browser::

    uv run --frozen pytest -m integration wesearch/chrome/

References:
    https://github.com/dsekz/chrome-x-browser-validation-header
        Reverse engineering of Chrome's ``x-browser-validation`` header.

"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final, Literal

import base64
import hashlib
import sys


__all__ = [
    "ChromePlatform",
    "chrome_brands",
    "chrome_client_hints",
    "chrome_headers_for_google",
    "chrome_identity_headers",
    "chrome_navigation_headers",
    "chrome_platform",
    "chrome_user_agent",
    "is_google_property",
    "is_identity_header",
    "user_agent_platform",
]

type ChromePlatform = Literal["Windows", "Linux", "macOS", "Android"]


def chrome_platform(impersonate: str, *, host: str = sys.platform) -> ChromePlatform:
    """Return the OS a Chrome impersonating ``impersonate`` presents.

    A desktop target names THIS host's OS, so a cookie the local Chrome earned
    returns from a browser naming the same OS.

    Args:
      impersonate: curl_cffi target or alias (``chrome``, ``chrome131_android``).
      host: ``sys.platform`` of the machine a desktop target presents as.

    Returns:
      platform: ``"Android"`` for an Android target, else the host's OS.

    """
    if impersonate.endswith("android"):
        return "Android"
    platforms: dict[str, ChromePlatform] = {"darwin": "macOS", "win32": "Windows"}
    return platforms.get(host, "Linux")


def chrome_brands(major: int, *, full_version: str = "") -> str:
    """Return Chrome's ``sec-ch-ua`` brand list for ``major``.

    Chromium derives the GREASE brand, its version, and the list order from the
    major version alone (``GenerateBrandVersionList``), so no release needs a
    table entry.

    Args:
      major: Chrome major version.
      full_version: Full version for ``sec-ch-ua-full-version-list``; empty
        gives the ``sec-ch-ua`` form, which carries majors only.

    Returns:
      brands: The header value.

    """
    chars = " (:-./);=?_"
    grease = f"Not{chars[major % 11]}A{chars[(major + 1) % 11]}Brand"
    grease_version = ("8", "99", "24")[major % 3] + (".0.0.0" if full_version else "")
    version = full_version or str(major)
    named = (
        (grease, grease_version),
        ("Chromium", version),
        ("Google Chrome", version),
    )
    order = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))[
        major % 6
    ]
    placed = sorted(zip(order, named, strict=True))
    return ", ".join(f'"{brand}";v="{v}"' for _, (brand, v) in placed)


def chrome_identity_headers(*, major: int, platform: ChromePlatform) -> dict[str, str]:
    """Return the basic client hints and User-Agent, cased as Chrome sends them.

    Args:
      major: Chrome major version.
      platform: The impersonated OS.

    Returns:
      headers: ``sec-ch-ua``, ``-mobile``, ``-platform``, and ``User-Agent``.

    """
    return {
        "sec-ch-ua": chrome_brands(major),
        "sec-ch-ua-mobile": "?1" if platform == "Android" else "?0",
        "sec-ch-ua-platform": f'"{platform}"',
        "User-Agent": chrome_user_agent(major, platform),
    }


def user_agent_platform(user_agent: str) -> ChromePlatform | None:
    """Return the OS a Chrome User-Agent names, or ``None`` for any other OS.

    Args:
      user_agent: A Chrome User-Agent string.

    Returns:
      platform: The platform its OS token names.

    """
    os_token = user_agent.partition("(")[2].partition(")")[0]
    if "Android" in os_token:
        return "Android"
    prefixes: dict[str, ChromePlatform] = {
        "Windows NT": "Windows",
        "X11; Linux": "Linux",
        "Macintosh": "macOS",
    }
    return next(
        (
            platform
            for prefix, platform in prefixes.items()
            if os_token.startswith(prefix)
        ),
        None,
    )


def is_identity_header(name: str) -> bool:
    """Whether ``name`` describes the browser rather than the request.

    Args:
      name: Header name, any case.

    Returns:
      owned: True for the User-Agent, client hints, and Google integrity set.

    """
    lowered = name.lower()
    return lowered in ("user-agent", "x-client-data") or lowered.startswith(
        ("sec-ch-ua", "x-browser-"),
    )


# Per-platform Google API keys hard-coded in chrome.dll, used as the prefix when
# hashing the User-Agent into x-browser-validation. From the reverse-engineering
# above; stable across Chrome builds to date.
_PLATFORM_API_KEY: Final[dict[ChromePlatform, str]] = {
    "Windows": "AIzaSyA2KlwBX3mkFo30om9LUFYQhpqLoa_BNhE",
    "Linux": "AIzaSyBqJZh-7pA44blAaAkH6490hUFOwX0KCYM",
    "macOS": "AIzaSyDr2UxVnv_U85AbhhY8XSHSIavUW0DC-sY",
    "Android": "AIzaSyA8Hr8czk2cWyLu-a_RSPVYUEHozUnu6bA",
}

# The UA OS token curl_cffi presents per platform (the string inside the UA's
# first parenthesis). Android carries a device model too, but a generic "K" is
# what a headless-of-recent Chrome reports.
_UA_OS: Final[dict[ChromePlatform, str]] = {
    "Windows": "Windows NT 10.0; Win64; x64",
    "Linux": "X11; Linux x86_64",
    "macOS": "Macintosh; Intel Mac OS X 10_15_7",
    "Android": "Linux; Android 10; K",
}


def chrome_user_agent(major: int, platform: str = "macOS") -> str:
    """Return the User-Agent string a Chrome of this major/platform presents.

    Args:
      major: Chrome major version number.
      platform: OS platform (Windows/Linux/macOS/Android).

    Returns:
      result: User-Agent header value for this Chrome version and platform.

    """
    os_name = _UA_OS.get(platform)
    if os_name is None:
        raise ValueError(f"Unsupported platform {platform!r}.")
    tail = "Mobile Safari/537.36" if platform == "Android" else "Safari/537.36"
    return (
        f"Mozilla/5.0 ({os_name}) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{major}.0.0.0 {tail}"
    )


def chrome_navigation_headers(
    *,
    major: int,
    platform: ChromePlatform = "macOS",
    method: str = "GET",
    content_type: str = "",
    origin: str = "",
    http2: bool = False,
) -> dict[str, str]:
    """Return the full Chrome header set for a request, in exact wire order.

    The complete core identity a real Chrome sends to any site -- the three basic
    client hints, User-Agent, Accept, Sec-Fetch-*, Priority, Accept-Encoding/
    Language -- matching the captured wire order of curl_cffi's impersonation
    (verified against real Chrome). Used by the stdlib transport, which has no
    impersonation and must reproduce the whole set by hand; the curl transport
    lets its impersonation emit these and adds only the Accept-CH extensions.

    Args:
      major: Chrome major version.
      platform: The impersonated OS (selects UA + ``sec-ch-ua-platform``).
      method: HTTP method; GET/HEAD get the navigation shape, others the
        fetch/XHR shape (``Accept: */*``, ``Origin``, cross-site Sec-Fetch).
      content_type: Request body content type, added for a non-GET/HEAD.
      origin: The request origin (``scheme://host``) for a non-GET/HEAD.
      http2: Whether the request rides HTTP/2. The ``Priority`` header is an
        HTTP/2 construct; a real Chrome omits it on HTTP/1.1, so the stdlib
        transport (HTTP/1.1 only) must leave ``http2`` false to match. The
        curl transport carries ``Priority`` on HTTP/1.1 regardless (curl_cffi's
        impersonation injects it pre-negotiation), a known curl_cffi limitation.

    Returns:
      headers: The ordered Chrome header map.

    """
    identity = chrome_identity_headers(major=major, platform=platform)
    user_agent = identity.pop("User-Agent")
    h = identity
    if method in ("GET", "HEAD"):
        h["Upgrade-Insecure-Requests"] = "1"
        h["User-Agent"] = user_agent
        h["Accept"] = (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        )
        h["Sec-Fetch-Site"] = "none"
        h["Sec-Fetch-Mode"] = "navigate"
        h["Sec-Fetch-User"] = "?1"
        h["Sec-Fetch-Dest"] = "document"
    else:
        h["User-Agent"] = user_agent
        h["Accept"] = "*/*"
        if content_type:
            h["Content-Type"] = content_type
        if origin:
            h["Origin"] = origin
        h["Sec-Fetch-Site"] = "cross-site"
        h["Sec-Fetch-Mode"] = "cors"
        h["Sec-Fetch-Dest"] = "empty"
    h["Accept-Encoding"] = "gzip, deflate, br, zstd"
    h["Accept-Language"] = "en-US,en;q=0.9"
    if http2:
        h["Priority"] = "u=0, i"
    return h


def chrome_client_hints(
    *,
    major: int,
    platform: ChromePlatform = "macOS",
    full_version: str = "",
) -> dict[str, str]:
    """Return the extended UA client-hint headers a real Chrome sends everywhere.

    The three basic hints (``sec-ch-ua`` / ``-mobile`` / ``-platform``) are
    supplied by curl_cffi's impersonation; this adds the low-entropy extensions
    it omits, matching the impersonated browser. Mobile (Android) reports
    ``sec-ch-ua-mobile: ?1`` and empty ``-arch``/``-bitness`` (a phone has no
    desktop CPU hints), which is what a real mobile Chrome sends.

    Args:
      major: Chrome major version (must equal the impersonated UA's version, or
        the hints and the UA disagree -- itself a tell).
      platform: The impersonated OS; selects ``sec-ch-ua-platform`` and the
        mobile/arch shape.
      full_version: Full Chrome version (e.g. ``"146.0.7379.0"``) for
        ``sec-ch-ua-full-version-list``; defaults to ``"<major>.0.0.0"``.

    Returns:
      headers: The extended client-hint header map to merge onto a request.

    """
    if platform not in _UA_OS:
        raise ValueError(f"Unsupported platform {platform!r}.")
    mobile = platform == "Android"
    # Same GREASE brand and order as sec-ch-ua: a real Chrome derives both lists
    # from one brand set, so a mismatch between them is itself a tell.
    full_list = chrome_brands(major, full_version=full_version or f"{major}.0.0.0")
    # Key order mirrors a real Chrome's opted-in wire order exactly:
    # arch, platform-version, model, bitness, wow64, full-version-list.
    return {
        "sec-ch-ua-arch": '""' if mobile else '"x86"',
        "sec-ch-ua-platform-version": '""',
        "sec-ch-ua-model": '"K"' if mobile else '""',
        "sec-ch-ua-bitness": '""' if mobile else '"64"',
        "sec-ch-ua-wow64": "?0",
        "sec-ch-ua-full-version-list": full_list,
    }


def is_google_property(host: str) -> bool:
    """Whether a real Chrome sends its ``x-browser-*`` integrity headers to ``host``.

    Chrome scopes the Google-only header set to Google-owned properties; a
    request to any other host must not carry them (sending them elsewhere is
    itself a bot tell). Matches the registrable Google domains by suffix.

    Args:
      host: The bare DNS hostname of the request target.

    Returns:
      is_google: True when ``host`` is a Google-owned property.

    """
    host = host.casefold()
    while host.endswith("."):
        host = host[:-1]
    return any(
        host == suffix or host.endswith(f".{suffix}")
        for suffix in _GOOGLE_HOST_SUFFIXES
    )


# The Google-owned property suffixes a real Chrome sends x-browser-*/x-client-data
# to. Not an exhaustive Google-domain list -- the properties this library fetches
# (Search, Scholar, and their static/api hosts).

_GOOGLE_HOST_SUFFIXES: Final = (
    "google.com",
    "gstatic.com",
    "googleapis.com",
    "googleusercontent.com",
)


def chrome_headers_for_google(
    *,
    major: int,
    platform: ChromePlatform = "macOS",
    x_client_data: str = "CInbygE=",
) -> dict[str, str]:
    """Return the Google-ONLY Chrome headers (``x-browser-*`` + ``x-client-data``).

    These are the integrity/telemetry headers a real Chrome sends ONLY to
    Google-owned hosts (google.com, scholar.google.com, gstatic, googleapis) --
    send them to a Google property, never to an arbitrary site. The extended
    client hints are NOT included here: :mod:`wesearch.fetch` adds those to
    every origin that opts in via ``Accept-CH`` (Google does), matching a real
    browser, so a caller need only add the Google-only set.

    Args:
      major: Chrome major version; must match the impersonated UA.
      platform: The impersonated OS; selects the validation API key + UA.
      x_client_data: The opaque ``x-client-data`` install token (a real
        low-entropy value by default, shared across many installs).

    Returns:
      headers: The Google-only header map.

    """
    return {
        "x-browser-channel": "stable",
        "x-browser-year": str(datetime.now(UTC).year),
        "x-browser-validation": _validation(
            chrome_user_agent(major, platform),
            platform,
        ),
        "x-client-data": x_client_data,
    }


def _validation(user_agent: str, platform: ChromePlatform) -> str:
    """Compute ``x-browser-validation``: ``base64(sha1(api_key + UA))``."""
    data = (_PLATFORM_API_KEY[platform] + user_agent).encode()
    return base64.b64encode(hashlib.sha1(data).digest()).decode()  # noqa: S324 -- Chrome compatibility requires this exact SHA-1 validation formula.
