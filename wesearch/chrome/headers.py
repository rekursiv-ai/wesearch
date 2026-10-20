"""Chrome request headers: User-Agent, client hints, and Google-only headers.

- :func:`chrome_identity_headers` / :func:`chrome_navigation_headers` -- the
  User-Agent and basic client hints, alone or in a full request header set.
- :func:`chrome_client_hints` -- the extended client hints (``sec-ch-ua-arch``
  etc.) Chrome sends only to an origin that requested them via ``Accept-CH``.
- :func:`chrome_headers_for_google` -- the ``x-browser-*`` + ``x-client-data``
  headers Chrome sends ONLY to Google-owned hosts; never send them elsewhere.

Every function takes the Chrome major version as an argument.

References:
    https://github.com/dsekz/chrome-x-browser-validation-header
        Reverse engineering of Chrome's ``x-browser-validation`` header.

"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal

import base64
import hashlib
import sys


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "ChromePlatform",
    "chrome_brands",
    "chrome_client_hints",
    "chrome_headers_for_google",
    "chrome_headers_for_host",
    "chrome_identity_headers",
    "chrome_navigation_headers",
    "chrome_platform",
    "chrome_user_agent",
    "is_google_property",
    "is_identity_header",
    "user_agent_platform",
]

type ChromePlatform = Literal["Windows", "Linux", "macOS", "Android"]


# Chrome's reduced User-Agent reports the Android device model as "K".
_UA_OS: Final[Mapping[str, str]] = {
    "Windows": "Windows NT 10.0; Win64; x64",
    "Linux": "X11; Linux x86_64",
    "macOS": "Macintosh; Intel Mac OS X 10_15_7",
    "Android": "Linux; Android 10; K",
}


def chrome_platform(impersonate: str, *, host: str = sys.platform) -> ChromePlatform:
    """Return the OS a Chrome impersonating ``impersonate`` presents.

    A desktop target presents ``host``'s OS.

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

    Mirrors Chromium's ``GenerateBrandVersionList``, which derives the GREASE
    brand and list order from the major version alone.

    Args:
      major: Chrome major version.
      full_version: Full version for ``sec-ch-ua-full-version-list``; empty
        gives the ``sec-ch-ua`` form, which carries majors only.

    Returns:
      brands: The header value, e.g. ``"Chromium";v="146", ...``.

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
      platform: The platform its OS token names, or ``None``.

    """
    os_token = user_agent.partition("(")[2].partition(")")[0]
    if "Android" in os_token:
        return "Android"
    # Distro builds insert a vendor token ("X11; Ubuntu; Linux x86_64").
    parts = os_token.split("; ")
    if parts[0] == "X11" and any(part.startswith("Linux") for part in parts[1:]):
        return "Linux"
    prefixes: dict[str, ChromePlatform] = {
        "Windows NT": "Windows",
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


def chrome_user_agent(major: int, platform: str) -> str:
    """Return the User-Agent string a Chrome of this major/platform presents.

    Args:
      major: Chrome major version number.
      platform: OS platform (Windows/Linux/macOS/Android).

    Returns:
      user_agent: The reduced User-Agent, e.g. ``... Chrome/146.0.0.0 ...``.

    Raises:
      ValueError: If ``platform`` is not a :data:`ChromePlatform`.

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
    platform: ChromePlatform,
    method: str = "GET",
    content_type: str = "",
    origin: str = "",
    http2: bool = False,
) -> dict[str, str]:
    """Return the full Chrome header set for a request, in exact wire order.

    Covers the basic client hints, User-Agent, Accept, Sec-Fetch-*, Priority,
    and Accept-Encoding/Language, for a client with no impersonation of its own.

    Args:
      major: Chrome major version.
      platform: The impersonated OS (selects UA + ``sec-ch-ua-platform``).
      method: HTTP method; GET/HEAD get the navigation shape, others the
        fetch/XHR shape (``Accept: */*``, ``Origin``, cross-site Sec-Fetch).
      content_type: Request body content type, added for a non-GET/HEAD.
      origin: The request origin (``scheme://host``) for a non-GET/HEAD.
      http2: Whether the request rides HTTP/2; adds ``Priority``, which Chrome
        omits on HTTP/1.1.

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
    platform: ChromePlatform = "Linux",
    full_version: str = "",
) -> dict[str, str]:
    """Return the extended UA client-hint headers an origin opts into.

    The basic hints come from :func:`chrome_identity_headers`. Android reports
    empty ``-arch``/``-bitness``.

    Args:
      major: Chrome major version; must match the impersonated UA.
      platform: The impersonated OS; selects ``sec-ch-ua-platform`` and the
        mobile/arch shape.
      full_version: Full Chrome version (e.g. ``"146.0.7379.0"``) for
        ``sec-ch-ua-full-version-list``; defaults to ``"<major>.0.0.0"``.

    Returns:
      headers: The extended client-hint header map, in Chrome's wire order.

    Raises:
      ValueError: If ``platform`` is not a :data:`ChromePlatform`.

    """
    if platform not in _UA_OS:
        raise ValueError(f"Unsupported platform {platform!r}.")
    mobile = platform == "Android"
    full_list = chrome_brands(major, full_version=full_version or f"{major}.0.0.0")
    # Key order is Chrome's wire order; do not sort.
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

    Matches ``host`` by suffix against the Google domains this library fetches.

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
        # Not exhaustive: only the Google properties this library fetches.
        for suffix in (
            "google.com",
            "gstatic.com",
            "googleapis.com",
            "googleusercontent.com",
        )
    )


def chrome_headers_for_google(
    *,
    major: int,
    platform: ChromePlatform,
    x_client_data: str = "CInbygE=",
) -> dict[str, str]:
    """Return the Google-ONLY Chrome headers (``x-browser-*`` + ``x-client-data``).

    Send only to a host for which :func:`is_google_property` is true.

    Args:
      major: Chrome major version; must match the impersonated UA.
      platform: The impersonated OS; selects the validation API key + UA.
      x_client_data: The opaque ``x-client-data`` install token; the default
        is a low-entropy value shared across many real installs.

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


def chrome_headers_for_host(
    host: str,
    *,
    major: int,
    platform: ChromePlatform,
) -> dict[str, str]:
    """Return the Google-only headers Chrome sends to ``host``; empty off Google.

    Args:
      host: Bare hostname of the request.
      major: Chrome major version; must match the impersonated UA.
      platform: The impersonated OS.

    Returns:
      headers: :func:`chrome_headers_for_google` on a Google property, else ``{}``.

    """
    if is_google_property(host):
        return chrome_headers_for_google(major=major, platform=platform)
    return {}


def _validation(user_agent: str, platform: ChromePlatform) -> str:
    """Compute ``x-browser-validation``: ``base64(sha1(api_key + UA))``."""
    # Per-platform Google API keys hard-coded in Chrome (see the module References).
    api_key = {
        "Windows": "AIzaSyA2KlwBX3mkFo30om9LUFYQhpqLoa_BNhE",
        "Linux": "AIzaSyBqJZh-7pA44blAaAkH6490hUFOwX0KCYM",
        "macOS": "AIzaSyDr2UxVnv_U85AbhhY8XSHSIavUW0DC-sY",
        "Android": "AIzaSyA8Hr8czk2cWyLu-a_RSPVYUEHozUnu6bA",
    }[platform]
    data = (api_key + user_agent).encode()
    return base64.b64encode(hashlib.sha1(data).digest()).decode()  # noqa: S324 -- Chrome compatibility requires this exact SHA-1 validation formula.
