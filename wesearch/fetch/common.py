"""HTTP helpers the fetch transports share."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse

import gzip
import ipaddress
import socket
import zlib

import brotli

from wesearch.chrome.headers import chrome_client_hints
from wesearch.fetch.types import ValidatedHost
from wesearch.lib import zstd_compat
from wesearch.profile import parse_set_cookie
from wesearch.types.errors import FetchError


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


__all__ = [
    "DEFAULT_PORTS",
    "ORIGIN_BOUND",
    "PendingRequest",
    "bracket_ipv6",
    "cookie_header",
    "decompress",
    "decompress_error_body",
    "host_header",
    "join_headers",
    "merge_headers",
    "origin",
    "pinned_host",
    "public_host",
    "redirect_target",
]

DEFAULT_PORTS: Mapping[str, int] = MappingProxyType({"http": 80, "https": 443})

# Headers a hop to another origin drops: they belong to the origin that set them.
ORIGIN_BOUND: frozenset[str] = frozenset(
    {"cookie", "authorization"}
    | {name.lower() for name in chrome_client_hints(major=1)},
)


def public_host(hostname: str) -> ValidatedHost:
    """Resolve ``hostname``; refuse it unless every address it has is public.

    Args:
      hostname: Host name or IP literal, without a port.

    Returns:
      validated: The host and the one public IP to connect to.

    Raises:
      ValueError: When the host does not resolve or any address is not public.

    """
    host = urlparse(f"//{hostname}").hostname
    if host is None:
        raise ValueError("URL has no host.")
    try:
        ips = [str(info[4][0]) for info in socket.getaddrinfo(host, None)]
    except (socket.gaierror, UnicodeError) as e:
        raise ValueError(f"DNS resolution failed for {host!r}: {e}") from e
    for candidate in ips:
        address = ipaddress.ip_address(candidate)
        if (
            address.is_loopback
            or address.is_link_local
            or address.is_private
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise ValueError(
                f"Refusing to fetch {host!r} (resolves to non-public address "
                f"{address}).",
            )
    if not ips:
        raise ValueError(f"DNS resolution returned no address for {host!r}.")
    # IPv4 first: a pinned IPv6 address on a network with no v6 route fails,
    # where an unpinned client would have fallen back to v4.
    return ValidatedHost(host=host, ip=next((a for a in ips if ":" not in a), ips[0]))


def pinned_host(url: str, *, allow_private_hosts: bool) -> ValidatedHost | None:
    """Return the public IP to connect to for ``url``, or ``None`` when allowed any.

    Args:
      url: The URL about to be requested.
      allow_private_hosts: Skip the public-address check.

    Returns:
      pin: The host and the public IP its connection must use.

    Raises:
      ValueError: When the scheme is not ``http`` or ``https``, or the host is
        missing, unresolvable, or not public.

    """
    parsed = urlparse(url)
    if parsed.scheme not in DEFAULT_PORTS:
        raise ValueError(f"Unsupported URL scheme {parsed.scheme!r}: {url!r}.")
    if allow_private_hosts:
        return None
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError(f"URL has no host: {url!r}.")
    return public_host(hostname)


def merge_headers(
    defaults: Mapping[str, str],
    overrides: Mapping[str, str | None],
) -> dict[str, str]:
    """Return ``defaults`` with ``overrides`` set over them, names in any case.

    Each name is sent once. A replaced header keeps its default's position on
    the wire; a new one goes last; of names differing only in case, the last
    override wins.

    Args:
      defaults: Headers in wire order.
      overrides: Headers that win; a ``None`` value removes that header.

    Returns:
      merged: The headers to send.

    """
    replaced = {name.lower(): (name, value) for name, value in overrides.items()}
    merged = {
        name: replaced.pop(name.lower(), (name, value))[1]
        for name, value in defaults.items()
    }
    merged.update(replaced.values())
    return {name: value for name, value in merged.items() if value is not None}


def origin(url: str) -> str:
    """Return ``url``'s origin, ``scheme://host[:port]``, lower-case, no default port.

    ``https://User:pw@EXAMPLE.com:443/a`` and ``https://example.com/b`` have the
    same origin.

    Args:
      url: Any URL.

    Returns:
      origin: The canonical origin.

    """
    parsed = urlparse(url)
    host = bracket_ipv6(parsed.hostname or "")
    return f"{parsed.scheme}://{host_header(host, parsed.port, parsed.scheme)}"


def decompress(body: bytes, encoding: str) -> bytes:
    """Return ``body`` decoded per its ``Content-Encoding``.

    Args:
      body: Response body.
      encoding: ``Content-Encoding`` value, e.g. ``gzip`` or ``gzip, br``.

    Returns:
      body: The decoded body.

    Raises:
      ValueError: On an unknown encoding or a corrupt body.

    """
    for enc in reversed([tok.strip().lower() for tok in encoding.split(",")]):
        if enc not in ("", "identity", "gzip", "deflate", "br", "zstd"):
            raise ValueError(f"Unknown Content-Encoding: {enc!r}")
        if enc == "deflate":
            try:
                body = zlib.decompress(body)
                continue
            except zlib.error:
                pass  # Some servers send raw deflate, no zlib header; browsers accept it.
        try:
            if enc == "gzip":
                body = gzip.decompress(body)
            elif enc == "deflate":
                body = zlib.decompress(body, -zlib.MAX_WBITS)
            elif enc == "br":
                body = brotli.decompress(body)
            elif enc == "zstd":
                body = zstd_compat.decompress(body)
        except (OSError, ValueError, zlib.error, brotli.error) as e:
            raise ValueError(f"Decompression failed ({enc}): {e}") from None
    return body


def decompress_error_body(body: bytes, headers: dict[str, str]) -> bytes:
    """Return an error response's body decoded, or as is if it will not decode.

    Args:
      body: Response body.
      headers: Response headers, names lower-case.

    Returns:
      body: The decoded body, or ``body`` unchanged.

    """
    encoding = headers.get("content-encoding") or ""
    try:
        return decompress(body, encoding)
    except ValueError:
        return body


def join_headers(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Return response headers as a dict, names lower-case.

    A repeated header joins with ``", "``; a repeated ``set-cookie`` joins with
    a newline, since cookie values contain ``", "``.

    Args:
      pairs: ``(name, value)`` pairs as received.

    Returns:
      headers: One value per name.

    """
    out: dict[str, str] = {}
    for k, v in pairs:
        key = k.lower()
        if key not in out:
            out[key] = v
        elif key == "set-cookie":
            out[key] = f"{out[key]}\n{v}"
        else:
            out[key] = f"{out[key]}, {v}"
    return out


def redirect_target(current_url: str, status: int, headers: dict[str, str]) -> str:
    """Return the absolute URL a redirect's ``Location`` points to.

    Args:
      current_url: The URL that answered with the redirect.
      status: Its status.
      headers: Its headers, names lower-case.

    Returns:
      target: The redirect target.

    Raises:
      FetchError: When there is no ``Location``.

    """
    location = headers.get("location", "")
    if not location:
        raise FetchError(
            current_url,
            status,
            headers,
            b"Redirect with no Location header",
        )
    return urljoin(current_url, location)


def host_header(host: str, port: int | None, scheme: str) -> str:
    """Return the ``Host`` header value: the host, plus a non-default port."""
    if port is None or port == DEFAULT_PORTS.get(scheme):
        return host
    return f"{host}:{port}"


def bracket_ipv6(host: str) -> str:
    """Return an IPv6 literal in brackets; return other hosts unchanged."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


@dataclass(slots=True, kw_only=True)
class PendingRequest:
    """The request to send next; :meth:`follow` turns it into a redirect's target.

    Attributes:
      url: Where the request goes.
      method: HTTP method.
      headers: Headers to send.
      body: Request body.
      remaining: Redirects still allowed.

    """

    url: str
    method: str
    headers: dict[str, str]
    body: bytes
    remaining: int

    def follow(
        self,
        status: int,
        headers: dict[str, str],
        *,
        on_response: Callable[[int, dict[str, str], str], None],
        on_redirect: Callable[[str], None],
        cookies_for_host: Callable[[str], Mapping[str, str]],
    ) -> bool:
        """Report a response; on a redirect within budget, become the next request.

        The next request is what Chrome sends: ``Origin`` names the new target, a
        301-303 of a non-GET becomes a bodyless GET, cookies the response set on
        this host are sent, and a hop to another origin drops
        :data:`ORIGIN_BOUND` headers and sends that origin's stored cookies.

        Args:
          status: Response status.
          headers: Response headers, names lower-case.
          on_response: Called with ``(status, headers, url)``.
          on_redirect: Called with the redirect target; raise to stop.
          cookies_for_host: Returns the stored cookies for a new origin's host.

        Returns:
          followed: Whether this is now the redirect's request, to send next.

        """
        on_response(status, headers, self.url)
        if status not in (301, 302, 303, 307, 308) or self.remaining <= 0:
            return False
        self.remaining -= 1
        target = redirect_target(self.url, status, headers)
        on_redirect(target)
        sent = {k: v for k, v in self.headers.items() if k.lower() != "origin"}
        if len(sent) < len(self.headers):
            sent["Origin"] = origin(target)
        if status in (301, 302, 303) and self.method != "GET":
            self.method = "GET"
            self.body = b""
            sent = {k: v for k, v in sent.items() if k.lower() != "content-type"}
        if origin(self.url) != origin(target):
            sent = {k: v for k, v in sent.items() if k.lower() not in ORIGIN_BOUND}
            if stored := cookies_for_host(urlparse(target).hostname or ""):
                sent["Cookie"] = cookie_header({}, stored)
        elif set_cookie := headers.get("set-cookie", ""):
            sent = merge_headers(
                sent,
                {"Cookie": cookie_header(sent, parse_set_cookie(set_cookie))},
            )
        self.headers = sent
        self.url = target
        return True


def cookie_header(headers: Mapping[str, str], cookies: Mapping[str, str]) -> str:
    """Return the ``Cookie`` value of ``headers`` with ``cookies`` set over it.

    Args:
      headers: Request headers, any case.
      cookies: Cookie names and values that win.

    Returns:
      cookie: The merged ``Cookie`` header value.

    """
    sent = [v for k, v in headers.items() if k.lower() == "cookie"][-1:]
    pairs = (pair.strip().partition("=") for v in sent for pair in v.split(";"))
    merged = {name: value for name, eq, value in pairs if eq and name} | dict(cookies)
    return "; ".join(f"{name}={value}" for name, value in merged.items())
