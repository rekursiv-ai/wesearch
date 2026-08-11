"""Capture the exact request a real Chrome sends, for parity testing.

Drives a headless Chrome (via the DevTools ``--log-net-log`` capture) to a URL
and extracts the request headers it put on the wire -- names, values, and ORDER
-- so a test can assert :mod:`wesearch.fetch` sends the same. Chrome is the
oracle: rather than guess what a browser sends, we read it.

Requires a ``google-chrome`` / ``google-chrome-stable`` binary; callers gate on
:func:`chrome_available` and skip when absent (Chrome is not a hard dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import json
import shutil
import subprocess
import tempfile


__all__ = [
    "CapturedRequest",
    "capture_chrome_request",
    "chrome_available",
]


class _NetLog(TypedDict):
    """The subset of a Chrome net-log JSON this module reads."""

    constants: dict[str, dict[str, int]]
    events: list[dict[str, Any]]


@dataclass(frozen=True, slots=True, kw_only=True)
class CapturedRequest:
    """One request Chrome sent, as observed on the wire.

    Attributes:
      url: The request URL.
      method: The HTTP method.
      headers: Ordered ``(name, value)`` pairs, pseudo-headers (``:method`` ...)
        excluded, exactly as Chrome ordered them in the HEADERS frame.

    """

    url: str
    method: str
    headers: tuple[tuple[str, str], ...]

    def header_names(self) -> tuple[str, ...]:
        """Return the lower-cased header names in wire order."""
        return tuple(name.lower() for name, _ in self.headers)

    def header(self, name: str) -> str | None:
        """Return the value of ``name`` (case-insensitive), or ``None``."""
        lower = name.lower()
        return next((v for n, v in self.headers if n.lower() == lower), None)


def chrome_available() -> bool:
    """Whether a Chrome binary is on ``PATH``."""
    return _chrome_binary() is not None


def capture_chrome_request(
    url: str,
    *,
    to_origin: str = "",
    timeout_sec: float = 40.0,
    ignore_certificate_errors: bool = False,
) -> list[CapturedRequest]:
    """Load ``url`` in headless Chrome; return the requests it sent to an origin.

    Args:
      url: The URL to navigate to.
      to_origin: Keep only requests whose URL starts with this ``scheme://host``
        prefix; empty keeps every request (useful to see sub-resources).
      timeout_sec: Hard cap on the Chrome invocation.
      ignore_certificate_errors: Accept an untrusted TLS certificate. Required
        only to reach a loopback oracle serving a self-signed cert; never enable
        against a real host.

    Returns:
      requests: The captured requests in the order Chrome issued them.

    Raises:
      RuntimeError: When no Chrome binary is available.

    """
    binary = _chrome_binary()
    if binary is None:
        raise RuntimeError("No Chrome binary found on PATH.")
    with tempfile.TemporaryDirectory(prefix="chrome-capture-") as profile:
        netlog = Path(profile) / "netlog.json"
        flags = ["--ignore-certificate-errors"] if ignore_certificate_errors else []
        subprocess.run(  # noqa: S603 -- fixed argv, binary resolved from PATH.
            [
                binary,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--incognito",
                f"--user-data-dir={profile}",
                f"--log-net-log={netlog}",
                *flags,
                "--dump-dom",
                url,
            ],
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        return _parse_netlog(netlog, to_origin=to_origin)


def _chrome_binary() -> str | None:
    """The first available Chrome binary name, or ``None``."""
    for name in ("google-chrome-stable", "google-chrome", "chromium", "chrome"):
        if shutil.which(name) is not None:
            return name
    return None


def _parse_netlog(netlog: Path, *, to_origin: str) -> list[CapturedRequest]:
    """Extract sent-request headers from a Chrome net-log, in issue order.

    Chrome logs the ordered wire headers under a protocol-specific event: the
    HTTP/2 event carries the ``:method`` / ``:scheme`` / ``:authority`` /
    ``:path`` pseudo-headers, while the HTTP/1.1 event carries a ``line`` request
    line plus a ``Host`` header. Both reconstruct the URL a parity check needs;
    an origin served only over HTTP/1.1 (e.g. a loopback oracle) would be invisible
    if only the HTTP/2 event were read.
    """
    data = _load_lenient(netlog.read_bytes().decode("utf-8", "replace"))
    event_types = data["constants"]["logEventTypes"]
    wanted = {
        code
        for name in (
            "HTTP_TRANSACTION_HTTP2_SEND_REQUEST_HEADERS",
            "HTTP_TRANSACTION_SEND_REQUEST_HEADERS",
        )
        if (code := event_types.get(name)) is not None
    }
    if not wanted:
        # Both constant names drifted (Chrome renamed the events). Fail loud
        # rather than silently match nothing and return [].
        raise RuntimeError(
            "net-log has no known SEND_REQUEST_HEADERS event type;"
            " Chrome may have renamed it."
        )
    out: list[CapturedRequest] = []
    for event in data["events"]:
        if event.get("type") not in wanted:
            continue
        request = _request_from_event(event.get("params", {}))
        if request is not None and request.url.startswith(to_origin):
            out.append(request)
    return out


def _load_lenient(raw: str) -> _NetLog:
    """Parse a net-log JSON, repairing a truncated tail (Chrome killed mid-write).

    A body killed after a complete event ends with ``"},"``; closing the array +
    object recovers the completed events. When there is no such boundary the
    body is unrepairable -- re-raise the ORIGINAL decode error (not a confusing
    one about the fabricated ``"]}"``), so the real failure is visible.
    """
    try:
        return cast("_NetLog", json.loads(raw))
    except json.JSONDecodeError:
        cut = raw.rfind("},")
        if cut == -1:
            raise
        return cast("_NetLog", json.loads(raw[: cut + 1] + "]}"))


def _request_from_event(params: dict[str, Any]) -> CapturedRequest | None:
    """Build a :class:`CapturedRequest` from an HTTP/2 or HTTP/1.1 send event.

    HTTP/2 encodes the URL in ``:scheme`` / ``:authority`` / ``:path``
    pseudo-headers; HTTP/1.1 encodes it in a ``line`` request line plus a
    ``Host`` header (there are no pseudo-headers). One helper handles both so a
    net-log from either protocol yields the same ordered record.
    """
    raw_headers = params.get("headers")
    if not isinstance(raw_headers, list):
        return None
    pseudo: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    for header in cast("list[object]", raw_headers):
        if not isinstance(header, str) or ": " not in header:
            continue
        name, value = header.split(": ", 1)
        if name.startswith(":"):
            pseudo[name] = value
        else:
            pairs.append((name, value))
    if pseudo:
        return CapturedRequest(
            url=f"{pseudo.get(':scheme', 'https')}://{pseudo.get(':authority', '')}"
            f"{pseudo.get(':path', '/')}",
            method=pseudo.get(":method", "GET"),
            headers=tuple(pairs),
        )
    return _http1_request(cast("str", params.get("line", "")), pairs)


def _http1_request(line: str, pairs: list[tuple[str, str]]) -> CapturedRequest | None:
    """Reconstruct an HTTP/1.1 record from its request line and header pairs.

    The request line is ``METHOD path HTTP/1.1`` and the authority is the
    ``Host`` header; together they form the absolute URL the HTTP/2 path derives
    from pseudo-headers.
    """
    parts = line.split(" ", 2)
    if len(parts) < 2:
        return None
    host = next((v for name, v in pairs if name.lower() == "host"), "")
    return CapturedRequest(
        url=f"https://{host}{parts[1]}",
        method=parts[0],
        headers=tuple(pairs),
    )
