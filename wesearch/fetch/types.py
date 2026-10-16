"""Types shared by :mod:`wesearch.fetch` and its transports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias

import threading

from wesearch.lib.userdirs import data_dir
from wesearch.profile import ProfileStore
from wesearch.types.params import ignore, no_cookies


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from wesearch.types.params import HttpMethod


__all__ = [
    "Closable",
    "CurlKey",
    "Fetch",
    "FetchSession",
    "Shutdownable",
    "ValidatedHost",
]


# (egress IP, hostname, pinned host, port).
CurlKey: TypeAlias = tuple[str, str, "ValidatedHost | None", int]  # noqa: UP040 -- The forward reference keeps the alias resolvable at runtime.


class Closable(Protocol):
    """A connection :meth:`FetchSession.close` closes."""

    def close(self) -> None:
        """Close the connection."""


class Shutdownable(Protocol):
    """A connection pool :meth:`FetchSession.close` shuts down."""

    def shutdown(self) -> None:
        """Close every connection in the pool."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedHost:
    """A hostname and the public IP it resolved to."""

    host: str
    ip: str


@dataclass(slots=True, kw_only=True)
class FetchSession:
    """Connections, cookies, and the egress IP that fetches share.

    Pass one session to every fetch that should reuse connections and the
    browser; a fetch given none uses the process's shared one (see
    :func:`wesearch.fetch.session.default_session`). Thread-safe. Call
    :meth:`close` when done with a session you made.

    Attributes:
      profiles: Where cookies persist, per egress IP and host, across processes.
      profile_dir: Chrome profile the browser transport uses.
      headless: Run that Chrome without a window.
      egress: This host's public IP once probed; ``""`` before.
      egress_retry_at: Monotonic time before which a failed probe is not
        repeated; ``0.0`` when none failed.
      curl: Open curl_cffi Sessions, one per egress, site, pinned IP, and port.
      browsers: Connections to Chrome, opened by the first browser fetch.
      tokens: Bearer tokens by service, with their monotonic expiry.
      lock: Guards ``egress``, ``curl``, ``browsers``, and ``tokens``.

    """

    profiles: ProfileStore = field(default_factory=ProfileStore)
    profile_dir: Path = field(
        default_factory=lambda: (
            data_dir() / "rekursiv-ai" / "wesearch" / "fetch-zendriver"
        ),
    )
    headless: bool = True
    egress: str = ""
    egress_retry_at: float = 0.0
    curl: dict[CurlKey, Closable] = field(default_factory=dict)
    browsers: Shutdownable | None = None
    tokens: dict[str, tuple[str, float]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        """Close every connection; a Chrome stays up for other processes."""
        with self.lock:
            sessions = list(self.curl.values())
            self.curl.clear()
            browsers, self.browsers = self.browsers, None
        for curl_session in sessions:
            curl_session.close()
        if browsers is not None:
            browsers.shutdown()


class Fetch(Protocol):
    """A transport implementing the common HTTP request contract.

    Implemented by :func:`wesearch.fetch.transport.curl.fetch_curl`,
    :func:`wesearch.fetch.transport.stdlib.fetch_stdlib`, and
    :func:`wesearch.fetch.transport.zendriver.fetch_zendriver`.
    """

    def __call__(
        self,
        url: str,
        *,
        method: HttpMethod = "GET",
        headers: Mapping[str, str | None] = MappingProxyType({}),
        body: bytes = b"",
        timeout_sec: float = 30.0,
        connect_timeout_sec: float = 10.0,
        max_redirects: int = 10,
        on_redirect: Callable[[str], None] = ignore,
        on_response: Callable[[int, dict[str, str], str], None] = ignore,
        allow_private_hosts: bool = False,
        session: FetchSession | None = None,
        cookies_for_host: Callable[[str], Mapping[str, str]] = no_cookies,
    ) -> bytes:
        """Fetch ``url``, following redirects; return the response body.

        Args:
          url: Fully-qualified URL.
          method: HTTP method.
          headers: Sent over the transport's own Chrome headers, names in any
            case; a ``None`` value removes that header. A ``Cookie`` header is
            the request's cookies.
          body: Request body.
          timeout_sec: Seconds allowed for the whole request.
          connect_timeout_sec: Seconds allowed to connect.
          max_redirects: Redirects to follow; at 0 the redirect is returned.
          on_redirect: Called with each redirect target; raise to stop.
          on_response: Called with ``(status, headers, url)`` for every
            response; ``headers`` names are lower-case.
          allow_private_hosts: Allow loopback, private, and link-local
            addresses; leave ``False`` for any URL you did not write.
          session: Connections to reuse; ``None`` uses the process's shared session.
          cookies_for_host: Returns the cookies to send a host a redirect moves
            to; the caller's ``Cookie`` stays with the first origin.

        Returns:
          body: The response body, decompressed.

        """
        ...
