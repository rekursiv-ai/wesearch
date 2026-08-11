"""curl_cffi transport for :mod:`wesearch.fetch`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast
from urllib.parse import urlparse

import threading

from wesearch.fetch.challenge import classify_http_error
from wesearch.fetch.common import (
    _REDIRECT_STATUSES,
    Observer,
    ValidatedHost,
    apply_redirect,
    bracket_ipv6,
    pinned_host,
    redirect_target,
)
from wesearch.types.errors import FetchError
from wesearch.types.params import Trust


if TYPE_CHECKING:
    from curl_cffi import requests as cc_requests
    from curl_cffi.requests import Response
    from curl_cffi.requests.impersonate import BrowserTypeLiteral
    from curl_cffi.requests.session import HttpMethod

    import curl_cffi
else:
    from wrapt import lazy_import

    curl_cffi = lazy_import("curl_cffi")


__all__ = [
    "close_curl_session",
    "close_curl_sessions_except",
    "curl_session",
    "fetch_curl",
    "seed_session_jar",
    "set_session_cookies",
]

# Pool identity: egress, registrable domain, impersonation target, SSRF pin,
# and the port that pin applies to. The pin participates because
# ``CurlOpt.RESOLVE`` is fixed when the Session is built.
_SessionKey: TypeAlias = tuple[  # noqa: UP040 -- forward ref in a type alias
    str, str, str, "ValidatedHost | None", int
]

# Live curl_cffi Sessions keyed by identity, so a session reuses one connection
# across requests -- the connection continuity a real browser has, and which a
# per-call request() (fresh TLS each time) lacks. Keyed on impersonate and on
# the SSRF pin, since both are fixed at construction.
# A live pool of open connections, not a tunable.
_curl_sessions: dict[_SessionKey, cc_requests.Session[Response]] = {}
# Guards every mutation of the live curl session pool.
_curl_lock = threading.Lock()


def curl_session(
    egress: str,
    domain: str,
    impersonate: str,
    *,
    pin: ValidatedHost | None = None,
    port: int = 443,
) -> cc_requests.Session[Response]:
    """Return the pooled curl_cffi Session for an identity, creating it once.

    Keyed on the REGISTRABLE domain (eTLD+1), not the exact host, so sibling
    subdomains of one site share a single connection + cookie jar -- the HTTP/2
    connection coalescing a real browser does for hosts on one certificate.
    ``www.google.com`` and ``scholar.google.com`` therefore reuse one session,
    so a warm-up GET to the apex carries its TLS handshake and Set-Cookie into a
    later request to the subdomain (a cold second connection is a bot tell that
    Scholar, in particular, budgets against).

    ``pin`` fixes the connect IP via ``CurlOpt.RESOLVE``, so a validated host
    cannot be re-resolved to a private address before the socket opens. It is
    passed as a Session-level ``curl_options`` entry rather than set on
    ``session.curl``: curl_cffi hands each thread its own handle by default, so
    a handle-level pin is invisible to the worker thread that performs the
    request (measured -- an off-thread request fails DNS outright), while
    ``use_thread_local_curl=False`` shares one handle and breaks under
    concurrency ("easy handle already used"). ``curl_options`` is applied per
    request, so it is the only form that is both thread- and concurrency-safe.

    Args:
      egress: Public egress IP the identity is keyed to.
      domain: Request hostname; coalesced to its registrable domain.
      impersonate: curl_cffi TLS-impersonation target.
      pin: Validated host/IP to pin the connection to, or ``None`` to let curl
        resolve. A pinned session is pooled separately from an unpinned one --
        the option cannot be changed on a live session without racing its
        concurrent users.
      port: Port the pin applies to; ``RESOLVE`` entries are per host:port.

    Returns:
      session: The pooled Session for this identity and pin.

    """
    key = (egress, _registrable_domain(domain), impersonate, pin, port)
    with _curl_lock:
        session = _curl_sessions.get(key)
        if session is None:
            options = (
                {
                    curl_cffi.CurlOpt.RESOLVE: [
                        f"{pin.host}:{port}:{bracket_ipv6(pin.ip)}"
                    ]
                }
                if pin is not None
                else {}
            )
            session = cast(
                "cc_requests.Session[Response]",
                curl_cffi.requests.Session(
                    impersonate=cast("BrowserTypeLiteral", impersonate),
                    curl_options=options,
                ),
            )
            _curl_sessions[key] = session
        return session


def seed_session_jar(
    session: cc_requests.Session[Response], domain: str, cookies: dict[str, str]
) -> None:
    """Load stored profile cookies into a curl session jar it does not yet hold.

    Cross-process persistence: the profile store outlives the in-memory session,
    so a fresh process seeds the jar from disk. Only names absent from the jar
    are added, so a live rotating cookie (curl tracking Scholar's NID/GSP) is
    never clobbered by a stale stored copy.
    """
    if not cookies:
        return
    # Cast keeps iteration typed under monorepo basedpyright; the export
    # disables unknown-member warnings and ty rejects the redundant cast, so
    # export ships the bare iteration.
    present = {c.name for c in session.cookies.jar}
    for name, value in cookies.items():
        if name not in present:  # never clobber a live jar cookie with a stale copy
            _jar_set(session, domain, name, value)


def set_session_cookies(
    session: cc_requests.Session[Response], domain: str, cookies: dict[str, str]
) -> None:
    """Set caller cookies into a curl session jar, OVERWRITING any prior value.

    Unlike :func:`seed_session_jar` (which preserves live jar cookies), a caller
    cookie is an explicit per-call override and must win, so it replaces a
    same-named jar entry. This keeps the jar the single cookie source on the curl
    path: sending the cookie via a header too would duplicate a name the jar
    already holds.
    """
    for name, value in cookies.items():
        _jar_set(session, domain, name, value)


def _jar_set(
    session: cc_requests.Session[Response], domain: str, name: str, value: str
) -> None:
    """Set one cookie in a curl jar, honoring RFC 6265bis name-prefix rules."""
    # RFC 6265bis 4.1.3 cookie-name prefixes, which curl_cffi enforces (and warns
    # + coerces when violated): a __Secure- cookie must be Secure; a __Host-
    # cookie must additionally be host-only (no Domain) with Path=/. Chrome only
    # ever sends these over https, so set them to match.
    if name.startswith("__Host-"):
        session.cookies.set(name, value, path="/", secure=True)
    elif name.startswith("__Secure-"):
        session.cookies.set(name, value, domain=domain, secure=True)
    else:
        session.cookies.set(name, value, domain=domain)


def _registrable_domain(host: str) -> str:
    """Return the eTLD+1 of a host (``a.b.example.co.uk`` -> ``example.co.uk``).

    A coarse public-suffix approximation: a two-label tail is the registrable
    domain, unless the last label is a 2-letter ccTLD and the second-to-last is
    a short (<=3-char) second-level label (``co.uk``, ``com.au``), in which case
    the tail is three labels. Sufficient for connection coalescing -- an
    over-broad grouping only shares a connection, never crosses a real origin
    boundary for cookies (those stay domain-scoped by the jar).
    """
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    tail = labels[-2:]
    if len(labels[-1]) == 2 and len(labels[-2]) <= 3:
        return ".".join(labels[-3:])
    return ".".join(tail)


def close_curl_session(egress: str, domain: str, impersonate: str) -> None:
    """Close and drop an identity's pooled Sessions (a burn ends the connection).

    Every pin and port for the identity is dropped, not one exact key: the pin
    participates in the pool key, so matching on a single tuple would leave the
    burned identity's other pinned sessions alive and the caller would keep
    presenting the cookies that just got it blocked.
    """
    prefix = (egress, _registrable_domain(domain), impersonate)
    with _curl_lock:
        sessions = [
            _curl_sessions.pop(key) for key in list(_curl_sessions) if key[:3] == prefix
        ]
    for session in sessions:
        session.close()  # I/O outside the lock; the pops already removed them.


def close_curl_sessions_except(egress: str | None) -> None:
    """Close pooled sessions that belong to a different egress."""
    with _curl_lock:
        sessions = [
            _curl_sessions.pop(key) for key in list(_curl_sessions) if key[0] != egress
        ]
    for session in sessions:
        session.close()


def _curl_set_cookies(resp: Response) -> list[str]:
    """Return the individual ``Set-Cookie`` headers of a curl response."""
    get_list = getattr(resp.headers, "get_list", None)
    if get_list is None:
        value = resp.headers.get("set-cookie")
        return [value] if value else []
    return list(cast("list[str]", get_list("set-cookie")))


@dataclass(slots=True, kw_only=True)
class _CurlLoop:
    """Mutable per-hop state shared by both curl backends' redirect loops.

    Holds the current URL, method, headers, body, and remaining redirect budget.
    :meth:`follow` runs the identical post-response decision both backends make:
    fire ``on_response``, and if the status is a followable redirect within
    budget, advance the state to the next hop (via :func:`apply_redirect`) and
    report ``True``. A ``False`` return means the response is terminal, leaving
    each backend to classify/return its (differently decompressed) body.
    """

    url: str
    method: str
    headers: dict[str, str]
    body: bytes | None
    remaining: int

    def follow(
        self,
        status: int,
        resp_headers: dict[str, str],
        *,
        on_response: Observer | None,
        on_redirect: Callable[[str], None] | None,
    ) -> bool:
        """Fire ``on_response``; advance to the next hop on a redirect within budget."""
        if on_response is not None:
            on_response(status, resp_headers, self.url)
        # A redirect is followed only while the budget allows; at 0 the contract
        # is "do not follow, return the 3xx body" (matching the stdlib path).
        if status not in _REDIRECT_STATUSES or self.remaining <= 0:
            return False
        self.remaining -= 1
        redirect_url = redirect_target(self.url, status, resp_headers)
        if on_redirect is not None:
            on_redirect(redirect_url)
        self.headers, self.method, self.body = apply_redirect(
            self.url,
            self.headers,
            self.method,
            body=self.body,
            status=status,
            redirect_url=redirect_url,
        )
        self.url = redirect_url
        return True


def fetch_curl(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_sec: float,
    connect_timeout_sec: float | None = None,
    max_redirects: int,
    impersonate: str,
    on_redirect: Callable[[str], None] | None,
    on_response: Observer | None,
    trust: Trust = "untrusted",
    session: cc_requests.Session[Response] | None = None,
    reseat: Callable[[str], cc_requests.Session[Response] | None] | None = None,
) -> bytes:
    """Perform a curl request, following redirects manually.

    ONE implementation for every request. SSRF pinning used to fork this into a
    second raw-handle path that took no Session -- so choosing pinning silently
    forfeited the pooled connection and its cookie jar, and the fork drifted
    from this one twice (headers, then cookies). Pinning is now an option on the
    pooled Session (see :func:`curl_session`), so there is nothing to diverge.

    Args:
      url: Fully-qualified URL.
      method: HTTP method.
      headers: Complete request headers, Cookie already merged.
      body: Encoded request body, or ``None``.
      timeout_sec: Per-request timeout.
      connect_timeout_sec: Ceiling on the handshake alone; ``None`` shares
        ``timeout_sec``.
      max_redirects: Redirect budget; at 0 the final 3xx body is returned.
      impersonate: curl_cffi TLS-impersonation target.
      on_redirect: Called with each redirect target before it is followed.
      on_response: Called with ``(status, headers, url)`` per hop.
      trust: Provenance of the URL. Under ``"untrusted"`` a one-shot request
        (no pooled ``session``) validates the host to a public address before
        connecting; a pooled session was already validated and pinned when the
        pool built it.
      session: Pooled Session to reuse, or ``None`` for a one-shot request.
      reseat: Called with the next hop's URL when a redirect crosses hosts,
        returning the Session for that host. A pin is fixed per Session, so a
        cross-host hop needs the pool entry for the NEW host; ``None`` keeps the
        current session for every hop.

    Returns:
      content: The decoded response body.

    Raises:
      FetchError: On a non-success status, or a transport failure (status 0).

    """
    # requests auto-decompresses .content, so no decompress call is needed.
    # Cookies are already in headers["Cookie"], so NO cookies= kwarg is passed
    # (curl would emit a second Cookie source -- verified both are sent).
    loop = _CurlLoop(
        url=url, method=method, headers=headers, body=body, remaining=max_redirects
    )
    impers = cast("BrowserTypeLiteral", impersonate)
    # curl_cffi reads a (connect, read) pair; a bare float budgets both together.
    timeout = (
        timeout_sec
        if connect_timeout_sec is None
        else (connect_timeout_sec, timeout_sec)
    )
    while True:
        # A keyless request has no pooled session to have been pinned at build
        # time, so it validates here. Validation is per hop: a redirect can
        # point at a private address, which is the classic SSRF bypass.
        if session is None:
            pinned_host(loop.url, trust)
        try:
            verb = cast("HttpMethod", loop.method)  # curl types verb as a Literal.
            resp = (
                session.request(  # pyright: ignore[reportUnknownMemberType] -- curl_cffi's **Unpack[RequestParams] TypedDict is unstubbed
                    verb,
                    loop.url,
                    headers=loop.headers,
                    data=loop.body,
                    impersonate=impers,
                    timeout=timeout,
                    allow_redirects=False,
                )
                if session is not None
                else curl_cffi.requests.request(  # pyright: ignore[reportUnknownMemberType] -- curl_cffi's **Unpack[RequestParams] TypedDict is unstubbed
                    verb,
                    loop.url,
                    headers=loop.headers,
                    data=loop.body,
                    impersonate=impers,
                    timeout=timeout,
                    allow_redirects=False,
                )
            )
        except curl_cffi.CurlError as e:
            raise FetchError(loop.url, 0, {}, str(e).encode()) from e
        # curl_cffi's request/Session.request type a None return for the
        # thread/stream overloads; the sync call here always yields a Response.
        assert resp is not None
        status = int(resp.status_code)
        resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
        # curl_cffi's Headers.items() lossily folds duplicate Set-Cookie with
        # ", " (and Set-Cookie values may contain ", "); get_list preserves the
        # individual headers, newline-joined to match join_headers so
        # parse_set_cookie splits them back exactly.
        cookies_list = _curl_set_cookies(resp)
        if cookies_list:
            resp_headers["set-cookie"] = "\n".join(cookies_list)
        content = bytes(resp.content or b"")
        current_url = loop.url
        if loop.follow(
            status, resp_headers, on_response=on_response, on_redirect=on_redirect
        ):
            # A pin is fixed per Session, so a hop onto a different host must
            # move to that host's pool entry -- otherwise the new host would be
            # fetched through the previous host's pin and fail to resolve.
            if (
                reseat is not None
                and urlparse(loop.url).hostname != urlparse(current_url).hostname
            ):
                session = reseat(loop.url)
            continue
        if status >= 400:
            raise classify_http_error(current_url, status, resp_headers, content)
        return content
