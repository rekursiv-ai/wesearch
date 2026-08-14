"""Unified HTTP fetch for ``wesearch``.

All HTTP in ``wesearch`` flows through :func:`fetch`. Sync, with
transparent decompression and optional retry. The transport is curl_cffi
(a Chrome-compatible TLS/HTTP-2 profile). A stdlib ``http.client`` path is
retained as a reference implementation for validating the curl path, not as a
runtime fallback -- curl_cffi is a hard dependency. Every call returns
``(body, session)``; the returned :class:`FetchSession` is a browsing identity
you thread into the next call.

Usage::

    from wesearch.fetch import fetch

    # Simple (99% case): ignore the returned session.
    body, _ = fetch(url)
    html = fetch(url)[0].decode("utf-8")

    # Repeated calls: thread the session so each request builds on the last
    # (cookies set, Accept-CH opt-ins), the way a real browser's do.
    body, session = fetch(url)
    body, session = fetch(next_url, session=session)
    body, session = fetch(next_url, session=session)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast
from urllib.parse import unquote, urlencode, urlparse, urlsplit, urlunsplit

import base64
import functools
import ipaddress
import json as json_lib
import logging
import threading
import time

from wesearch.chrome.headers import (
    chrome_client_hints,
    chrome_headers_for_google,
    chrome_navigation_headers,
    impersonate_version_platform,
    is_google_property,
)
from wesearch.chrome.useragents import draw_user_agent, kind_for_impersonate
from wesearch.fetch.common import (
    Observer,
    default_port,
    origin,
    pinned_host,
)
from wesearch.fetch.transport import transport_routing
from wesearch.fetch.transport.curl import (
    close_curl_session,
    close_curl_sessions_except,
    curl_session,
    fetch_curl,
    seed_session_jar,
    set_session_cookies,
)
from wesearch.fetch.transport.stdlib import fetch_stdlib
from wesearch.lib.userdirs import data_dir
from wesearch.profile import Profile, ProfileStore, parse_set_cookie
from wesearch.types.errors import BotDetectionError, FetchError
from wesearch.types.params import (
    NO_BODY,
    ContentParams,
    RequestParams,
    RetryParams,
    Transport,
)


if TYPE_CHECKING:
    from curl_cffi import requests as cc_requests
    from curl_cffi.requests import Response
    from curl_cffi.requests.session import HttpMethod

    import wesearch.fetch.transport.zendriver as zendriver_backend
else:
    from wrapt import lazy_import

    zendriver_backend = lazy_import("wesearch.fetch.transport.zendriver")


__all__ = [
    "FetchSession",
    "egress_ip",
    "fetch",
    "last_known_egress_ip",
    "on_egress_rotation",
    "resolve_transport",
    "set_last_egress_ip",
]

logger = logging.getLogger(__name__)


def resolve_transport(
    transport: Transport,
    *,
    method: HttpMethod = "GET",
    raw_headers: bool = False,
    has_body: bool = False,
) -> Transport:
    """Resolve ``auto`` to a concrete transport for this request."""
    if transport != "auto":
        return transport
    if method != "GET" or raw_headers or has_body:
        return "curl"
    return "curl-then-zendriver"


_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True, kw_only=True)
class FetchSession:
    """Frozen browsing state threaded through :func:`fetch` calls.

    Returned by every :func:`fetch` call carrying what the response set; pass it
    back as the ``session`` argument of the next call. Serializable via
    :meth:`serialize` / :meth:`deserialize`.

    Attributes:
      impersonate: The curl_cffi TLS-impersonation target; the User-Agent and
        client hints are derived from it.
      cookies: The cookie jar, keyed by origin (``scheme://host``) then cookie
        name, and updated from ``Set-Cookie``. Scoped like
        :class:`wesearch.profile.ProfileStore`, which keys by
        ``(egress_ip, domain)``: a flat ``name -> value`` jar sent a cookie
        ``a.example`` set to ``b.example`` on the next call, which no browser
        does and which leaks a session identifier to an unrelated host.

        Origin is the ONLY scope kept. ``Path``, ``Domain``, and ``SameSite``
        are dropped by :func:`wesearch.profile.parse_set_cookie` -- we are
        a client deciding what to send back, not a browser enforcing a security
        boundary, and over-sending within one origin costs a header rather than
        an identity. The prefixes where that is NOT true (``__Host-``,
        ``__Secure-``) are enforced in the curl jar's ``_jar_set``, because RFC
        6265bis makes them a correctness requirement rather than a preference.
      accept_ch: Per-origin (``scheme://host``) sets of extended client-hint
        header names the origin requested via ``Accept-CH``.

    """

    impersonate: str = "chrome"
    cookies: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict[str, Mapping[str, str]]
    )
    accept_ch: Mapping[str, frozenset[str]] = field(
        default_factory=dict[str, frozenset[str]]
    )

    def cookies_for(self, url: str) -> dict[str, str]:
        """Return the cookies this session may send to ``url``'s origin."""
        return dict(self.cookies.get(origin(url), {}))

    def with_cookies(self, url: str, updates: Mapping[str, str]) -> FetchSession:
        """Return a copy whose jar for ``url``'s origin is merged with ``updates``."""
        if not updates:
            return self
        key = origin(url)
        merged = {**self.cookies.get(key, {}), **updates}
        return replace(self, cookies={**self.cookies, key: merged})

    def with_accept_ch(self, origin: str, hints: frozenset[str]) -> FetchSession:
        """Return a copy recording ``origin``'s ``hints`` (unchanged if same)."""
        if not hints or self.accept_ch.get(origin) == hints:
            return self
        return replace(self, accept_ch={**self.accept_ch, origin: hints})

    def serialize(self) -> dict[str, object]:
        """Return a JSON-serializable dict of this session's state.

        Returns:
          data: A dict with ``impersonate``, ``cookies``, and ``accept_ch``. Each hint set is emitted as a sorted list purely for a
            deterministic serialized form; the hints are an unordered set (the
            outgoing request emits them in Chrome's own client-hint order, not
            this one), and :meth:`deserialize` rebuilds a ``frozenset``.

        """
        return {
            "impersonate": self.impersonate,
            "cookies": {org: dict(jar) for org, jar in self.cookies.items()},
            "accept_ch": {
                origin: sorted(hints) for origin, hints in self.accept_ch.items()
            },
        }

    @classmethod
    def deserialize(cls, data: Mapping[str, object]) -> FetchSession:
        """Reconstruct a session from :meth:`serialize` output.

        Args:
          data: A dict as produced by :meth:`serialize`.

        Returns:
          session: The reconstructed :class:`FetchSession`.

        """
        accept_ch = cast("Mapping[str, list[str]]", data.get("accept_ch", {}))
        jars = cast("Mapping[str, Mapping[str, str]]", data.get("cookies", {}))
        return cls(
            impersonate=cast("str", data.get("impersonate", "chrome")),
            cookies={org: dict(jar) for org, jar in jars.items()},
            accept_ch={origin: frozenset(hints) for origin, hints in accept_ch.items()},
        )


def fetch(
    url: str,
    *,
    session: FetchSession | None = None,
    request: RequestParams | None = None,
) -> tuple[bytes, FetchSession]:
    """Fetch a URL; return the body and the updated browsing session.

    Args:
      url: Fully-qualified URL (http or https).
      session: A prior returned session to reuse, or ``None`` for a fresh one.
      request: Per-call parameters, or ``None`` for the defaults.

    Returns:
      body: The response bytes.
      session: A :class:`FetchSession` carrying what the response set; pass it to
        the next call.

    Raises:
      FetchError: On non-success HTTP status after exhausting retries.
      ValueError: On unsupported or corrupt Content-Encoding.

    """
    return _Request(
        url=url,
        session=FetchSession() if session is None else session,
        params=RequestParams() if request is None else request,
    ).fetch()


class _ResponseLearner:
    """Accumulates what responses teach a session, then folds it in.

    Its :meth:`observe` is the ``on_response`` sink for a :func:`fetch`
    call: it records ``Set-Cookie`` and ``Accept-CH`` from every response (each
    redirect hop and the final one), passing the status/headers through to the
    caller's own callback. :meth:`merge_into` returns the session updated with
    everything observed.
    """

    def __init__(
        self, *, url: str, caller: Callable[[int, dict[str, str]], None] | None
    ) -> None:
        self._url = url
        self._caller = caller
        self._cookies: dict[str, dict[str, str]] = {}
        self._accept_ch: dict[str, frozenset[str]] = {}

    def observe(self, status: int, resp_headers: dict[str, str], url: str) -> None:
        """Record cookies + Accept-CH from a hop; forward to the caller.

        Cookies are filed under the RESPONDING origin, so a redirect target's
        ``Set-Cookie`` warms that origin's jar rather than this request's -- a
        browser following ``a -> b`` learns b's cookies, and the per-origin jar
        can now express that. They were discarded outright while the jar was
        flat, because the only alternative then was mis-attributing them to the
        requesting origin. Accept-CH opt-ins are keyed the same way.
        """
        set_cookie = resp_headers.get("set-cookie")
        if set_cookie:
            self._cookies.setdefault(origin(url), {}).update(
                parse_set_cookie(set_cookie)
            )
        hints = _accept_ch_hints(resp_headers)
        if hints:
            self._accept_ch[origin(url)] = hints
        if self._caller is not None:
            self._caller(status, resp_headers)

    def merge_into(self, session: FetchSession) -> FetchSession:
        """Return ``session`` updated with the observed cookies + opt-ins."""
        updated = session
        for org, jar in self._cookies.items():
            updated = updated.with_cookies(org, jar)
        for org, hints in self._accept_ch.items():
            updated = updated.with_accept_ch(org, hints)
        return updated


@dataclass(frozen=True, slots=True, kw_only=True)
class _Request:
    """One :func:`fetch` invocation: its URL, session, and per-call params.

    ``observer`` is the internal per-hop sink (:class:`_ResponseLearner`, which
    also forwards to the caller's public 2-arg ``params.on_response``). It is
    kept OFF ``params`` so the public 2-arg contract and this 3-arg internal
    contract don't conflate.
    """

    url: str
    session: FetchSession
    params: RequestParams
    observer: Observer | None = None

    @property
    def domain(self) -> str:
        """The hostname the session identity is keyed on (``""`` if hostless)."""
        return urlparse(self.url).hostname or ""

    def fetch(self) -> tuple[bytes, FetchSession]:
        """Perform the request; return the body and the updated session.

        ``trust`` deliberately takes no part in transport resolution. It used to:
        an SSRF-validated request was forced onto plain curl, which skipped the
        learned-domain routing below AND made every browser transport an error,
        so the one caller that asked for safety lost the browser entirely. Trust
        is now honored at connect time by whichever transport runs.
        """
        p = self.params
        resolved = resolve_transport(
            p.policy.transport,
            method=p.content.method,
            raw_headers=p.content.raw_headers,
            has_body=p.content.has_body,
        )
        # A domain learned to require the browser skips the curl-then-zendriver
        # probe -- but only when the request is browser-eligible to begin with.
        # A POST, body, or raw-header request resolves to plain curl, which the
        # learned hint must NOT override into the GET-only browser leg.
        if (
            resolved == "curl-then-zendriver"
            and self.domain in transport_routing.zendriver_domains()
        ):
            resolved = "zendriver"
        if resolved != p.policy.transport:
            p = replace(p, policy=replace(p.policy, transport=resolved))
        learner = _ResponseLearner(url=self.url, caller=p.observe.on_response)
        request = replace(self, params=p, observer=learner.observe)
        seeded_cookies = {
            **self.session.cookies_for(self.url),
            **(p.content.cookies or {}),
        }
        if p.content.raw_headers:
            body = request.send(
                headers=p.content.headers,
                cookies=seeded_cookies or None,
                raw_headers=True,
            )
        else:
            body = _fetch_with_identity(
                request,
                caller_headers=p.content.headers,
                caller_cookies=seeded_cookies or None,
            )
        return body, learner.merge_into(self.session)

    def send(
        self,
        *,
        headers: dict[str, str] | None,
        cookies: dict[str, str] | None,
        raw_headers: bool,
        on_response: Observer | None = None,
        curl: cc_requests.Session[Response] | None = None,
        reseat: Callable[[str], cc_requests.Session[Response] | None] | None = None,
    ) -> bytes:
        """Perform the request once (with retries) via the raw transport."""
        return _fetch_once(
            self.url,
            self.params,
            headers=headers,
            cookies=cookies,
            raw_headers=raw_headers,
            impersonate=self.session.impersonate,
            accept_ch=self.session.accept_ch,
            on_response=self.observer if on_response is None else on_response,
            session=curl,
            reseat=reseat,
        )


def _validated_body(request: _Request, body: bytes) -> bytes:
    """Validate and return ``body`` using the caller's provider-specific rule."""
    if request.params.observe.body_validator is not None:
        request.params.observe.body_validator(body)
    return body


def _fetch_with_identity(
    request: _Request,
    *,
    caller_headers: dict[str, str] | None,
    caller_cookies: dict[str, str] | None,
) -> bytes:
    """Send ``request`` under the stored per-``(egress, domain)`` identity."""
    if request.params.policy.transport == "zendriver":
        return _validated_body(
            request,
            _send_via_zendriver(
                request,
                headers=caller_headers,
                cookies=caller_cookies,
            ),
        )
    if request.params.policy.transport == "curl-then-zendriver":
        # Curl first (fast, cheap); fall back to the real browser ONLY when curl
        # is bot-blocked -- the one case the browser can clear that curl cannot.
        # A non-block failure (404, timeout) propagates: the browser would not
        # help and must not silently pay Chrome's launch cost.
        try:
            return _fetch_with_identity(
                replace(
                    request,
                    params=replace(
                        request.params,
                        policy=replace(request.params.policy, transport="curl"),
                    ),
                ),
                caller_headers=caller_headers,
                caller_cookies=caller_cookies,
            )
        except BotDetectionError:
            if request.domain:
                transport_routing.remember_zendriver_domain(request.domain)
            return _validated_body(
                request,
                _send_via_zendriver(
                    request,
                    headers=caller_headers,
                    cookies=caller_cookies,
                ),
            )
    domain = request.domain
    if not domain:
        return _validated_body(
            request,
            _send_as(request, None, None, caller_headers, caller_cookies),
        )

    store = ProfileStore.shared()
    # A cheap last-known egress finds an existing session; a NEW session must pin
    # to the live egress so a stale last-known never mis-keys the identity.
    egress = egress_ip(cache=True)
    profile = store.load(egress, domain) if egress is not None else None
    if profile is None:
        egress = egress_ip(cache=False)

    try:
        return _validated_body(
            request,
            _send_as(request, profile, egress, caller_headers, caller_cookies),
        )
    except BotDetectionError:
        if profile is None:
            raise  # First contact burned: no known identity to discard or retry.
        assert egress is not None  # A profile only loads once egress resolved.
        store.discard(egress, domain)
        close_curl_session(egress, domain, request.session.impersonate)
        # The burn may be a VPN rotation: re-resolve live before the fresh retry.
        return _validated_body(
            request,
            _send_as(
                request, None, egress_ip(cache=False), caller_headers, caller_cookies
            ),
        )


def _send_as(
    request: _Request,
    profile: Profile | None,
    egress: str | None,
    caller_headers: dict[str, str] | None,
    caller_cookies: dict[str, str] | None,
) -> bytes:
    """Send seeded with ``profile``'s UA + cookies (caller wins), save Set-Cookie."""
    # egress=None => keyless: draw a UA, persist nothing. A drawn UA matches the
    # request's impersonated browser so the UA and TLS fingerprint agree.
    impersonate = request.session.impersonate
    ua = (
        profile.ua
        if profile is not None
        else draw_user_agent(kind_for_impersonate(impersonate))
    )
    jar = dict(profile.cookies) if profile is not None else {}
    # On the curl path, curl_cffi's impersonate emits a COHERENT User-Agent that
    # matches its TLS/HTTP-2 fingerprint; seeding a stored/drawn UA here would
    # override it and make the two disagree (a bot tell). Let impersonate own the
    # UA on curl; only the stdlib reference path (no impersonation) needs one.
    seeded_headers: dict[str, str] = {**(caller_headers or {})}
    if request.params.policy.transport == "stdlib":
        seeded_headers = {"User-Agent": ua, **seeded_headers}
    captured: dict[str, str] = {}

    request_origin = origin(request.url)

    def capture(status: int, resp_headers: dict[str, str], url: str) -> None:
        # Persist only cookies the request's OWN origin set; a cross-origin
        # redirect target's Set-Cookie belongs to that origin's profile, not this
        # one, so it must not pollute (egress, request.domain).
        if origin(url) == request_origin:
            set_cookie = resp_headers.get("set-cookie")
            if set_cookie:
                captured.update(parse_set_cookie(set_cookie))
        if request.observer is not None:
            request.observer(status, resp_headers, url)

    curl = (
        curl_session(
            egress,
            request.domain,
            impersonate,
            pin=pinned_host(request.url, request.params.policy.trust),
            port=urlparse(request.url).port
            or default_port(urlparse(request.url).scheme),
        )
        if egress is not None and request.params.policy.transport == "curl"
        else None
    )
    # Single cookie source to avoid a duplicated Cookie header: when a curl
    # session drives the request, ITS jar persists and resends cookies across the
    # coalesced connection. Both the stored profile cookies AND the caller
    # cookies are loaded INTO that jar (caller value overwriting any jar entry of
    # the same name), so exactly one value per cookie goes on the wire -- sending
    # a caller cookie via the Cookie header too would duplicate a name the jar
    # already holds (a bot tell). On the stdlib path (no jar) both are seeded
    # into the header.
    if curl is not None:
        seed_session_jar(curl, request.domain, jar)
        set_session_cookies(curl, request.domain, caller_cookies or {})
        seeded_cookies = None
    else:
        seeded_cookies = {**jar, **(caller_cookies or {})}
    body = request.send(
        headers=seeded_headers,
        cookies=seeded_cookies or None,
        raw_headers=False,
        on_response=capture,
        curl=curl,
        reseat=functools.partial(_reseat, request, egress, impersonate)
        if curl is not None
        else None,
    )
    if egress is None:
        return body  # Keyless: nothing to persist.
    store = ProfileStore.shared()
    if profile is None:
        store.save(egress, request.domain, Profile(ua=ua, cookies=captured))
    elif captured:
        store.update_cookies(egress, request.domain, captured)
    return body


def _send_via_zendriver(
    request: _Request,
    *,
    headers: dict[str, str] | None,
    cookies: dict[str, str] | None,
) -> bytes:
    """Fetch ``request`` through the headless-Chrome backend, warming the session.

    Reuses the identity layer's egress resolution and ProfileStore so a browser
    fetch and a curl fetch on the same ``(egress, domain)`` share cookies. The
    cookies the browser acquired are folded back through the per-hop observer, so
    the :class:`FetchSession` the caller receives is warm and a following curl
    fetch reuses them.

    Under ``"untrusted"`` the host is validated but NOT pinned: Chrome resolves
    its own DNS and cannot be handed an IP without a proxy in front of it. That
    leaves a narrow window between validation and Chrome's own resolution --
    accepted deliberately, because the alternative (a per-request proxy) costs
    one Chrome process per pin and the measured 32x warm-browser reuse with it.
    """
    pinned_host(request.url, request.params.policy.trust)
    egress = egress_ip(cache=True) or egress_ip(cache=False)
    browser_url = _url_with_params(request.url, request.params.content.params)
    result = zendriver_backend.fetch_zendriver(
        browser_url,
        profile_dir=data_dir() / "rekursiv-ai" / "wesearch" / "fetch-zendriver",
        egress=egress or "",
        timeout_sec=request.params.retry.timeout_sec,
        headers=headers,
        cookies=cookies,
        on_redirect=request.params.observe.on_redirect,
    )
    if request.observer is not None:
        # Fold the harvested jar into the returned session (and the caller's
        # on_response) as a synthesized Set-Cookie for this origin, so the
        # session warms exactly as a header-level backend's would.
        #
        # Called even with an empty jar: ObserveParams.on_response promises a callback
        # for every response, and gating on cookies meant a successful
        # cookie-free browser fetch notified the caller of nothing at all.
        # Status and headers are synthesized because Chrome does not surface the
        # navigation response to this layer.
        synthesized = "\n".join(f"{k}={v}" for k, v in result.cookies.items())
        request.observer(
            200, {"set-cookie": synthesized} if result.cookies else {}, request.url
        )
    if egress is not None and request.domain and result.cookies:
        store = ProfileStore.shared()
        if store.load(egress, request.domain) is None:
            store.save(
                egress,
                request.domain,
                Profile(
                    ua=draw_user_agent(
                        kind_for_impersonate(request.session.impersonate)
                    ),
                    cookies=dict(result.cookies),
                ),
            )
        else:
            store.update_cookies(egress, request.domain, result.cookies)
    return result.body


def _reseat(
    request: _Request, egress: str | None, impersonate: str, url: str
) -> cc_requests.Session[Response] | None:
    """Return the pooled Session for a cross-host redirect target.

    A pin is fixed when its Session is built, so following a redirect onto a new
    host on the old host's Session would connect through the wrong pin. The new
    host is validated here, exactly as the initial hop was -- a redirect target
    is attacker-controlled and skipping its check is the classic SSRF bypass.
    """
    if egress is None:
        return None
    parsed = urlparse(url)
    return curl_session(
        egress,
        parsed.hostname or "",
        impersonate,
        pin=pinned_host(url, request.params.policy.trust),
        port=parsed.port or default_port(parsed.scheme),
    )


_egress_lock = threading.Lock()  # guards egress state, not a tunable

# Memoized from the most recent successful probe by any :func:`egress_ip` call,
# so the whole process shares one observed egress. ``None`` until the first
# successful probe; a failed probe leaves the prior value intact. A deliberate
# module global: shared *observed* state (a last-seen cache), not a tunable.
_last_egress_ip: str | None = None


def last_known_egress_ip() -> str | None:
    """Return the last-known egress IP without any network, or ``None`` if unseen."""
    with _egress_lock:
        return _last_egress_ip


# Callbacks fired when the egress rolls, so any transport that pools live,
# egress-keyed resources (open connections, running browsers) can invalidate
# them WITHOUT fetch.py naming that transport. fetch.py tears down its OWN curl
# pool inline; other backends (e.g. the zendriver browser pool) register here.
# An observer registry, not a tunable.
_on_egress_rotation: list[Callable[[str | None], None]] = []


def on_egress_rotation(callback: Callable[[str | None], None]) -> None:
    """Register ``callback`` to run when the egress IP rolls (a VPN change).

    Called with the new IP whenever :func:`set_last_egress_ip` observes a change.
    A backend that pools egress-keyed live resources registers its teardown here,
    so this module never has to know that backend exists to invalidate it.
    """
    _on_egress_rotation.append(callback)


def set_last_egress_ip(ip: str | None) -> None:
    """Set the process-wide last-known egress IP (e.g. after a known VPN roll).

    A changed IP means the exit rolled, so every pooled Session for a DIFFERENT
    egress is now dead (its connection went out the old exit) -- close and drop
    them, leaving only the new egress's sessions. Registered rotation callbacks
    (:func:`on_egress_rotation`) then fire so other backends invalidate their own
    egress-keyed pools.
    """
    global _last_egress_ip  # noqa: PLW0603 -- memoize shared observed state.
    with _egress_lock:
        rolled = ip != _last_egress_ip
        _last_egress_ip = ip
    if rolled:
        close_curl_sessions_except(ip)
        for callback in _on_egress_rotation:
            callback(ip)


# The zendriver browser pool is egress-keyed live state (running Chromes), so a
# roll invalidates it exactly like the curl pool. shutdown_browsers is a no-op
# when no pool exists, so this subscription is free until a browser fetch runs.
# Registered via the rotation hook so the teardown mechanism is uniform, not a
# special case wired into set_last_egress_ip.
on_egress_rotation(lambda _ip: zendriver_backend.shutdown_browsers())


def egress_ip(
    *,
    cache: bool = True,
    ipv6: bool = False,
    v4_echoes: Sequence[str] = (
        "https://ipv4.icanhazip.com",
        "https://api.ipify.org",
    ),
    v6_echoes: Sequence[str] = (
        "https://ipv6.icanhazip.com",
        "https://api64.ipify.org",
    ),
    timeout_sec: float = 5.0,
) -> str | None:
    """Return this host's public egress IP, or ``None`` if none resolves.

    Args:
      cache: When true (default), return the last-known value if set, probing
        only to fill it. False always probes live. Ignored when ``ipv6`` is set:
        the cache holds the v4 egress that keys the identity layer's profiles.
      ipv6: Resolve the IPv6 egress instead of IPv4. Always probes.
      v4_echoes: v4-only echo hosts tried in order.
      v6_echoes: v6-only echo hosts tried in order.
      timeout_sec: Per-request HTTP timeout.

    Returns:
      ip: The public address of the requested family, or ``None`` when none
        resolves (offline, or no egress of that family).

    """
    # The cache holds ONE address, and the identity layer keys profiles by it,
    # so it is the v4 egress. A v6 lookup must not read it -- returning a cached
    # v4 for ipv6=True answered a different question than the caller asked.
    if cache and not ipv6 and (cached := last_known_egress_ip()) is not None:
        return cached
    echoes = v6_echoes if ipv6 else v4_echoes
    for url in echoes:
        try:
            # raw_headers bypasses the identity layer: the profile is keyed by
            # egress IP and resolving it is THIS call, so a profiled echo would
            # recurse. A bare GET is all an echo service needs.
            body, _ = fetch(
                url,
                request=RequestParams(
                    content=ContentParams(headers={}, raw_headers=True),
                    retry=RetryParams(timeout_sec=timeout_sec),
                ),
            )
            ip = body.decode().strip()
        except (FetchError, OSError, ValueError):
            continue
        if _is_valid_ip_address(ip, ipv6=ipv6):
            set_last_egress_ip(ip)
            return ip
    return None


def _is_valid_ip_address(text: str, *, ipv6: bool) -> bool:
    """Whether *text* is a valid IP address of the requested family."""
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return False
    return addr.version == (6 if ipv6 else 4)


def _url_with_params(
    url: str,
    params: Mapping[str, str | int] | None,
) -> str:
    """Return ``url`` with encoded query parameters merged into its query.

    Split rather than concatenated: a fragment ends the URL, so appending to the
    raw string put the query AFTER it (``https://e/p#section?q=x``). Fragments
    are never sent to a server, so those parameters silently vanished from the
    wire.
    """
    if not params:
        return url
    parts = urlsplit(url)
    query = f"{parts.query}&{urlencode(params)}" if parts.query else urlencode(params)
    return urlunsplit(parts._replace(query=query))


def _split_userinfo(url: str) -> tuple[str, str | None]:
    """Strip ``user:pass@`` from a URL; return the URL and Basic auth."""
    parsed = urlparse(url)
    if not (parsed.username or parsed.password):
        return url, None
    user = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    credentials = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    netloc = parsed.netloc[parsed.netloc.rfind("@") + 1 :]
    return parsed._replace(netloc=netloc).geturl(), f"Basic {credentials}"


def _fetch_once(
    url: str,
    params: RequestParams,
    *,
    headers: dict[str, str] | None,
    cookies: dict[str, str] | None,
    raw_headers: bool,
    impersonate: str,
    accept_ch: Mapping[str, frozenset[str]],
    on_response: Observer | None,
    session: cc_requests.Session[Response] | None,
    reseat: Callable[[str], cc_requests.Session[Response] | None] | None = None,
) -> bytes:
    """Build and send one request (with retries), no profile layer.

    The raw transport core: encodes query/body/headers and dispatches to the
    curl or stdlib path with the shared retry loop. The profile-aware
    :func:`fetch` wraps this; ``raw_headers`` callers reach it directly (no
    cookie jar). ``params`` supplies the validated per-call policy; ``headers`` /
    ``cookies`` / ``impersonate`` are resolved by the identity layer above;
    ``accept_ch`` is the session's per-origin extended-hint opt-ins;
    ``on_response`` may wrap ``params.on_response`` to capture cookies; and
    ``session`` is a pooled curl_cffi Session to reuse (the identity's persistent
    connection), or ``None`` to open a throwaway one.
    """
    url, basic_auth = _split_userinfo(url)
    url = _url_with_params(url, params.content.params)
    body_bytes: bytes | None = None
    body_content_type: str | None = None
    if params.content.data is not None:
        body_bytes = urlencode(params.content.data).encode()
        body_content_type = "application/x-www-form-urlencoded"
    elif params.content.json is not NO_BODY:
        body_bytes = json_lib.dumps(params.content.json).encode()
        body_content_type = "application/json"
    merged = _build_headers(
        method=params.content.method,
        url=url,
        content_type=body_content_type,
        extra=headers,
        raw_headers=raw_headers,
        impersonate=impersonate,
        use_curl=params.policy.transport == "curl",
        accept_ch=accept_ch,
    )
    if basic_auth is not None:
        merged.setdefault("Authorization", basic_auth)
    # HTTP header names are case-insensitive: collapse any caller-supplied
    # case-variant "cookie" header and the cookies= param into ONE Cookie key.
    # Two dict keys ("cookie" + "Cookie") would emit two Cookie lines on the wire
    # -- a bot tell. The param values follow the caller's header pairs.
    cookie_parts = [
        merged.pop(key) for key in [k for k in merged if k.lower() == "cookie"]
    ]
    if cookies:
        cookie_parts.append("; ".join(f"{k}={v}" for k, v in cookies.items()))
    if cookie_parts:
        merged["Cookie"] = "; ".join(cookie_parts)
    method = params.content.method
    backend = fetch_curl if params.policy.transport == "curl" else fetch_stdlib
    for attempt in range(1 + params.retry.retries):
        try:
            return backend(
                url,
                method=method,
                headers=merged,
                body=body_bytes,
                timeout_sec=params.retry.timeout_sec,
                connect_timeout_sec=params.retry.connect_timeout_sec,
                max_redirects=params.retry.max_redirects,
                impersonate=impersonate,
                on_redirect=params.observe.on_redirect,
                on_response=on_response,
                trust=params.policy.trust,
                session=session,
                reseat=reseat,
            )
        except FetchError as e:
            # status 0 is the transport-failure sentinel (a curl CurlError, or a
            # connection/TLS failure wrapped by a transport) -- retryable like the
            # OSError below, which the stdlib path raises for the same class of
            # failure. Without this the two transports disagree on `retries=`.
            retryable = e.status in _RETRYABLE_STATUSES or e.status == 0
            if not retryable or attempt == params.retry.retries:
                raise
            delay_sec = params.retry.backoff_delay(attempt, e.headers)
            logger.debug(
                "fetch %s → %d, retry in %.1fs",
                url,
                e.status,
                delay_sec,
            )
            time.sleep(delay_sec)
        except (OSError, TimeoutError) as e:
            if attempt == params.retry.retries:
                raise
            delay_sec = params.retry.backoff_delay(attempt, {})
            logger.debug(
                "fetch %s failed: %s, retry in %.1fs",
                url,
                e,
                delay_sec,
            )
            time.sleep(delay_sec)
    # The loop returns on success and re-raises on the final attempt, so this
    # is unreachable; it exists only to satisfy the type checker.
    raise AssertionError("retry loop exited without returning or raising")


def _build_headers(
    *,
    method: str,
    url: str,
    content_type: str | None,
    extra: Mapping[str, str] | None,
    raw_headers: bool,
    impersonate: str,
    use_curl: bool,
    accept_ch: Mapping[str, frozenset[str]],
) -> dict[str, str]:
    """Build canonical-order Chrome request headers.

    On the high-level curl transport, curl_cffi's ``impersonate`` supplies the
    coherent Chrome fingerprint (User-Agent, ``sec-ch-ua`` hints, Accept,
    Sec-Fetch-*, Priority) matching its TLS/HTTP-2 profile exactly. Overriding
    those with hand-built values makes the identities disagree -- a bot tell --
    so that path emits ONLY the structural headers curl does not set
    (Origin/Content-Type on a POST), the extended client hints an origin opted
    into via ``Accept-CH``, and caller extras.
    """
    # Host and Content-Length are omitted: http.client auto-adds both first on
    # the wire (the connection path overrides Host when validated_hosts splits
    # SNI/IP); curl adds them itself.
    if raw_headers:
        return dict(extra) if extra else {}
    if use_curl:
        return _curl_structural_headers(
            method=method,
            url=url,
            content_type=content_type,
            extra=extra,
            impersonate=impersonate,
            accept_ch=accept_ch,
        )
    # The stdlib path gets no impersonation-injected request headers, so it
    # reproduces the full Chrome header set by hand -- from the SAME source
    # (chrome_navigation_headers, matched to the impersonate target) the
    # high-level curl path's fingerprint uses, so every transport presents one
    # coherent identity, not drifting ones.
    parsed = urlparse(url)
    major, platform = impersonate_version_platform(impersonate)
    h = chrome_navigation_headers(
        major=major,
        platform=platform,
        method=method,
        content_type=content_type or "",
        origin=f"{parsed.scheme}://{parsed.netloc}",
        # This branch is reached only by the stdlib path, which is HTTP/1.1; a
        # real Chrome omits the HTTP/2-only Priority header there.
        http2=False,
    )
    h.update(_google_headers(url, impersonate))
    if extra:
        # Caller wins; dict.update preserves slot for existing keys and
        # appends new ones at the end.
        h.update(extra)
    return h


def _google_headers(url: str, impersonate: str) -> dict[str, str]:
    """The Google-only integrity headers for ``url``, empty off a Google host.

    A real Chrome sends ``x-browser-*`` / ``x-client-data`` only to Google
    properties, so :func:`fetch` adds them by host -- every transport, no caller
    wiring -- and nothing to any other origin.
    """
    if not is_google_property(urlparse(url).hostname or ""):
        return {}
    major, platform = impersonate_version_platform(impersonate)
    return chrome_headers_for_google(major=major, platform=platform)


def _accept_ch_hints(resp_headers: dict[str, str]) -> frozenset[str]:
    """The extended client-hint names an ``Accept-CH`` response requested."""
    accept_ch = resp_headers.get("accept-ch")
    if accept_ch is None:
        return frozenset[str]()
    wanted = {tok.strip().lower() for tok in accept_ch.split(",") if tok.strip()}
    return frozenset(name for name in chrome_client_hints(major=1) if name in wanted)


def _curl_structural_headers(
    *,
    method: str,
    url: str,
    content_type: str | None,
    extra: Mapping[str, str] | None,
    impersonate: str,
    accept_ch: Mapping[str, frozenset[str]],
) -> dict[str, str]:
    """Headers for the curl path: what impersonate omits + Accept-CH opt-ins.

    Verified on Linux against real Chrome (146, the impersonate target): the
    FIRST request to an
    origin sends only the core set curl_cffi's impersonate reproduces (UA, the
    three basic ``sec-ch-ua`` hints, Accept, Sec-Fetch-*, Priority,
    Accept-Encoding/Language). It adds the EXTENDED client hints
    (``sec-ch-ua-arch`` etc.) only AFTER the server opts in via ``Accept-CH``,
    on subsequent same-origin requests. We mirror that exactly: a cold origin
    gets nothing extra (adding hints an unrequesting site never asked for is
    itself a tell), and an origin that has sent Accept-CH gets precisely the
    hints it requested, version-matched to the impersonate target. Only a POST's
    Origin/Content-Type and caller extras follow (Cookie is merged by caller).
    """
    h: dict[str, str] = {}
    wanted = accept_ch.get(origin(url))
    if wanted:
        major, platform = impersonate_version_platform(impersonate)
        hints = chrome_client_hints(major=major, platform=platform)
        h.update({name: value for name, value in hints.items() if name in wanted})
    if method not in ("GET", "HEAD"):
        if content_type:
            h["Content-Type"] = content_type
        parsed = urlparse(url)
        h["Origin"] = f"{parsed.scheme}://{parsed.netloc}"
    h.update(_google_headers(url, impersonate))
    if extra:
        h.update(extra)
    return h
