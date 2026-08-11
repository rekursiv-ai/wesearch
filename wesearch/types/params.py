"""Per-call parameters for :func:`wesearch.fetch.fetch`, grouped by concern.

Four groups, split by who owns the value and how often it varies:

- :class:`Content` -- what to send. The caller's request, per call.
- :class:`Retry` -- how hard to try. The caller's patience, rarely varied.
- :class:`Observe` -- who watches. The caller's instrumentation, per call.
- :class:`Policy` -- transport and trust. The APPLICATION's decision, constant
  for its whole lifetime.

The split is not cosmetic. A flat parameter bag let a security policy
(``validated_hosts``) double as a transport selector and a session-pool
disabler: asking for SSRF safety silently disabled the browser transports and
dropped the cookie jar. Grouping makes that category error unrepresentable --
:class:`Content` holds nothing a transport can route on, and :class:`Policy`
holds nothing a server ever sees.

``Policy`` is one opaque object a provider forwards rather than a pair of knobs
it must restate; a provider that forgets it gets a type error, never a silent
unvalidated fetch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, TypeAlias

import math
import random

from wesearch.lib.custom_json import JSONValue
from wesearch.profile import parsedate_to_datetime_or_none


if TYPE_CHECKING:
    from curl_cffi.requests.session import HttpMethod


__all__ = [
    "Content",
    "Extractor",
    "Observe",
    "Policy",
    "RequestParams",
    "Retry",
    "Transport",
    "Trust",
]

# Retrieval transport. ``"auto"`` selects curl-then-Zendriver for eligible GETs
# and curl for requests a browser cannot replay. ``"curl"`` is the curl_cffi
# impersonated path; ``"stdlib"`` the http.client reference path; ``"zendriver"``
# a real headless Chrome; ``"curl-then-zendriver"`` tries curl and falls back to
# the browser ONLY on a bot block.
#
# A ``Literal`` rather than a ``type`` alias: the sagent tool schemas enumerate
# these with ``get_args`` into JSON Schema, which a lazily-evaluated ``type``
# alias breaks.
Transport: TypeAlias = Literal[  # noqa: UP040 -- type keyword breaks get_args()
    "auto", "curl", "curl-then-zendriver", "zendriver", "stdlib"
]

# Where the URL came from, NOT what DNS to do about it. ``"untrusted"`` (the
# default) means the URL came from somewhere the application does not control --
# an agent, a user, a fetched page -- so the host is validated to a public
# address before connecting. ``"internal"`` is an explicit opt-out for a URL the
# application authored: a loopback test oracle, a private-network service.
#
# Safe by default, and deliberately: the previous contract defaulted to
# unvalidated, so every caller had to opt in and the one that did lost the
# browser transports for it.
Trust: TypeAlias = Literal["untrusted", "internal"]  # noqa: UP040 -- get_args()

# How a fetched HTML page becomes text. ``"html2text"`` renders every text node
# as Markdown; ``"markdownify"`` converts the document's elements instead, which
# keeps nested structure a text walk flattens; ``"trafilatura"`` scores blocks
# and returns only what it judges to be the article; ``"raw"`` returns the
# source untouched.
#
# They are not ranked versions of one idea -- they answer different questions,
# and the article-shaped assumption behind ``"trafilatura"`` is wrong for a
# dictionary entry, a Q&A thread, or a profile timeline, where it returns
# plausible-looking output with the substance missing (a StackOverflow thread
# minus every answer). ``wesearch/scripts/compare_extractors.py`` measures
# all of them against a corpus of those shapes.
#
# A ``Literal`` for the same reason as ``Transport`` above.
Extractor: TypeAlias = Literal[  # noqa: UP040 -- type keyword breaks get_args()
    "html2text", "markdownify", "trafilatura", "raw"
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Content:
    """What to send: the request itself.

    Attributes:
      method: HTTP method.
      params: Query parameters appended to the URL.
      data: Form data, sent as application/x-www-form-urlencoded. Mutually
        exclusive with ``json``.
      json: JSON-serializable body, sent as application/json. Mutually exclusive
        with ``data``.
      headers: Extra headers, merged over the session identity (these win).
      cookies: Cookies to send, merged over the session jar (these win). On the
        curl path these are written INTO the pooled jar, which outlives the
        call, so a per-call override persists to later requests on the same
        ``(egress, domain)`` pool until it is overwritten or the pool closes.
        Deliberate: the jar must stay the single cookie source on that path,
        because sending a caller cookie via a header too would duplicate a name
        the jar already holds -- a bot tell.
      raw_headers: Send exactly ``headers`` plus cookies and auth; skip the
        Chrome identity and the session jar.

    """

    method: HttpMethod = "GET"
    params: dict[str, str | int] | None = None
    data: dict[str, str] | None = None
    json: JSONValue = None
    headers: dict[str, str] | None = None
    cookies: dict[str, str] | None = None
    raw_headers: bool = False

    def __post_init__(self) -> None:
        """Reject a request carrying two mutually exclusive bodies."""
        if self.data is not None and self.json is not None:
            raise ValueError("'data' and 'json' are mutually exclusive.")

    @property
    def has_body(self) -> bool:
        """Whether this request carries a body a browser cannot replay."""
        return self.data is not None or self.json is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class Retry:
    """How hard to try: the caller's patience.

    Attributes:
      retries: Retry attempts for transient failures.
      timeout_sec: Socket timeout in seconds, covering the whole request.
      connect_timeout_sec: Ceiling on the TCP/TLS handshake alone, before any
        byte of the response. Separate from ``timeout_sec`` because the two
        bound different failures: a slow PAGE is worth waiting out, an
        unreachable HOST is not, and one budget cannot say so. A dropped SYN
        yields no RST, so the only signal is the clock -- with a single budget
        an unroutable host costs the full ``timeout_sec`` to learn what the
        handshake already knew. ``None`` lets it share ``timeout_sec``.
      max_redirects: Maximum redirects to follow; 0 disables.

    """

    retries: int = 0
    timeout_sec: float = 30
    connect_timeout_sec: float | None = None
    max_redirects: int = 10

    def __post_init__(self) -> None:
        """Reject out-of-range patience values."""
        if self.retries < 0:
            raise ValueError(f"'retries' must be >= 0, got {self.retries}.")
        if self.timeout_sec <= 0:
            raise ValueError(f"'timeout_sec' must be > 0, got {self.timeout_sec}.")
        if self.connect_timeout_sec is not None and self.connect_timeout_sec <= 0:
            raise ValueError(
                f"'connect_timeout_sec' must be > 0, got {self.connect_timeout_sec}."
            )
        if self.max_redirects < 0:
            raise ValueError(f"'max_redirects' must be >= 0, got {self.max_redirects}.")

    def backoff_delay(self, attempt: int, headers: dict[str, str]) -> float:
        """Retry backoff in seconds for ``attempt``, honoring any ``Retry-After``.

        ``Retry-After`` is delta-seconds OR an HTTP-date (RFC 9110 SS 10.2.3);
        both forms are honored, capped at 30s. A malformed value falls through to
        the computed exponential backoff.

        Args:
          attempt: Zero-based attempt index.
          headers: Response headers of the failed attempt; empty for a
            network-error retry, which always uses the computed backoff.

        Returns:
          delay_sec: Seconds to wait before the next attempt.

        """
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                seconds = float(retry_after)
            except ValueError:
                seconds = math.nan
            # A hostile or broken origin can send "-1" or "nan"; both reach
            # time.sleep, which raises ValueError on a negative and never wakes
            # on a NaN. Either turns a retryable response into a crash that
            # masks the underlying FetchError.
            if math.isfinite(seconds):
                return min(max(seconds, 0.0), 30)
            when = parsedate_to_datetime_or_none(retry_after.strip())
            if when is not None:
                return min(max((when - datetime.now(UTC)).total_seconds(), 0.0), 30)
        delay_sec = min(1.0 * (2**attempt), 30)
        return delay_sec + random.uniform(0, delay_sec * 0.5)  # noqa: S311 -- jitter


@dataclass(frozen=True, slots=True, kw_only=True)
class Observe:
    """Who watches: the caller's instrumentation.

    Attributes:
      on_redirect: Called with the redirect target URL before following; raise to
        abort.
      on_response: Called with ``(status, headers)`` for every received response.
        Observational; must not raise. The browser transport cannot see the
        navigation response, so it synthesizes ``200`` and reports only the
        cookies Chrome harvested.
      body_validator: Called with every final response body before it is accepted.
        Raise :class:`~wesearch.types.errors.BotDetectionError` when a
        provider-specific success body proves browser or human interaction is
        required; automatic transport fallback then learns the domain.

    """

    on_redirect: Callable[[str], None] | None = None
    on_response: Callable[[int, dict[str, str]], None] | None = None
    body_validator: Callable[[bytes], None] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Policy:
    """Transport, extractor, and trust: the application's decision, not the request's.

    Constant for an application's lifetime -- sagent's URLs are always
    agent-supplied, a test oracle's are always its own -- so it is threaded as
    one object rather than restated at each call site.

    ``trust`` never selects a transport. It is read once, at connect time, by
    whichever transport runs: the header transports validate and pin the connect
    IP, the browser validates and lets Chrome resolve (Chrome owns its own DNS
    and cannot be handed an IP without a proxy). Conflating the two is what made
    a security choice disable the browser.

    ``extractor`` sits here rather than in :class:`Content` because it is the
    application's rendering choice and nothing a server ever sees -- the same
    test that puts ``transport`` here.

    Attributes:
      transport: Retrieval transport; see :data:`Transport`.
      extractor: HTML-to-text extractor; see :data:`Extractor`.
      trust: Provenance of the URL; see :data:`Trust`.

    """

    transport: Transport = "auto"
    extractor: Extractor = "html2text"
    trust: Trust = "untrusted"


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestParams:
    """Per-call parameters for :func:`wesearch.fetch.fetch`.

    Attributes:
      content: What to send.
      retry: How hard to try.
      observe: Who watches.
      policy: Transport and trust.

    """

    content: Content = field(default_factory=Content)
    retry: Retry = field(default_factory=Retry)
    observe: Observe = field(default_factory=Observe)
    policy: Policy = field(default_factory=Policy)

    def __post_init__(self) -> None:
        """Reject a request a browser transport could never perform.

        Checked here, at construction, rather than mid-fetch: a request that can
        never succeed must fail where it is written, not five frames deep in the
        transport. A runtime-only guard is how a browser+SSRF combination stayed
        broken for weeks -- it constructed cleanly and raised only when the
        caller happened to name an explicit transport.
        """
        if self.policy.transport not in ("zendriver", "curl-then-zendriver"):
            return
        # The browser leg replays GET navigation, headers, and cookies, but not
        # a request body or byte-exact raw-header mode.
        if self.content.method != "GET":
            raise ValueError(
                f"The {self.policy.transport} backend supports only GET requests."
            )
        if self.content.has_body:
            raise ValueError(
                f"The {self.policy.transport} backend cannot send a request body."
            )
        if self.content.raw_headers:
            raise ValueError(
                f"The {self.policy.transport} transport cannot honor 'raw_headers'."
            )
