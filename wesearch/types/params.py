"""Per-call parameters for :func:`wesearch.fetch.fetch`, grouped by concern.

Four groups, split by who owns the value and how often it varies:

- :class:`ContentParams` -- what to send. The caller's request, per call.
- :class:`RetryParams` -- how hard to try. The caller's patience, rarely varied.
- :class:`ObserveParams` -- who watches. The caller's instrumentation, per call.
- :class:`PolicyParams` -- transport and trust. The APPLICATION's decision,
  constant for its whole lifetime.

The split is not cosmetic. A flat parameter bag let a security policy
(``validated_hosts``) double as a transport selector and a session-pool
disabler: asking for SSRF safety silently disabled the browser transports and
dropped the cookie jar. Grouping makes that category error unrepresentable --
:class:`ContentParams` holds nothing a transport can route on, and
:class:`PolicyParams` holds nothing a server ever sees.

``PolicyParams`` is one opaque object a provider forwards rather than a pair of
knobs it must restate; a provider that forgets it gets a type error, never a
silent unvalidated fetch.

These are the VALUES a caller passes. The DESCRIPTIONS a tool surface renders --
type, default, and prose per parameter -- live in
:mod:`wesearch.types.schema`, which reads its defaults back off
:meth:`PolicyParams.field_default`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import math
import random

from wesearch.lib.custom_json import JSONValue
from wesearch.profile import parsedate_to_datetime_or_none


if TYPE_CHECKING:
    from curl_cffi.requests.session import HttpMethod


__all__ = [
    "ContentParams",
    "Extractor",
    "ObserveParams",
    "PolicyParams",
    "RequestParams",
    "RetryParams",
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
# as Markdown; ``"trafilatura"`` scores blocks and returns only what it judges to
# be the article; ``"raw"`` returns the source untouched; ``"markdownify"``
# converts the document's elements, which keeps nested structure a text walk
# flattens.
#
# They are not ranked versions of one idea -- they answer different questions --
# but the default is not a toss-up. ``scripts/compare_extractors.py`` scores
# every converter against hand-picked content probes on an 11-page corpus of the
# shapes that break extractors. Measured there, ``html2text`` loses 0 of 37
# probes and ``trafilatura`` loses 12: it keeps ONE answer of a StackOverflow
# thread (spliced, trailing "9more comments"), returns 864 characters of a
# profile timeline where ``html2text`` returns 4_530, and drops a dictionary
# entry's pronunciation. It is also
# undetectable -- trafilatura's API is ``str | None`` with no confidence signal,
# no HTML5 landmark predicts the loss (x.com matches its strongest rule and loses
# every probe; Hacker News matches none and loses none), and the one available
# proxy -- output length against another converter's -- is anti-correlated with
# the compression that motivates using it at all.
#
# So ``html2text`` is the default: whole, at ~2.5x the characters. Reach for
# ``trafilatura`` deliberately, on a page whose substance IS one contiguous prose
# body (an encyclopedia article, a spec document), where it is both smaller and
# lossless. Ordered here by that expected reach; ``get_args`` feeds every schema
# enum, so this order is what a caller reads first.
#
# A ``Literal`` for the same reason as ``Transport`` above.
Extractor: TypeAlias = Literal[  # noqa: UP040 -- type keyword breaks get_args()
    "html2text", "trafilatura", "raw", "markdownify"
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentParams:
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
class RetryParams:
    """How hard to try: the caller's patience.

    Attributes:
      retries: RetryParams attempts for transient failures.
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
        # Finite, not merely positive: NaN fails every comparison so it slipped
        # through as "in range", and infinity passed outright -- both then reach
        # socket and browser timeout APIs, removing the ceiling this class
        # exists to impose.
        if not math.isfinite(self.timeout_sec) or self.timeout_sec <= 0:
            raise ValueError(
                f"'timeout_sec' must be a finite number > 0, got {self.timeout_sec}."
            )
        if self.connect_timeout_sec is not None and (
            not math.isfinite(self.connect_timeout_sec) or self.connect_timeout_sec <= 0
        ):
            raise ValueError(
                "'connect_timeout_sec' must be a finite number > 0,"
                f" got {self.connect_timeout_sec}."
            )
        if self.max_redirects < 0:
            raise ValueError(f"'max_redirects' must be >= 0, got {self.max_redirects}.")

    def backoff_delay(self, attempt: int, headers: dict[str, str]) -> float:
        """RetryParams backoff in seconds for ``attempt``, honoring any ``RetryParams-After``.

        ``RetryParams-After`` is delta-seconds OR an HTTP-date (RFC 9110 SS 10.2.3);
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
class ObserveParams:
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
class PolicyParams:
    """Transport, extractor, and trust: the application's decision, not the request's.

    Constant for an application's lifetime -- sagent's URLs are always
    agent-supplied, a test oracle's are always its own -- so it is threaded as
    one object rather than restated at each call site.

    ``trust`` never selects a transport. It is read once, at connect time, by
    whichever transport runs: the header transports validate and pin the connect
    IP, the browser validates and lets Chrome resolve (Chrome owns its own DNS
    and cannot be handed an IP without a proxy). Conflating the two is what made
    a security choice disable the browser.

    ``extractor`` sits here rather than in :class:`ContentParams` because it is the
    application's rendering choice and nothing a server ever sees -- the same
    test that puts ``transport`` here.

    The class-level defaults ARE the defaults every surface renders. A
    signature default cannot call anything (ruff B008), so a tool reads
    ``PolicyParams.extractor`` -- the annotation on the field, not an instance -- and
    a spec reads the same. Spelling ``"html2text"`` at each site instead is how
    the previous switch needed four edits plus two prose files to land.

    Attributes:
      transport: Retrieval transport; see :data:`Transport`.
      extractor: HTML-to-text extractor; see :data:`Extractor`.
      trust: Provenance of the URL; see :data:`Trust`.

    """

    transport: Transport = "auto"
    extractor: Extractor = "html2text"
    trust: Trust = "untrusted"

    @classmethod
    def field_default[T](cls, name: str, kind: type[T] | object = object) -> T:
        """Return the class's declared default for ``name``.

        ``slots=True`` replaces the class attribute with a descriptor, so
        ``PolicyParams.extractor`` is a slot object rather than ``"html2text"``;
        ``dataclasses.fields`` is where the declared value survives. ``kind``
        types the result for a caller whose annotation is a ``Literal``; it is
        not re-checked, since this class declared the value.

        Raises:
          KeyError: When ``name`` is not a field of this class.

        """
        del kind
        for field_ in fields(cls):
            if field_.name == name:
                return cast("T", field_.default)
        raise KeyError(f"{cls.__name__} has no field {name!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestParams:
    """Per-call parameters for :func:`wesearch.fetch.fetch`.

    Attributes:
      content: What to send.
      retry: How hard to try.
      observe: Who watches.
      policy: Transport and trust.

    """

    content: ContentParams = field(default_factory=ContentParams)
    retry: RetryParams = field(default_factory=RetryParams)
    observe: ObserveParams = field(default_factory=ObserveParams)
    policy: PolicyParams = field(default_factory=PolicyParams)

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
