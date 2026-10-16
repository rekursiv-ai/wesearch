"""Parameters for :func:`wesearch.fetch.fetch`.

- :class:`ContentParams`: what to send.
- :class:`RetryParams`: how long and how often to try.
- :class:`ObserveParams`: callbacks that watch or stop a fetch.
- :class:`PolicyParams`: transport, extractor, and private-host access, which
  an application usually fixes once.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from types import MappingProxyType
from typing import (
    Final,
    Literal,
    TypeAlias,
    cast,
    final,
    override,
)

import math
import random

from treekle import PlainTree

from wesearch.profile import parsedate_to_datetime_or_none


__all__ = [
    "NO_BODY",
    "ContentParams",
    "Extractor",
    "HttpMethod",
    "NoBody",
    "ObserveParams",
    "PolicyParams",
    "RequestParams",
    "RetryParams",
    "Transport",
    "ignore",
    "no_cookies",
]


# Declared here rather than imported from curl_cffi, which is optional at runtime:
# ``ContentParams`` is a dataclass, and resolving its hints needs this name.
HttpMethod: TypeAlias = Literal[  # noqa: UP040 -- get_type_hints must resolve it.
    "GET",
    "POST",
    "PUT",
    "DELETE",
    "OPTIONS",
    "HEAD",
    "TRACE",
    "PATCH",
    "QUERY",
]

# ``auto`` tries curl, then the browser if curl is blocked, for a GET; curl
# otherwise. ``curl`` looks like Chrome; ``stdlib`` needs no extra package;
# ``zendriver`` is a real Chrome; ``curl-then-zendriver`` is ``auto`` for a GET.
# A ``Literal``, not a ``type`` alias: tool schemas read it with ``get_args``.
Transport: TypeAlias = Literal[  # noqa: UP040 -- The type keyword preserves the runtime alias shape required by get_args().
    "auto",
    "curl",
    "curl-then-zendriver",
    "zendriver",
    "stdlib",
]

# How a page becomes text; see :func:`wesearch.fetch.html_to_text.html_to_text`.
# ``html2text`` is the default: on an 11-page benchmark
# (``scripts/compare_extractors.py``) it lost 0 of 37 content probes and
# ``trafilatura`` lost 12. Ordered by how often each fits; schemas list them so.
Extractor: TypeAlias = Literal[  # noqa: UP040 -- The type keyword preserves the runtime alias shape required by get_args().
    "html2text",
    "trafilatura",
    "raw",
    "markdownify",
]


@final
class NoBody:
    """The type of :data:`NO_BODY`; see it for why ``None`` cannot serve."""

    @override
    def __repr__(self) -> str:
        return "NO_BODY"


# No JSON body. Not ``None``: ``json=None`` sends the JSON literal ``null``.
NO_BODY: Final = NoBody()


def ignore(*_args: object) -> None:
    """Do nothing: the default for every optional callback."""


def no_cookies(host: str) -> Mapping[str, str]:
    """Return no cookies for ``host``: the default cookie source of a transport."""
    del host
    return MappingProxyType[str, str]({})


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentParams:
    """What to send: the request itself.

    Attributes:
      method: HTTP method.
      params: Query parameters appended to the URL.
      data: Form data, sent as application/x-www-form-urlencoded. Mutually
        exclusive with ``json``.
      json: JSON body, sent as application/json; :data:`NO_BODY` for none.
        Mutually exclusive with ``data``.
      headers: Sent over the transport's own Chrome headers, names in any
        case; a ``None`` value removes that header.
      cookies: Sent with the session's cookies for the host; these win.
      raw_headers: Send no stored cookies and no implied ``Content-Type``.

    """

    method: HttpMethod = "GET"
    params: dict[str, str | int] = field(default_factory=dict[str, str | int])
    data: dict[str, str] | None = None
    json: PlainTree | NoBody = NO_BODY
    headers: Mapping[str, str | None] = field(
        default_factory=dict[str, str | None],
    )
    cookies: dict[str, str] = field(default_factory=dict[str, str])
    raw_headers: bool = False

    def __post_init__(self) -> None:
        """Reject a request carrying two mutually exclusive bodies."""
        if self.data is not None and self.json is not NO_BODY:
            raise ValueError("'data' and 'json' are mutually exclusive.")

    @property
    def has_body(self) -> bool:
        """Whether this request carries a body a browser cannot replay."""
        return self.data is not None or self.json is not NO_BODY


@dataclass(frozen=True, slots=True, kw_only=True)
class RetryParams:
    """How hard to try: the caller's patience.

    Attributes:
      retries: Retries after a 429, a 5xx, or a failed connection.
      timeout_sec: Seconds allowed for the whole request.
      connect_timeout_sec: Seconds allowed to connect, so an unreachable host
        fails fast; the browser transport ignores it.
      max_redirects: Redirects to follow; 0 returns the redirect response.

    """

    retries: int = 0
    timeout_sec: float = 30.0
    connect_timeout_sec: float = 10.0
    max_redirects: int = 10

    def __post_init__(self) -> None:
        """Reject out-of-range patience values."""
        if self.retries < 0:
            raise ValueError(f"'retries' must be >= 0, got {self.retries}.")
        for name, value in (
            ("timeout_sec", self.timeout_sec),
            ("connect_timeout_sec", self.connect_timeout_sec),
        ):
            # NaN passes ``<= 0`` and infinity removes the limit.
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name!r} must be a finite number > 0, got {value}.")
        if self.max_redirects < 0:
            raise ValueError(f"'max_redirects' must be >= 0, got {self.max_redirects}.")

    def backoff_delay(self, attempt: int, headers: dict[str, str]) -> float:
        """Return seconds to wait before retry ``attempt``, at most 30.

        Honors the failed response's ``Retry-After`` (seconds or a date); else
        doubles from 1s, with jitter.

        Args:
          attempt: Retries already made.
          headers: The failed response's headers, names lower-case.

        Returns:
          delay_sec: Seconds to wait.

        """
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                seconds = float(retry_after)
            except ValueError:
                seconds = math.nan
            # "-1" or "nan" would crash or hang ``time.sleep``.
            if math.isfinite(seconds):
                return min(max(seconds, 0.0), 30)
            when = parsedate_to_datetime_or_none(retry_after.strip())
            if when is not None:
                return min(max((when - datetime.now(UTC)).total_seconds(), 0.0), 30)
        delay_sec = min(float(1 << attempt), 30)
        return delay_sec + random.uniform(0, delay_sec * 0.5)  # noqa: S311 -- This non-security jitter controls retry timing only.


@dataclass(frozen=True, slots=True, kw_only=True)
class ObserveParams:
    """Callbacks that watch, and may stop, a fetch.

    Attributes:
      on_redirect: Called with each redirect target before it is followed;
        raise to stop the fetch.
      on_response: Called with ``(status, headers)`` for every response,
        redirects included.
      body_validator: Called with the final body; raise
        :class:`~wesearch.types.errors.BotDetectionError` when a 200 page is
        really a block, so ``auto`` retries it in the browser.

    """

    on_redirect: Callable[[str], None] = ignore
    on_response: Callable[[int, dict[str, str]], None] = ignore
    body_validator: Callable[[bytes], None] = ignore


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyParams:
    """How to fetch: settings an application usually fixes once.

    Attributes:
      transport: Retrieval transport; see :data:`Transport`.
      extractor: HTML-to-text extractor; see :data:`Extractor`.
      allow_private_hosts: Allow loopback, private, and link-local addresses.
        Leave it ``False`` for any URL an agent, user, or page supplied; set it
        only for a service you run yourself.

    """

    transport: Transport = "auto"
    extractor: Extractor = "html2text"
    allow_private_hosts: bool = False

    @classmethod
    def field_default[T](cls, name: str, kind: type[T] | object = object) -> T:
        """Return the default of field ``name``.

        ``PolicyParams.extractor`` is a slot descriptor, not the default, so read
        defaults here.

        Args:
          name: Field name.
          kind: Type of the default, for the caller's checker; not verified.

        Returns:
          value: The field's default.

        Raises:
          KeyError: When ``name`` is not a field of this class.

        """
        del kind
        for field_ in fields(cls):
            if field_.name == name:
                return cast(T, field_.default)
        raise KeyError(f"{cls.__name__} has no field {name!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestParams:
    """Everything :func:`wesearch.fetch.fetch` sends and how.

    Attributes:
      content: What to send.
      retry: How long and how often to try.
      observe: Callbacks that watch or stop the fetch.
      policy: How to fetch.

    """

    content: ContentParams = field(default_factory=ContentParams)
    retry: RetryParams = field(default_factory=RetryParams)
    observe: ObserveParams = field(default_factory=ObserveParams)
    policy: PolicyParams = field(default_factory=PolicyParams)

    def __post_init__(self) -> None:
        """Reject a request the browser transport cannot send: a bodyless GET only."""
        if self.policy.transport not in ("zendriver", "curl-then-zendriver"):
            return
        if self.content.method != "GET":
            raise ValueError(
                f"The {self.policy.transport} backend supports only GET requests.",
            )
        if self.content.has_body:
            raise ValueError(
                f"The {self.policy.transport} backend cannot send a request body.",
            )
        if self.content.raw_headers:
            raise ValueError(
                f"The {self.policy.transport} transport cannot honor 'raw_headers'.",
            )
