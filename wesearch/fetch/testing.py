"""Shared test doubles for fetch transport tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from curl_cffi import requests as cc_requests


def lower_headers(kw: dict[str, Any]) -> dict[str, str]:
    """Lower-cased request headers from a curl ``request`` mock's kwargs."""
    headers = cast(dict[str, str] | None, kw.get("headers")) or {}
    return {k.lower(): v for k, v in headers.items()}


def const_curl_session(stub: object) -> Callable[..., object]:
    """Return a ``curl_session`` replacement that always returns ``stub`` (typed).

    Args:
      stub: The session stand-in every call hands back.

    Returns:
      factory: Accepts ``curl_session``'s arguments and returns ``stub``.

    """

    def factory(*_args: object, **_kwargs: object) -> object:
        return stub

    return factory


class StubCookie:
    """A minimal jar entry: just a name/value, enough for the pooled-path tests."""

    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value


class StubCookies:
    """Minimal curl-cookies stand-in: a recording jar plus a ``set`` that stores."""

    def __init__(self) -> None:
        self.jar: list[Any] = []

    def set(
        self,
        name: str,
        value: str,
        *,
        domain: str = "",
        path: str = "/",
        secure: bool = False,
    ) -> None:
        """Set a stub response."""
        del domain, path, secure
        self.jar = [c for c in self.jar if getattr(c, "name", None) != name]
        self.jar.append(StubCookie(name, value))


class StubSession:
    """A pooled-Session stand-in whose request delegates to the module-level.

    ``curl_cffi.requests.request`` -- so one ``patch("curl_cffi.requests.request")``
    intercepts both the identity (session) and keyless paths.
    """

    def __init__(self) -> None:
        self.cookies = StubCookies()

    def request(self, *args: Any, **kwargs: Any) -> cc_requests.Response:  # noqa: ANN401 -- forwarded verbatim to curl_cffi's request.
        """Perform one request.

        Args:
          *args: Positional arguments of ``curl_cffi.requests.request``.
          **kwargs: Keyword arguments of ``curl_cffi.requests.request``.

        Returns:
          response: The response ``curl_cffi`` produced.

        """
        return cc_requests.request(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType] -- curl_cffi's **RequestParams TypedDict is unstubbed

    def close(self) -> None:
        """Release held resources."""


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
