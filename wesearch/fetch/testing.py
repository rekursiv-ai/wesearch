"""Shared test doubles for fetch transport tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import io

from curl_cffi import requests as cc_requests

from wesearch.lib.custom_json import DictCodec


try:
    from compression import zstd
except ImportError:
    zstd = None
if TYPE_CHECKING:
    from collections.abc import Callable

    import zstandard  # pyright: ignore[reportMissingModuleSource] -- Stubbed in typings/; the wheel is only installed on <3.14.
else:
    from wrapt import lazy_import

    zstandard = lazy_import("zstandard")


def zstd_compress(data: bytes, *, streaming: bool = False) -> bytes:
    """Compress ``data`` as one zstd frame, via stdlib on 3.14+, else the wheel.

    Args:
      data: Bytes to compress.
      streaming: Emit a streaming frame, which omits the content size from the
        frame header -- the shape Cloudflare serves.

    Returns:
      frame: The compressed frame.

    """
    if zstd is not None:
        if not streaming:
            return zstd.compress(data)
        compressor = zstd.ZstdCompressor()
        return compressor.compress(data) + compressor.flush()
    if not streaming:
        return zstandard.ZstdCompressor().compress(data)
    buf = io.BytesIO()
    with zstandard.ZstdCompressor().stream_writer(buf, closefd=False) as w:
        _ = w.write(data)
    return buf.getvalue()


def lower_headers(kw: dict[str, object]) -> dict[str, str]:
    """Lower-cased request headers from a curl ``request`` mock's kwargs.

    Args:
      kw: Kw.

    Returns:
      result: The dict[str, str].

    """
    headers = DictCodec.coerce(kw.get("headers"), str)
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
        self.jar: list[StubCookie] = []

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

    def request(self, *args: object, **kwargs: object) -> cc_requests.Response:
        """Perform one request.

        Args:
          *args: Positional arguments of ``curl_cffi.requests.request``.
          **kwargs: Keyword arguments of ``curl_cffi.requests.request``.

        Returns:
          response: The response ``curl_cffi`` produced.

        """
        request = cast(_CurlRequest, cc_requests.request)
        return request(*args, **kwargs)

    def close(self) -> None:
        """Release held resources."""


class _CurlRequest(Protocol):
    """The curl request slice used by the session test double."""

    def __call__(self, *args: object, **kwargs: object) -> cc_requests.Response: ...


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
