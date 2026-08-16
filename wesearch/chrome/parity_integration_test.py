r"""Live parity test: our fetch request must match a real Chrome's.

Chrome is the oracle. A real headless Chrome and :func:`wesearch.fetch` --
on BOTH its backends, the curl_cffi impersonation path and the stdlib reference
path -- request a loopback header-echo endpoint that records the request headers
it received, in wire order. We assert every one of our requests carries the same
headers in the same order as Chrome's -- the definition of "indistinguishable
from Chrome". The test fails the moment a backend diverges, keeping both honest
as Chrome evolves.

The endpoint is :class:`wesearch.chrome.echo.EchoOracle`: a self-signed
loopback HTTPS server that records each request's ordered headers server-side,
so the comparison never depends on a third-party echo host whose certificate can
expire. Chrome reaches it with ``--ignore-certificate-errors``; the fetch
backends trust its ``ca_path`` (``SSL_CERT_FILE`` for stdlib, curl's own CA
bundle honors it too).

The oracle offers ALPN ``http/1.1`` only -- the one protocol every backend
speaks. One wire consequence: real Chrome omits the HTTP/2-only ``Priority``
header on HTTP/1.1. The stdlib backend reproduces that exactly; the curl backend
sends ``Priority`` on HTTP/1.1 regardless (a known curl_cffi bug, unreachable by
real HTTP/2 origins), so the curl comparison drops it.

Requires a Chrome binary on PATH (skipped without one so CI stays green). Run::

    uv run --frozen pytest -m integration \\
        wesearch/chrome/parity_integration_test.py
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import importlib

import pytest

from wesearch.chrome.capture import chrome_available, drive_chrome
from wesearch.chrome.echo import EchoOracle
from wesearch.fetch import (
    ContentParams,
    FetchSession,
    PolicyParams,
    RequestParams,
    RetryParams,
    Transport,
    fetch,
)
from wesearch.fetch.transport.curl import close_curl_sessions_except
from wesearch.fetch.transport.zendriver import (
    BrowserUnavailableError,
    shutdown_browsers,
)


if TYPE_CHECKING:
    from curl_cffi.requests import Response


# The MODULE, not the `fetch` function of the same name imported above: these
# tests patch module attributes (`egress_ip`, `curl_session`). A plain
# `import wesearch.fetch.fetch` binds the package's re-exported function
# here, so `patch.object` would target the function and fail.
fetch_mod = importlib.import_module("wesearch.fetch.fetch")

# curl_cffi injects the HTTP/2-only Priority header on HTTP/1.1 connections; a
# real Chrome does not. Dropped from the curl comparison (see module docstring).
_HTTP2_ONLY = frozenset({"priority"})

# Header values that must match the local Chrome EXACTLY: they encode the
# navigation intent, not the browser build, so a real Chrome sends these
# identically regardless of version or platform. Measured against a live
# Chrome 151 on Linux vs our impersonated Chrome 146 on macOS -- every one
# agreed. A divergence here is drift, not configuration.
_INTENT_VALUES = (
    "accept",
    "accept-encoding",
    "accept-language",
    "sec-fetch-site",
    "sec-fetch-mode",
    "sec-fetch-user",
    "sec-fetch-dest",
    "upgrade-insecure-requests",
)

# The build-identity values (user-agent, sec-ch-ua, sec-ch-ua-platform) are
# deliberately NOT compared to the local Chrome: the harness browser is
# whatever the host installed (and announces HeadlessChrome/), while we
# impersonate a fixed target. Pinning them would fail on browser upgrades
# instead of on drift. What must hold is that OUR values agree with EACH
# OTHER -- a UA naming one OS beside a platform hint naming another is the
# incoherence that gives a bot away.
_UA_OS_TOKEN = {
    "Windows": "Windows NT",
    "Linux": "X11; Linux",
    "macOS": "Macintosh",
    "Android": "Android",
}

_needs_chrome = pytest.mark.skipif(
    not chrome_available(), reason="No Chrome binary on PATH."
)


@pytest.fixture
def oracle() -> Iterator[EchoOracle]:
    """A running loopback echo oracle whose CA the fetch backends trust."""
    # A pooled curl Session keys on eTLD+1 (``localhost``), so one cached under a
    # prior oracle's CA would be reused against this oracle's -- and fail TLS.
    # No egress key matches "", so this drops EVERY pooled session and each
    # oracle gets a fresh, correctly-trusted one.
    close_curl_sessions_except("")
    # stdlib (ssl) honors SSL_CERT_FILE; curl_cffi (BoringSSL) honors
    # CURL_CA_BUNDLE. Both must trust the oracle's self-signed CA.
    with (
        EchoOracle() as running,
        patch.dict(
            "os.environ",
            {
                "SSL_CERT_FILE": str(running.ca_path),
                "CURL_CA_BUNDLE": str(running.ca_path),
            },
        ),
    ):
        yield running


@pytest.mark.integration
@_needs_chrome
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("curl", id="curl"),
        pytest.param("stdlib", id="stdlib"),
    ],
)
def test_fetch_wire_request_matches_chrome(
    backend: Transport, oracle: EchoOracle
) -> None:
    """Our wire request has the same headers, in the same order, as Chrome's.

    Drives the public :func:`wesearch.fetch.fetch` API exactly as a caller
    would, proving the API reproduces a real Chrome request. The curl path
    (curl_cffi impersonation owns the fingerprint) and the stdlib path (the same
    Chrome header set hand-built by ``chrome_navigation_headers``) must each
    match Chrome's ordered wire headers, modulo the curl HTTP/1.1 ``Priority``
    quirk.
    """
    omit = _HTTP2_ONLY if backend == "curl" else frozenset[str]()
    chrome = _drop(_chrome_headers(oracle), omit)
    chrome_lines = _drop_lines(oracle.captured_lines(), omit)
    assert chrome, "Chrome sent no request to the oracle; nothing to compare."
    # trust="internal": the oracle is a loopback server, which the default
    # ``untrusted`` SSRF check refuses before any wire bytes exist. The test
    # authored this URL, so declaring it is exactly true -- and is the seam that
    # keeps the check real for every other caller instead of weakening it.
    fetch(
        oracle.url,
        request=RequestParams(policy=PolicyParams(transport=backend, trust="internal")),
    )
    ours = _drop(oracle.captured(), omit)
    assert ours == chrome, (
        f"wire header set/order diverged from Chrome (transport={backend}).\n"
        f"  chrome: {chrome}\n  ours:   {ours}"
    )
    # Names alone let a header carry the wrong VALUE and still pass. The
    # intent-bearing values do not vary by browser build, so they must match
    # the live Chrome exactly.
    ours_lines = _drop_lines(oracle.captured_lines(), omit)
    for name in _INTENT_VALUES:
        assert _value(ours_lines, name) == _value(chrome_lines, name), (
            f"{name} value diverged from Chrome (transport={backend}).\n"
            f"  chrome: {_value(chrome_lines, name)}\n"
            f"  ours:   {_value(ours_lines, name)}"
        )
    _assert_identity_is_coherent(ours_lines, backend=backend)


@pytest.mark.integration
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("curl", id="curl"),
        pytest.param("stdlib", id="stdlib"),
    ],
)
def test_session_threads_across_requests(
    backend: Transport, oracle: EchoOracle
) -> None:
    """A threaded session preserves the wire identity across requests.

    Call once, feed the returned session to the next call, and the second
    request reflects what the first learned -- mirroring a real browser. The
    oracle sets no Accept-CH, so a reused session must send the same wire headers
    as a fresh one.
    """
    request = RequestParams(policy=PolicyParams(transport=backend, trust="internal"))
    _, session = fetch(oracle.url, request=request)
    assert isinstance(session, FetchSession)
    fetch(oracle.url, session=session, request=request)
    reused = oracle.captured()
    fetch(oracle.url, request=request)
    assert reused == oracle.captured()


@pytest.mark.integration
def test_pooled_session_does_not_duplicate_cookie_on_wire(oracle: EchoOracle) -> None:
    # On the pooled-curl path, a caller cookie whose name the session jar already
    # holds must NOT be sent twice (jar value + header value). A real browser
    # sends exactly one value per cookie name; two is a bot tell.
    from curl_cffi import requests  # noqa: PLC0415 -- test-only, off collection.

    pooled: requests.Session[Response] = requests.Session(impersonate="chrome")
    pooled.cookies.set("CONSENT", "SERVERSET", domain="localhost")

    def _pooled(*_args: object, **_kwargs: object) -> requests.Session[Response]:
        return pooled

    with (
        patch.object(fetch_mod, "egress_ip", return_value="9.9.9.9"),
        patch.object(fetch_mod, "curl_session", _pooled),
        patch("wesearch.fetch.fetch.ProfileStore"),
    ):
        fetch(
            oracle.url,
            request=RequestParams(
                content=ContentParams(cookies={"CONSENT": "YES+"}),
                policy=PolicyParams(trust="internal"),
            ),
        )
    joined = " ".join(_cookie_lines(oracle.captured_lines()))
    assert joined.count("CONSENT=") == 1, f"CONSENT sent more than once: {joined}"


@pytest.mark.integration
def test_case_variant_cookie_header_not_duplicated_on_wire(oracle: EchoOracle) -> None:
    # A caller lowercase headers={"cookie":...} plus a cookies= param must not
    # produce two Cookie header lines on the wire.
    with patch.object(fetch_mod, "egress_ip", return_value=None):
        fetch(
            oracle.url,
            request=RequestParams(
                content=ContentParams(headers={"cookie": "a=1"}, cookies={"b": "2"}),
                policy=PolicyParams(trust="internal"),
            ),
        )
    lines = _cookie_lines(oracle.captured_lines())
    assert len(lines) == 1, f"expected one Cookie header line, got: {lines}"


@pytest.mark.integration
@_needs_chrome
def test_browser_backend_fetches_live_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opt-in browser backend renders a live page headless and warms cookies.

    Drives the public API with ``transport="zendriver"`` against a real site
    through a throwaway profile dir, proving the zendriver backend launches
    Chrome, returns rendered bytes, and threads the acquired cookies into the
    returned session. Chrome-gated; skipped without a binary.

    The throwaway profile is redirected through ``XDG_DATA_HOME`` -- the seam
    ``data_dir`` actually reads -- so the operator's real logged-in Chrome
    profile is never touched.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    try:
        body, session = fetch(
            "https://example.com/",
            request=RequestParams(
                retry=RetryParams(timeout_sec=30.0),
                policy=PolicyParams(transport="zendriver"),
            ),
        )
    except BrowserUnavailableError as error:
        # A Chrome on PATH still cannot always be DRIVEN: hosted runners refuse
        # the DevTools attach that the capture harness above never needs. That
        # is a capability gap, so `_needs_chrome` alone does not cover this test.
        pytest.skip(f"browser subsystem unavailable: {error}")
    finally:
        shutdown_browsers()
    # Structure, not content: the page belongs to a third party and its wording
    # is not this backend's contract. Rendered HTML with a body is.
    assert b"<html" in body.lower()
    assert b"</body>" in body.lower()
    assert isinstance(session, FetchSession)


def _chrome_headers(oracle: EchoOracle) -> tuple[str, ...]:
    """The ordered header names a real Chrome sent to the oracle.

    Read from the oracle, the same server-side record the fetch backends are
    judged by, so both sides of the comparison come from one observer. Call
    before issuing our own request: ``captured()`` returns only the most recent,
    and the per-test fixture guarantees no earlier capture can be standing.

    A Chrome killed at the timeout AFTER navigating still leaves a complete
    record, so one retry covers only the hang that produced nothing -- otherwise
    a slow CI runner fails a test about header order.
    """
    if _drive(oracle) or oracle.captured():
        return oracle.captured()
    _drive(oracle)
    return oracle.captured()


def _drive(oracle: EchoOracle) -> bool:
    """Drive Chrome at the oracle once; whether it finished without a kill."""
    # disable_sandbox: hosted runners execute as root, where Chrome refuses to
    # start sandboxed. Scoped to this loopback oracle, never a real host.
    return not drive_chrome(
        oracle.url, ignore_certificate_errors=True, disable_sandbox=True
    )


def _assert_identity_is_coherent(lines: tuple[str, ...], *, backend: str) -> None:
    """Our UA, platform hint, and brand version must describe one browser.

    A request whose ``sec-ch-ua-platform`` says one OS while its User-Agent
    names another is provably not a real browser, and no header ORDER check can
    see it -- both headers are present and correctly placed.
    """
    user_agent = _value(lines, "user-agent")
    platform = _value(lines, "sec-ch-ua-platform").strip('"')
    assert platform in _UA_OS_TOKEN, f"unknown platform hint {platform!r}"
    assert _UA_OS_TOKEN[platform] in user_agent, (
        f"sec-ch-ua-platform says {platform!r} but the User-Agent does not "
        f"(transport={backend}).\n  user-agent: {user_agent}"
    )
    major = user_agent.split("Chrome/", 1)[-1].split(".", 1)[0]
    assert f'v="{major}"' in _value(lines, "sec-ch-ua"), (
        f"sec-ch-ua brand version disagrees with the User-Agent's Chrome/{major} "
        f"(transport={backend}).\n  sec-ch-ua: {_value(lines, 'sec-ch-ua')}"
    )


def _drop_lines(lines: tuple[str, ...], omit: frozenset[str]) -> tuple[str, ...]:
    """Raw header lines minus the ones :func:`_drop` removes by name."""
    skip = {"host", "connection"} | omit
    return tuple(
        line for line in lines if line.split(":", 1)[0].strip().lower() not in skip
    )


def _value(lines: tuple[str, ...], name: str) -> str:
    """The value of header ``name`` among raw lines, or ``""`` when absent."""
    prefix = f"{name}:"
    return next(
        (
            line.split(":", 1)[1].strip()
            for line in lines
            if line.lower().startswith(prefix)
        ),
        "",
    )


def _drop(names: tuple[str, ...], omit: frozenset[str]) -> tuple[str, ...]:
    """Header names minus ``host``, ``connection``, and any in ``omit``.

    ``host`` is HTTP/1.1's mandatory analog of the HTTP/2 ``:authority``
    pseudo-header and ``connection`` is an HTTP/1.1 hop-by-hop control header;
    neither is part of the browser identity being compared.
    """
    skip = {"host", "connection"} | omit
    return tuple(name for name in names if name not in skip)


def _cookie_lines(lines: tuple[str, ...]) -> list[str]:
    """The raw ``cookie:`` header lines from captured request lines."""
    return [line for line in lines if line.lower().startswith("cookie")]


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
