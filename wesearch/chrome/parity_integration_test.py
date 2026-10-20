r"""Live parity test: our fetch request must match a real Chrome's.

Chrome is the oracle. A real headless Chrome and :func:`wesearch.fetch` --
on BOTH its backends, the curl_cffi impersonation path and the stdlib reference
path -- request a loopback header-echo endpoint that records the request headers
it received, in wire order. We assert every one of our requests carries the same
headers in the same order as Chrome's -- the definition of "indistinguishable
from Chrome". The test fails the moment a backend diverges, keeping both honest
as Chrome evolves.

The endpoint is the ``oracle`` fixture from this directory's conftest: a
self-signed loopback HTTPS server that records each request's ordered headers
server-side, so the comparison never depends on a third-party echo host whose
certificate can expire. Chrome reaches it with ``--ignore-certificate-errors``;
the fixture points both fetch backends' CA variables at its certificate.

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

from typing import TYPE_CHECKING

from curl_cffi.requests.impersonate import REAL_TARGET_MAP

import pytest

from wesearch.chrome.capture import chrome_available, drive_chrome
from wesearch.chrome.headers import (
    chrome_brands,
    chrome_identity_headers,
    chrome_platform,
    user_agent_platform,
)
from wesearch.chrome.useragents import newest_chrome_major
from wesearch.fetch import (
    ContentParams,
    PolicyParams,
    RequestParams,
    RetryParams,
    Transport,
    fetch,
)
from wesearch.fetch.transport.zendriver import BrowserUnavailableError
from wesearch.fetch.types import FetchSession
from wesearch.profile import Profile, ProfileStore


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from wesearch.chrome.conftest import EchoOracle


# A documentation address (RFC 5737): a preset egress skips the live IP probe.
_EGRESS = "203.0.113.9"

# curl_cffi injects the HTTP/2-only Priority header on HTTP/1.1 connections; a
# real Chrome does not. Dropped from the curl comparison (see module docstring).
_HTTP2_ONLY = frozenset({"priority"})

# Compare navigation values exactly; drive_chrome pins off the field-trial
# features that change them across builds.
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

# The version each transport claims comes from its own source (curl's from the
# target curl_cffi impersonates, stdlib's from the User-Agent pool), so it is
# compared to that source; the OS is compared to the local Chrome.
_IDENTITY_VALUES = ("sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform", "user-agent")

_needs_chrome = pytest.mark.skipif(
    not chrome_available(),
    reason="No unconfined Chrome binary on PATH.",
)


@pytest.fixture
def session(tmp_path: Path) -> Iterator[FetchSession]:
    """Return a session with a preset egress and an empty cookie store."""
    opened = FetchSession(
        profiles=ProfileStore(base_dir=tmp_path / "profiles"),
        egress=_EGRESS,
    )
    try:
        yield opened
    finally:
        opened.close()


@pytest.mark.browser_chrome
@_needs_chrome
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("curl", id="curl"),
        pytest.param("stdlib", id="stdlib"),
    ],
)
def test_fetch_wire_request_matches_chrome(
    backend: Transport,
    oracle: EchoOracle,
    session: FetchSession,
) -> None:
    """Our wire request is a real Chrome's: headers, order, intent, and OS.

    Drives the public :func:`wesearch.fetch.fetch` API exactly as a caller
    would, then compares against a live headless Chrome on this host:

    - header names and order equal Chrome's, modulo curl's HTTP/1.1 ``Priority``;
    - navigation-intent values equal Chrome's;
    - ``sec-ch-ua-platform`` equals Chrome's;
    - User-Agent and client hints name the version this transport presents,
      with the brand list Chrome derives for it.
    """
    omit = _HTTP2_ONLY if backend == "curl" else frozenset[str]()
    chrome = _drop(_chrome_headers(oracle), omit)
    chrome_lines = _drop_lines(oracle.captured_lines(), omit)
    assert chrome, "Chrome sent no request to the oracle; nothing to compare."
    # allow_private_hosts=True: the oracle is a loopback server, which the default
    # SSRF check refuses before any wire bytes exist. The test authored this
    # URL, so declaring it is exactly true -- and is the seam that keeps the
    # check real for every other caller instead of weakening it.
    fetch(
        oracle.url,
        request=RequestParams(
            policy=PolicyParams(transport=backend, allow_private_hosts=True),
        ),
        session=session,
    )
    ours = _drop(oracle.captured(), omit)
    assert ours == chrome, (
        f"wire header set/order diverged from Chrome (transport={backend}).\n"
        f"  chrome: {chrome}\n  ours:   {ours}"
    )
    # Names alone let a header carry the wrong value and still pass.
    # Codec features are pinned by drive_chrome for the exact comparison.
    ours_lines = _drop_lines(oracle.captured_lines(), omit)
    for name in _INTENT_VALUES:
        assert _value(ours_lines, name) == _value(chrome_lines, name), (
            f"{name} value diverged from Chrome (transport={backend}).\n"
            f"  chrome: {_value(chrome_lines, name)}\n"
            f"  ours:   {_value(ours_lines, name)}"
        )
    _assert_identity_matches(ours_lines, chrome_lines, backend=backend)


@pytest.mark.browser_chrome
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("curl", id="curl"),
        pytest.param("stdlib", id="stdlib"),
    ],
)
def test_repeat_requests_send_the_same_wire_headers(
    backend: Transport,
    oracle: EchoOracle,
    session: FetchSession,
) -> None:
    """A second request to an origin presents the identity the first did."""
    request = RequestParams(
        policy=PolicyParams(transport=backend, allow_private_hosts=True),
    )
    fetch(oracle.url, request=request, session=session)
    first = oracle.captured()
    fetch(oracle.url, request=request, session=session)
    assert oracle.captured() == first


@pytest.mark.browser_chrome
def test_stored_cookie_is_not_duplicated_on_wire(
    oracle: EchoOracle,
    session: FetchSession,
) -> None:
    # A caller cookie whose name the stored profile already holds must NOT be
    # sent twice (stored value + caller value). A real browser sends exactly one
    # value per cookie name; two is a bot tell.
    session.profiles.save(
        _EGRESS,
        "localhost",
        Profile(cookies={"CONSENT": "SERVERSET"}),
    )
    fetch(
        oracle.url,
        request=RequestParams(
            content=ContentParams(cookies={"CONSENT": "YES+"}),
            policy=PolicyParams(transport="curl", allow_private_hosts=True),
        ),
        session=session,
    )
    joined = " ".join(_cookie_lines(oracle.captured_lines()))
    assert joined.count("CONSENT=") == 1, f"CONSENT sent more than once: {joined}"
    assert "CONSENT=YES+" in joined, f"the caller's cookie lost: {joined}"


@pytest.mark.browser_chrome
def test_case_variant_cookie_header_not_duplicated_on_wire(
    oracle: EchoOracle,
    session: FetchSession,
) -> None:
    # A caller lowercase headers={"cookie":...} plus a cookies= param must not
    # produce two Cookie header lines on the wire.
    fetch(
        oracle.url,
        request=RequestParams(
            content=ContentParams(headers={"cookie": "a=1"}, cookies={"b": "2"}),
            policy=PolicyParams(transport="curl", allow_private_hosts=True),
        ),
        session=session,
    )
    lines = _cookie_lines(oracle.captured_lines())
    assert len(lines) == 1, f"expected one Cookie header line, got: {lines}"


@pytest.mark.browser_chrome
@_needs_chrome
def test_browser_backend_fetches_live_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in browser backend renders a live page headless.

    Drives the public API with ``transport="zendriver"`` against a real site
    through a throwaway profile dir, proving the zendriver backend launches
    Chrome and returns rendered bytes. Chrome-gated; skipped without a binary.

    The throwaway profile is redirected through ``XDG_DATA_HOME`` -- the seam
    ``data_dir`` actually reads -- so the operator's real logged-in Chrome
    profile is never touched.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    body: bytes | None = None
    try:
        body = fetch(
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
    # No per-test Chrome teardown. The autouse module fixture in
    # wesearch/conftest.py owns that, and closing here would only protect THIS
    # test -- which is how 378 browsers from the sibling search tests leaked.
    assert body is not None
    # Structure, not content: the page belongs to a third party and its wording
    # is not this backend's contract. Rendered HTML with a body is.
    assert b"<html" in body.lower()
    assert b"</body>" in body.lower()


# Read from the oracle, the same server-side record the fetch backends are judged by, so
# both sides of the comparison come from one observer. Call before issuing our own
# request: ``captured()`` returns only the most recent, and the per-test fixture
# guarantees no earlier capture can be standing.
#
# A Chrome killed at the timeout AFTER navigating still leaves a complete record, so one
# retry covers only the hang that produced nothing -- otherwise a slow CI runner fails a
# test about header order.
def _chrome_headers(oracle: EchoOracle) -> tuple[str, ...]:
    """Return the ordered header names a real Chrome sent to the oracle."""
    if _drive(oracle) or oracle.captured():
        return oracle.captured()
    _drive(oracle)
    return oracle.captured()


def _drive(oracle: EchoOracle) -> bool:
    """Drive Chrome at the oracle once; whether it finished without a kill."""
    # disable_sandbox: hosted runners execute as root, where Chrome refuses to
    # start sandboxed. Scoped to this loopback oracle, never a real host.
    return not drive_chrome(
        oracle.url,
        ignore_certificate_errors=True,
        disable_sandbox=True,
    )


def _assert_identity_matches(
    ours: tuple[str, ...],
    chrome: tuple[str, ...],
    *,
    backend: str,
) -> None:
    """Our identity names this host's OS and the version our transport presents."""
    platform = chrome_platform("chrome")
    major = (
        int(REAL_TARGET_MAP["chrome"].removeprefix("chrome"))
        if backend == "curl"
        else newest_chrome_major(platform)
    )
    expected = {
        name.lower(): value
        for name, value in chrome_identity_headers(
            major=major,
            platform=platform,
        ).items()
    }
    for name in _IDENTITY_VALUES:
        assert _value(ours, name) == expected[name], (
            f"{name} diverged from the {backend} identity.\n"
            f"  expected: {expected[name]}\n  ours:     {_value(ours, name)}"
        )
    assert _value(ours, "sec-ch-ua-platform") == _value(chrome, "sec-ch-ua-platform")
    chrome_ua = _value(chrome, "user-agent")
    assert user_agent_platform(chrome_ua) == platform
    chrome_major = int(chrome_ua.split("Chrome/", 1)[1].split(".", 1)[0])
    assert _value(chrome, "sec-ch-ua") == chrome_brands(chrome_major)


def _drop_lines(lines: tuple[str, ...], omit: frozenset[str]) -> tuple[str, ...]:
    """Raw header lines minus the ones :func:`_drop` removes by name."""
    skip = {"host", "connection"} | omit
    return tuple(
        line for line in lines if line.split(":", 1)[0].strip().lower() not in skip
    )


def _value(lines: tuple[str, ...], name: str) -> str:
    """Return the value of header ``name`` among raw lines, or ``""`` when absent."""
    prefix = f"{name}:"
    return next(
        (
            line.split(":", 1)[1].strip()
            for line in lines
            if line.lower().startswith(prefix)
        ),
        "",
    )


# ``host`` is HTTP/1.1's mandatory analog of the HTTP/2 ``:authority`` pseudo-header and
# ``connection`` is an HTTP/1.1 hop-by-hop control header; neither is part of the
# browser identity being compared.
def _drop(names: tuple[str, ...], omit: frozenset[str]) -> tuple[str, ...]:
    """Header names minus ``host``, ``connection``, and any in ``omit``."""
    skip = {"host", "connection"} | omit
    return tuple(name for name in names if name not in skip)


def _cookie_lines(lines: tuple[str, ...]) -> list[str]:
    """Return the raw ``cookie:`` header lines from captured request lines."""
    return [line for line in lines if line.lower().startswith("cookie")]


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
