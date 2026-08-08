"""Hermetic fixtures for fetch package tests."""

from pathlib import Path

import importlib
import ipaddress
import socket

import pytest

from wesearch.fetch.common import ValidatedHost
from wesearch.fetch.test_helpers import StubSession
from wesearch.profile import ProfileStore

import wesearch.fetch.curl as curl_mod


fetch_mod = importlib.import_module("wesearch.fetch.fetch")


# One synthesized ``getaddrinfo`` answer. Only the address at ``[4][0]`` is read
# (by ``public_host``); the rest is the shape the stdlib promises.
def _addrinfo(
    ip: str,
) -> list[tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]]:
    """One TCP addrinfo record for ``ip``."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]


@pytest.fixture(autouse=True)
def offline_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer every DNS lookup locally, without weakening the SSRF check.

    Trust is honored by default, so every fetch resolves its host -- and the
    suite is full of hostnames nobody registered (``a.com``, ``walled.example``).
    Left alone those raise DNS errors that look like SSRF refusals, which would
    both make the tests depend on a resolver AND hide the check they exist to
    prove.

    A name answers with a public address, so validation passes; an address
    literal answers as ITSELF, so a test fetching ``127.0.0.1`` still resolves
    to loopback and is still refused under ``untrusted``. Nothing delegates to
    the real resolver, so the suite issues no DNS at all.
    """

    def fake(
        host: str | None, *_args: object, **_kwargs: object
    ) -> list[
        tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]
    ]:
        try:
            ipaddress.ip_address(str(host))
        except ValueError:
            return _addrinfo("93.184.216.34")
        return _addrinfo(str(host))

    monkeypatch.setattr(socket, "getaddrinfo", fake)


@pytest.fixture(autouse=True)
def isolate_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic identity layer: a tmp store and a fixed egress, no network.

    ``fetch`` transparently loads/saves a per-``(egress_ip, domain)`` profile.
    Without isolation these tests would share the real on-disk store (state
    leaking between cases) and call the live egress echo. Point the store at a
    tmp dir and pin the egress so the transport assertions stay deterministic.
    """

    def fixed_egress(*, cache: bool = True, **_kw: object) -> str:
        del cache, _kw
        return "203.0.113.1"

    store = ProfileStore(base_dir=tmp_path)

    def shared(_cls: type[ProfileStore]) -> ProfileStore:
        return store

    monkeypatch.setattr(ProfileStore, "shared", classmethod(shared))
    monkeypatch.setattr(fetch_mod, "egress_ip", fixed_egress)
    monkeypatch.setattr(fetch_mod, "_last_egress_ip", None)

    def stub_session(
        egress: str,
        domain: str,
        impersonate: str,
        *,
        pin: ValidatedHost | None = None,
        port: int = 443,
    ) -> StubSession:
        del egress, domain, impersonate, pin, port
        return StubSession()

    monkeypatch.setattr(fetch_mod, "curl_session", stub_session)
    monkeypatch.setattr(curl_mod, "_curl_sessions", {})
