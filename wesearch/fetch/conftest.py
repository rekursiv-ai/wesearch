"""Hermetic fixtures for fetch package tests."""

import ipaddress
import socket

import pytest


@pytest.fixture(autouse=True)
def offline_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer every DNS lookup locally, without weakening the private-host check.

    A name answers with a public address, so the check passes; an IP literal
    answers as itself, so ``127.0.0.1`` is still refused unless
    ``allow_private_hosts`` is set.
    """

    def fake(
        host: str | None,
        *_args: object,
        **_kwargs: object,
    ) -> list[
        tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]
    ]:
        try:
            ip = str(ipaddress.ip_address(str(host)))
        except ValueError:
            ip = "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
