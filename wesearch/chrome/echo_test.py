"""Unit tests for the loopback echo oracle."""

from __future__ import annotations

from pathlib import Path

import http.client
import socket
import ssl
import struct
import time

from wesearch.chrome.echo import (
    EchoOracle,
    _header_lines,
    _header_names,
    _requests_root,
    self_signed_localhost_cert,
)


class TestHeaderParsing:
    def test_header_names_are_lowercased_in_order(self) -> None:
        request = "GET / HTTP/1.1\r\nHost: x\r\nUser-Agent: C\r\nAccept: */*"
        assert _header_names(request) == ("host", "user-agent", "accept")

    def test_header_lines_are_verbatim(self) -> None:
        request = "GET / HTTP/1.1\r\nHost: x\r\nCookie: a=1; b=2"
        assert _header_lines(request) == ("Host: x", "Cookie: a=1; b=2")

    def test_requests_root_only_for_root_path(self) -> None:
        assert _requests_root("GET / HTTP/1.1\r\n")
        assert not _requests_root("GET /favicon.ico HTTP/1.1\r\n")
        assert not _requests_root("garbage")


class TestSelfSignedCert:
    def test_writes_pem_cert_and_key(self, tmp_path: Path) -> None:
        cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
        self_signed_localhost_cert(cert, key)
        assert cert.read_bytes().startswith(b"-----BEGIN CERTIFICATE-----")
        assert b"PRIVATE KEY" in key.read_bytes()


class TestEchoOracle:
    def test_captures_ordered_headers_of_a_live_request(self) -> None:
        with EchoOracle() as oracle:
            context = ssl.create_default_context(cafile=str(oracle.ca_path))
            port = int(oracle.url.rsplit(":", 1)[1].rstrip("/"))
            conn = http.client.HTTPSConnection(
                "localhost", port, timeout=5, context=context
            )
            conn.request("GET", "/", headers={"User-Agent": "probe", "Accept": "*/*"})
            conn.getresponse().read()
            conn.close()
            names = oracle.captured()
        assert "user-agent" in names
        assert names.index("host") < names.index("user-agent")

    def test_ignores_non_root_requests(self) -> None:
        with EchoOracle() as oracle:
            context = ssl.create_default_context(cafile=str(oracle.ca_path))
            port = int(oracle.url.rsplit(":", 1)[1].rstrip("/"))
            conn = http.client.HTTPSConnection(
                "localhost", port, timeout=5, context=context
            )
            conn.request("GET", "/favicon.ico")
            conn.getresponse().read()
            conn.close()
            assert oracle.captured() == ()


class TestEchoOracleResilience:
    def test_survives_a_client_that_resets_the_connection(self) -> None:
        """An aborted client must not kill the accept loop.

        A client that RSTs before finishing the TLS handshake raises
        ``ConnectionResetError`` -- an ``OSError``, not an ``ssl.SSLError`` --
        out of ``wrap_socket``. Uncaught, it unwinds ``_serve`` and the oracle
        stops accepting: pytest reports ``PytestUnhandledThreadExceptionWarning``
        and every later request in the session hangs until its timeout.
        """
        with EchoOracle() as oracle:
            port = int(oracle.url.rsplit(":", 1)[1].rstrip("/"))
            aborting = socket.socket()
            # SO_LINGER with a zero timeout makes close() send RST, not FIN.
            aborting.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            aborting.connect(("127.0.0.1", port))
            aborting.close()

            context = ssl.create_default_context(cafile=str(oracle.ca_path))
            conn = http.client.HTTPSConnection(
                "localhost", port, timeout=5, context=context
            )
            conn.request("GET", "/", headers={"User-Agent": "probe"})
            conn.getresponse().read()
            conn.close()
            assert "user-agent" in oracle.captured()

    def test_an_idle_client_does_not_block_the_next_one(self) -> None:
        """A connection that stalls mid-request must not wedge the oracle.

        Chrome preconnects: it opens sockets speculatively and leaves them
        idle. Handled serially with no timeout, the first such socket parks the
        accept loop in ``recv`` forever and every subsequent client -- including
        the parity test's own fetch -- blocks until its own timeout.
        """
        with EchoOracle() as oracle:
            port = int(oracle.url.rsplit(":", 1)[1].rstrip("/"))
            context = ssl.create_default_context(cafile=str(oracle.ca_path))
            stalled = context.wrap_socket(
                socket.create_connection(("localhost", port), timeout=5),
                server_hostname="localhost",
            )
            stalled.sendall(b"GET / HTTP")  # A head that never terminates.

            served = http.client.HTTPSConnection(
                "localhost", port, timeout=5, context=context
            )
            served.request("GET", "/", headers={"User-Agent": "probe"})
            served.getresponse().read()
            served.close()
            stalled.close()
            assert "user-agent" in oracle.captured()


class TestEchoOracleShutdown:
    def test_close_returns_promptly(self) -> None:
        """close() must wake the accept loop, not wait out the join timeout.

        ``socket.close()`` alone does NOT interrupt a thread blocked in
        ``accept()`` on Linux: the thread stays parked, ``join(timeout=5.0)``
        burns the full five seconds, and the daemon thread is abandoned. Every
        oracle test paid that 5s. ``shutdown()`` first wakes it immediately.
        """
        oracle = EchoOracle()
        start = time.perf_counter()
        oracle.close()
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"close() took {elapsed:.2f}s; accept loop not woken"


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
