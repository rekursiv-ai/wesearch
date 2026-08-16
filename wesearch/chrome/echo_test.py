"""Unit tests for the loopback echo oracle."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import http.client
import socket
import ssl
import struct
import tempfile
import threading
import time

import pytest

from wesearch.chrome import echo
from wesearch.chrome.echo import (
    EchoOracle,
    _header_lines,
    _header_names,
    _read_head,
    _requests_root,
    self_signed_localhost_cert,
)


class TestHeaderParsing:
    def test_header_names_are_lowercased_in_order(self) -> None:
        request = "GET / HTTP/1.1\r\nHost: x\r\nUser-Agent: C\r\nAccept: */*\r\n\r\n"
        assert _header_names(request) == ("host", "user-agent", "accept")

    def test_header_lines_are_verbatim(self) -> None:
        request = "GET / HTTP/1.1\r\nHost: x\r\nCookie: a=1; b=2\r\n\r\n"
        assert _header_lines(request) == ("Host: x", "Cookie: a=1; b=2")

    def test_requests_root_only_for_root_path(self) -> None:
        assert _requests_root("GET / HTTP/1.1\r\n")
        assert not _requests_root("GET /favicon.ico HTTP/1.1\r\n")
        assert not _requests_root("garbage")

    def test_body_lines_are_not_header_lines(self) -> None:
        # A colon-bearing body line must never reach captured_lines(): the
        # parity suite asserts on the NUMBER of Cookie lines, so a forged one
        # would defeat the duplicate-cookie checks.
        request = "POST / HTTP/1.1\r\nHost: x\r\n\r\nCookie: forged=1"
        assert _header_lines(request) == ("Host: x",)


class TestReadHead:
    """``_read_head`` must distinguish a finished head from a truncated one."""

    def test_a_head_that_never_terminated_reads_as_nothing(self) -> None:
        """A peer that hangs up mid-head sent no request, not a partial one.

        ``GET / HTTP`` -- what a Chrome preconnect leaves on the wire -- has a
        request line targeting ``/`` and no header block, so returning the
        partial bytes made it capture as a request carrying ZERO headers.
        ``captured()`` reports the LAST capture, so that empty record replaced
        the real one and the parity assertion read ``()``.

        Reaching EOF is how it surfaces where OpenSSL delivers a ``close_notify``
        as a clean end-of-stream; where it raises instead, ``_handle`` discards
        the connection and never reaches this. Only the former is reproducible
        in-process, and it is the one that broke CI.
        """
        assert _read_head(_EofAfter(b"GET / HTTP")) == ""

    def test_a_terminated_head_reads_through(self) -> None:
        head = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"
        assert _read_head(_EofAfter(head)) == head.decode()

    def test_body_bytes_arriving_with_the_head_are_not_returned(self) -> None:
        # One recv() can deliver head AND body. Returning the body too let a
        # body line containing ":" parse as a header, so a request could forge
        # the very Cookie lines the parity suite asserts on.
        head = b"POST / HTTP/1.1\r\nHost: x\r\n\r\nCookie: forged=1"
        assert _read_head(_EofAfter(head)) == "POST / HTTP/1.1\r\nHost: x\r\n\r\n"

    def test_max_bytes_caps_the_read(self) -> None:
        # The cap bounded the loop, not the read: recv(4096) overshot it by up
        # to 4095 bytes, so a small cap barely constrained anything.
        head = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"
        assert _read_head(_EofAfter(head), max_bytes=4) == ""


class _EofAfter:
    """A reader yielding ``chunks``, then a clean EOF forever.

    Honors ``bufsize`` like a real socket: a reader that ignored it could not
    show whether the caller's byte cap reaches the read at all.
    """

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = list(chunks)

    def recv(self, bufsize: int, /) -> bytes:
        if not self._chunks:
            return b""
        chunk = self._chunks[0]
        if len(chunk) <= bufsize:
            return self._chunks.pop(0)
        self._chunks[0] = chunk[bufsize:]
        return chunk[:bufsize]


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
            conn = http.client.HTTPSConnection(
                "localhost", oracle.port, timeout=5, context=context
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
            conn = http.client.HTTPSConnection(
                "localhost", oracle.port, timeout=5, context=context
            )
            conn.request("GET", "/favicon.ico")
            conn.getresponse().read()
            conn.close()
            assert oracle.captured() == ()


class TestEchoOracleResilience:
    def test_a_reset_client_raises_nothing_on_a_handler_thread(self) -> None:
        """An aborted client must not leave an exception on a handler thread.

        A client that RSTs before finishing the TLS handshake raises
        ``ConnectionResetError`` -- an ``OSError``, not an ``ssl.SSLError`` --
        out of ``wrap_socket``. Since handling moved to one thread per
        connection, an escape kills only that worker, so asserting "a later
        request still succeeds" passes with or without the guard. What still
        distinguishes them is whether the worker died with an exception.
        """
        escaped: list[BaseException | None] = []

        def record(args: threading.ExceptHookArgs) -> None:
            escaped.append(args.exc_value)

        with patch.object(threading, "excepthook", record), EchoOracle() as oracle:
            aborting = socket.socket()
            # SO_LINGER with a zero timeout makes close() send RST, not FIN.
            aborting.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            aborting.connect(("127.0.0.1", oracle.port))
            aborting.close()

            context = ssl.create_default_context(cafile=str(oracle.ca_path))
            conn = http.client.HTTPSConnection(
                "localhost", oracle.port, timeout=5, context=context
            )
            conn.request("GET", "/", headers={"User-Agent": "probe"})
            conn.getresponse().read()
            conn.close()
            assert "user-agent" in oracle.captured()
        assert not escaped, f"handler thread raised: {escaped}"

    def test_an_idle_client_does_not_block_the_next_one(self) -> None:
        """A connection that stalls mid-request must not wedge the oracle.

        Chrome preconnects: it opens sockets speculatively and leaves them
        idle. Handled serially with no timeout, the first such socket parks the
        accept loop in ``recv`` forever and every subsequent client -- including
        the parity test's own fetch -- blocks until its own timeout.
        """
        with EchoOracle() as oracle:
            context = ssl.create_default_context(cafile=str(oracle.ca_path))
            stalled = context.wrap_socket(
                socket.create_connection(("localhost", oracle.port), timeout=5),
                server_hostname="localhost",
            )
            stalled.sendall(b"GET / HTTP")  # A head that never terminates.

            served = http.client.HTTPSConnection(
                "localhost", oracle.port, timeout=5, context=context
            )
            served.request("GET", "/", headers={"User-Agent": "probe"})
            served.getresponse().read()
            served.close()
            stalled.close()
            assert "user-agent" in oracle.captured()


class TestEchoOracleCleanup:
    """Every resource the oracle acquires must be released by ``close()``."""

    def test_close_removes_the_certificate_directory(self) -> None:
        oracle = EchoOracle()
        directory = oracle.ca_path.parent
        assert directory.exists()
        oracle.close()
        assert not directory.exists(), f"leaked {directory}"

    def test_failed_construction_leaves_no_directory(self) -> None:
        # __init__ acquires a temp dir, then a socket, then a thread. A raise
        # part-way through stranded everything acquired so far: the caller
        # never gets an object, so close() is unreachable.
        before = set(Path(tempfile.gettempdir()).glob("echo-oracle-*"))
        with (
            patch.object(echo, "self_signed_localhost_cert", side_effect=OSError("x")),
            pytest.raises(OSError, match="x"),
        ):
            EchoOracle()
        assert set(Path(tempfile.gettempdir()).glob("echo-oracle-*")) == before

    def test_close_joins_a_stalled_handler_thread(self) -> None:
        # close() joined the accept thread only, so a handler could still be
        # inside _handle -- holding a client socket -- after teardown returned.
        oracle = EchoOracle(client_timeout_sec=0.2)
        context = ssl.create_default_context(cafile=str(oracle.ca_path))
        stalled = context.wrap_socket(
            socket.create_connection(("localhost", oracle.port), timeout=5),
            server_hostname="localhost",
        )
        stalled.sendall(b"GET / HTTP")  # A head that never terminates.
        time.sleep(0.1)
        oracle.close()
        assert not [t for t in threading.enumerate() if t.name.startswith("echo-")]
        stalled.close()


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
