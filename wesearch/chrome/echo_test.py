"""Unit tests for the loopback echo oracle."""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta
from functools import partial
from http import client
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import socket
import ssl
import struct
import tempfile
import threading
import time

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

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


if TYPE_CHECKING:
    from collections.abc import Callable


class TestHeaderParsing:
    def test_header_names_are_lowercased_in_order(self) -> None:
        request = "GET / HTTP/1.1\r\nHost: x\r\nUser-Agent: C\r\nAccept: */*\r\n\r\n"
        assert _header_names(request) == ("host", "user-agent", "accept")

    def test_header_lines_are_verbatim(self) -> None:
        request = "GET / HTTP/1.1\r\nHost: x\r\nCookie: a=1; b=2\r\n\r\n"
        assert _header_lines(request) == ("Host: x", "Cookie: a=1; b=2")

    def test_header_lines_without_terminator_keep_all_header_lines(self) -> None:
        request = "GET / HTTP/1.1\r\nHost: x\r\nCookie: a=1"
        assert _header_lines(request) == ("Host: x", "Cookie: a=1")

    def test_requests_root_only_for_root_path(self) -> None:
        assert _requests_root("GET /")
        assert _requests_root("GET / HTTP/1.1\r\n")
        assert _requests_root("GET / HTTP/1.1\r\nGET /favicon.ico")
        assert not _requests_root("GET /favicon.ico HTTP/1.1\r\n")
        assert not _requests_root("garbage")

    def test_body_lines_are_not_header_lines(self) -> None:
        # A colon-bearing body line must never reach captured_lines(): the
        # parity suite asserts on the NUMBER of Cookie lines, so a forged one
        # would defeat the duplicate-cookie checks.
        request = "POST / HTTP/1.1\r\nHost: x\r\n\r\nCookie: forged=1"
        assert _header_lines(request) == ("Host: x",)

    def test_header_name_splits_only_at_first_colon(self) -> None:
        request = "GET / HTTP/1.1\r\nX: a:b\r\n\r\n"
        assert _header_names(request) == ("x",)


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

    def test_max_bytes_accepts_a_terminated_head_at_the_boundary(self) -> None:
        head = b"\r\n\r\n"
        assert _read_head(_EofAfter(head), max_bytes=4) == "\r\n\r\n"

    def test_max_bytes_stops_after_an_unterminated_boundary_length(self) -> None:
        reader = _EofAfter(b"abcd")
        assert _read_head(reader, max_bytes=4) == ""
        assert reader.calls == [4]

    def test_reader_requests_at_most_the_remaining_limit(self) -> None:
        reader = _EofAfter(b"a" * 5, b"\r\n\r\n")
        assert _read_head(reader, max_bytes=9) == "a" * 5 + "\r\n\r\n"
        assert reader.calls == [9, 4]

    def test_reader_caps_each_recv_at_4096_bytes(self) -> None:
        reader = _EofAfter(b"a" * 5_000, b"\r\n\r\n")
        _read_head(reader, max_bytes=10_000)
        assert reader.calls[0] == 4_096

    def test_latin_one_preserves_non_ascii_header_bytes(self) -> None:
        head = b"GET / HTTP/1.1\r\nX-Byte: \xff\r\n\r\n"
        assert "ÿ" in _read_head(_EofAfter(head))

    def test_terminated_head_uses_the_first_separator(self) -> None:
        chunks = (b"GET / HTTP/1.1\r\n\r\nBODY\r\n\r\n",)
        assert _read_head(_EofAfter(*chunks)) == "GET / HTTP/1.1\r\n\r\n"

    def test_terminated_head_can_arrive_in_multiple_chunks(self) -> None:
        chunks = (b"GET / HTTP/1.1\r\n", b"Host: x\r\n", b"\r\nBODY")
        assert _read_head(_EofAfter(*chunks)) == "GET / HTTP/1.1\r\nHost: x\r\n\r\n"

    # ``""`` is what ``_read_head`` returns for a head that never terminated; a
    # ``/`` after the request line (here in a header) is not its target.
    @pytest.mark.parametrize(
        "head",
        [
            "GET /x HTTP/1.1",
            "GET  / HTTP/1.1",
            "garbage\r\nGET /x",
            "",
            "GET\r\nX: / y",
        ],
    )
    def test_requests_root_requires_a_well_formed_root_target(
        self,
        head: str,
    ) -> None:
        assert not _requests_root(head)


class _EofAfter:
    """A reader yielding ``chunks``, then a clean EOF forever.

    Honors ``bufsize`` like a real socket: a reader that ignored it could not
    show whether the caller's byte cap reaches the read at all.
    """

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = list(chunks)
        self.calls: list[int] = []

    def recv(self, bufsize: int, /) -> bytes:
        self.calls.append(bufsize)
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

    def test_certificate_has_exact_localhost_identity_and_lifetime(
        self,
        tmp_path: Path,
    ) -> None:
        cert_path, key_path = tmp_path / "c.pem", tmp_path / "k.pem"
        self_signed_localhost_cert(cert_path, key_path)
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        now = datetime.now(UTC)
        assert cert.subject.rfc4514_string() == "CN=localhost"
        assert cert.issuer.rfc4514_string() == "CN=localhost"
        public_key = cert.public_key()
        assert isinstance(public_key, rsa.RSAPublicKey | ec.EllipticCurvePublicKey)
        assert public_key.key_size == 2048
        assert cert.not_valid_before_utc >= now - timedelta(days=1, seconds=2)
        assert cert.not_valid_after_utc <= now + timedelta(days=3650, seconds=2)
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        assert san.critical is False
        assert san.value.get_values_for_type(x509.DNSName) == ["localhost"]
        serialization.load_pem_private_key(key_path.read_bytes(), password=None)


class TestEchoOracle:
    def test_default_client_timeout_is_exact(self) -> None:
        with EchoOracle() as oracle:
            assert oracle._client_timeout_sec == 10.0

    def test_initialization_exposes_exact_server_identity(self) -> None:
        with EchoOracle(client_timeout_sec=2.5) as oracle:
            assert oracle.ca_path.name == "cert.pem"
            assert sorted(path.name for path in oracle.ca_path.parent.iterdir()) == [
                "cert.pem",
                "key.pem",
            ]
            assert oracle.url == f"https://localhost:{oracle.port}/"
            assert oracle._client_timeout_sec == 2.5
            assert oracle._thread.name == "echo-oracle-accept"
            assert oracle._thread.daemon is True
            assert oracle._sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 1

    def test_initialization_uses_exact_socket_tls_and_thread_settings(
        self,
        tmp_path: Path,
    ) -> None:
        temporary = MagicMock()
        temporary.__enter__.return_value = str(tmp_path)
        context = MagicMock()
        sock = MagicMock()
        sock.__enter__.return_value = sock
        sock.getsockname.return_value = ("127.0.0.1", 4321)
        thread = MagicMock()
        thread.is_alive.return_value = False
        stack = MagicMock()
        stack.enter_context.side_effect = [str(tmp_path), sock]
        with (
            patch("wesearch.chrome.echo.ExitStack", return_value=stack),
            patch(
                "wesearch.chrome.echo.tempfile.TemporaryDirectory",
                return_value=temporary,
            ) as temporary_factory,
            patch("wesearch.chrome.echo.self_signed_localhost_cert"),
            patch("wesearch.chrome.echo.ssl.SSLContext", return_value=context),
            patch("wesearch.chrome.echo.socket.socket", return_value=sock),
            patch(
                "wesearch.chrome.echo.threading.Thread",
                return_value=thread,
            ) as thread_factory,
        ):
            oracle = EchoOracle(client_timeout_sec=2.5)
            oracle.close()
        temporary_factory.assert_called_once_with(prefix="echo-")
        stack.callback.assert_called_once_with(oracle._thread.join, timeout=5.0)
        stack.close.assert_called_once_with()
        context.set_alpn_protocols.assert_called_once_with(["http/1.1"])
        sock.setsockopt.assert_called_once_with(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        sock.bind.assert_called_once_with(("127.0.0.1", 0))
        sock.listen.assert_called_once_with(8)
        thread_factory.assert_called_once_with(
            target=oracle._serve,
            name="echo-oracle-accept",
            daemon=True,
        )
        thread.start.assert_called_once_with()

    def test_handle_sets_timeout_and_sends_exact_stub_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        oracle = object.__new__(EchoOracle)
        oracle._client_timeout_sec = 3.5
        context = MagicMock()
        monkeypatch.setattr(oracle, "_context", context, raising=False)
        raw = socket.socket()
        conn = MagicMock()
        context.wrap_socket.return_value = conn
        oracle._lock = threading.Lock()
        oracle._captures = []
        request = "GET / HTTP/1.1\r\nHost: x\r\n\r\n"
        with patch(
            "wesearch.chrome.echo._read_head",
            return_value=request,
        ):
            oracle._handle(raw)
        raw.close()
        context.wrap_socket.assert_called_once_with(raw, server_side=True)
        conn.sendall.assert_called_once_with(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
            b"Content-Length: 15\r\nConnection: close\r\n\r\n<html>ok</html>",
        )
        conn.close.assert_called_once_with()
        assert oracle._captures == [(("host",), ("Host: x",))]

    def test_serve_starts_exact_daemon_handler_for_each_connection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        oracle = object.__new__(EchoOracle)
        raw = MagicMock()
        sock = MagicMock()
        sock.accept.side_effect = [(raw, object()), OSError("closed")]
        monkeypatch.setattr(oracle, "_sock", sock, raising=False)
        oracle._stopped = threading.Event()
        handlers: list[threading.Thread] = []
        monkeypatch.setattr(oracle, "_handlers", handlers, raising=False)
        handle = MagicMock()
        monkeypatch.setattr(oracle, "_handle", handle, raising=False)
        handler = MagicMock()
        with patch(
            "wesearch.chrome.echo.threading.Thread",
            return_value=handler,
        ) as factory:
            oracle._serve()
        factory.assert_called_once_with(
            target=handle,
            args=(raw,),
            name="echo-oracle-handler",
            daemon=True,
        )
        assert handlers == [handler]
        handler.start.assert_called_once_with()

    def test_captures_ordered_headers_of_a_live_request(self) -> None:
        with EchoOracle() as oracle:
            context = ssl.create_default_context(cafile=str(oracle.ca_path))
            conn = client.HTTPSConnection(
                "localhost",
                oracle.port,
                timeout=5,
                context=context,
            )
            conn.request("GET", "/", headers={"User-Agent": "probe", "Accept": "*/*"})
            conn.getresponse().read()
            conn.close()
            names = oracle.captured()
            lines = oracle.captured_lines()
        assert "user-agent" in names
        assert names.index("host") < names.index("user-agent")
        assert any(line.startswith("User-Agent: probe") for line in lines)

    def test_ignores_non_root_requests(self) -> None:
        with EchoOracle() as oracle:
            context = ssl.create_default_context(cafile=str(oracle.ca_path))
            conn = client.HTTPSConnection(
                "localhost",
                oracle.port,
                timeout=5,
                context=context,
            )
            conn.request("GET", "/favicon.ico")
            conn.getresponse().read()
            conn.close()
            assert oracle.captured() == ()
            assert oracle.captured_lines() == ()


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
                socket.SOL_SOCKET,
                socket.SO_LINGER,
                struct.pack("ii", 1, 0),
            )
            aborting.connect(("127.0.0.1", oracle.port))
            aborting.close()

            context = ssl.create_default_context(cafile=str(oracle.ca_path))
            conn = client.HTTPSConnection(
                "localhost",
                oracle.port,
                timeout=5,
                context=context,
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

            served = client.HTTPSConnection(
                "localhost",
                oracle.port,
                timeout=5,
                context=context,
            )
            served.request("GET", "/", headers={"User-Agent": "probe"})
            served.getresponse().read()
            served.close()
            stalled.close()
            assert "user-agent" in oracle.captured()


class TestEchoOracleCleanup:
    """Every resource the oracle acquires must be released by ``close()``."""

    def test_close_uses_exact_shutdown_and_join_timeouts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        oracle = object.__new__(EchoOracle)
        oracle._client_timeout_sec = 10.0
        sock = MagicMock()
        sock.shutdown.side_effect = OSError("already closed")
        monkeypatch.setattr(oracle, "_sock", sock, raising=False)
        oracle.port = 4321
        thread = MagicMock()
        thread.is_alive.return_value = True
        monkeypatch.setattr(oracle, "_thread", thread, raising=False)
        handlers = [MagicMock()]
        monkeypatch.setattr(oracle, "_handlers", handlers, raising=False)
        oracle._stopped = threading.Event()
        stack = MagicMock()
        monkeypatch.setattr(oracle, "_stack", stack, raising=False)
        wake = MagicMock()
        with patch(
            "wesearch.chrome.echo.socket.create_connection",
            return_value=wake,
        ) as connect:
            oracle.close()
        sock.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        assert oracle._stopped.is_set()
        connect.assert_called_once_with(("127.0.0.1", 4321), timeout=1.0)
        wake.close.assert_called_once_with()
        thread.join.assert_called_once_with(timeout=1.0)
        handlers[0].join.assert_called_once_with(timeout=11.0)
        stack.close.assert_called_once_with()

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

    @pytest.mark.network_localhost
    def test_close_joins_a_stalled_handler_thread(self) -> None:
        # close() joined the accept thread only, so a handler could still be
        # inside _handle -- holding a client socket -- after teardown returned.
        started = threading.Event()
        release = threading.Event()
        oracle = EchoOracle()
        try:
            with (
                patch.object(
                    oracle,
                    "_handle",
                    partial(_stall_handler, started=started, release=release),
                ),
                closing(
                    socket.create_connection(("localhost", oracle.port), timeout=5),
                ),
            ):
                assert started.wait(5), "handler never accepted the connection"
                handler = oracle._handlers[0]
                with patch.object(
                    handler,
                    "join",
                    partial(_release_and_join, join=handler.join, release=release),
                ):
                    oracle.close()
                assert release.is_set(), "close did not join the stalled handler"
                assert not handler.is_alive()
        finally:
            release.set()
            oracle.close()


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


def _stall_handler(
    raw: socket.socket,
    *,
    started: threading.Event,
    release: threading.Event,
) -> None:
    with raw:
        started.set()
        assert release.wait(5), "handler was never released"


def _release_and_join(
    timeout: float | None = None,
    *,
    join: Callable[[float | None], None],
    release: threading.Event,
) -> None:
    release.set()
    join(timeout)


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
