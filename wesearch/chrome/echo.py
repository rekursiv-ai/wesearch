"""Loopback TLS header-echo oracle for wire-parity tests.

A parity test proves that :func:`wesearch.fetch.fetch` presents the same
ordered wire headers as a real Chrome. That needs a server which (a) reports the
exact header order it received and (b) serves valid TLS both a headless Chrome
and the fetch backends will accept -- without depending on a third-party echo
host whose certificate can expire out from under the suite.

:class:`EchoOracle` is that server, self-contained and dependency-light: it mints
a self-signed ``localhost`` certificate with :mod:`cryptography`, serves HTTPS on
a loopback port from a daemon thread, and records the ordered header names of
each request to ``/``. Chrome reaches it with ``--ignore-certificate-errors``
(via :func:`wesearch.chrome.capture.capture_chrome_request`); the fetch
backends reach it by trusting :attr:`ca_path` (``SSL_CERT_FILE`` for the stdlib
path, ``verify`` for curl).

The listener offers only ALPN ``http/1.1``. HTTP/1.1 is the one protocol every
backend speaks (``http.client`` has no HTTP/2), so a single plain
:mod:`ssl` socket captures curl, stdlib, and Chrome alike with no HTTP/2 framing
library. The one wire consequence -- Chrome omits the HTTP/2-only ``Priority``
header on HTTP/1.1 -- is reproduced by the stdlib backend and is a known
curl_cffi divergence on the curl backend (see
:func:`wesearch.chrome.headers.chrome_navigation_headers`).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Self

import contextlib
import socket
import ssl
import tempfile
import threading

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


__all__ = [
    "EchoOracle",
    "self_signed_localhost_cert",
]


class EchoOracle:
    """A loopback HTTPS server that records each request's ordered header names.

    Serves ``https://localhost:<port>/`` over a self-signed certificate from a
    daemon thread. Every request to ``/`` has its header names captured in wire
    order; :meth:`captured` returns the most recent capture. Use as a context
    manager so the socket and thread are always released.

    Attributes:
      url: The base URL clients should request (``https://localhost:<port>/``).
      ca_path: Path to the PEM certificate clients must trust to reach the
        oracle (``SSL_CERT_FILE`` for stdlib, ``verify`` for curl).

    """

    def __init__(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="echo-oracle-"))
        self.ca_path = self._dir / "cert.pem"
        key_path = self._dir / "key.pem"
        self_signed_localhost_cert(self.ca_path, key_path)
        self._context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._context.load_cert_chain(self.ca_path, key_path)
        self._context.set_alpn_protocols(["http/1.1"])
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._captures: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self._lock = threading.Lock()
        self.url = f"https://localhost:{self._sock.getsockname()[1]}/"
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def captured(self) -> tuple[str, ...]:
        """Return the ordered header names of the most recent request to ``/``.

        Returns:
          names: Lower-cased header names in the order the request sent them,
            empty when no request to ``/`` has been received.

        """
        with self._lock:
            return self._captures[-1][0] if self._captures else ()

    def captured_lines(self) -> tuple[str, ...]:
        """Return the raw ``name: value`` header lines of the most recent ``/``.

        Returns:
          lines: Verbatim request header lines in wire order, for assertions on
            values (e.g. a single ``Cookie`` line); empty when none received.

        """
        with self._lock:
            return self._captures[-1][1] if self._captures else ()

    def close(self) -> None:
        """Stop the listener and release the socket (idempotent)."""
        # shutdown() BEFORE close(): closing a listening socket does not wake a
        # thread blocked in accept() on Linux, so the join below would wait out
        # its full timeout on every teardown and then abandon the thread.
        # shutdown() fails the accept immediately.
        # Already shut down or never connected: close() below still applies.
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _serve(self) -> None:
        """Accept connections until the socket closes, capturing each request."""
        while True:
            try:
                raw, _ = self._sock.accept()
            except OSError:
                return  # Socket closed by close(); normal shutdown.
            self._handle(raw)

    def _handle(self, raw: socket.socket) -> None:
        """TLS-wrap one connection, capture its header order, send a stub reply."""
        try:
            conn = self._context.wrap_socket(raw, server_side=True)
        except ssl.SSLError:
            raw.close()
            return
        try:
            request = _read_head(conn)
            if _requests_root(request):
                with self._lock:
                    self._captures.append(
                        (_header_names(request), _header_lines(request))
                    )
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
                b"Content-Length: 15\r\nConnection: close\r\n\r\n<html>ok</html>"
            )
        except OSError:
            pass  # A client hangup mid-exchange is not an oracle failure.
        finally:
            conn.close()


def self_signed_localhost_cert(cert_path: Path, key_path: Path) -> None:
    """Write a self-signed ``localhost`` certificate and key to the given paths.

    Args:
      cert_path: Destination for the PEM certificate.
      key_path: Destination for the PEM private key.

    """
    key = rsa.generate_private_key(public_exponent=65_537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )


def _read_head(conn: ssl.SSLSocket, *, max_bytes: int = 1 << 16) -> str:
    """Read a request up to the end of its header block."""
    data = b""
    while b"\r\n\r\n" not in data and len(data) < max_bytes:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    return data.decode("latin-1")


def _requests_root(request: str) -> bool:
    """Whether the request line targets ``/`` (ignore Chrome sub-resource GETs)."""
    line = request.split("\r\n", 1)[0]
    parts = line.split(" ")
    return len(parts) >= 2 and parts[1] == "/"


def _header_names(request: str) -> tuple[str, ...]:
    """Ordered lower-cased header names from a raw HTTP/1.1 request head."""
    return tuple(line.split(":", 1)[0].lower() for line in _header_lines(request))


def _header_lines(request: str) -> tuple[str, ...]:
    """Verbatim ``name: value`` header lines from a raw HTTP/1.1 request head."""
    return tuple(line for line in request.split("\r\n")[1:] if ":" in line)
