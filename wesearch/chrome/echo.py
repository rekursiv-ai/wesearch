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
(via :func:`wesearch.chrome.capture.drive_chrome`); the fetch
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

from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

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

    Args:
      client_timeout_sec: How long one connection may stall before the oracle
        drops it. Bounds a client that opens a socket and never finishes its
        request; well above any loopback exchange, well below a test timeout,
        so a wedge fails an assertion rather than the suite.

    Attributes:
      url: The base URL clients should request (``https://localhost:<port>/``).
      port: The bound loopback port, for a client that needs it apart from
        :attr:`url` (``http.client`` takes host and port separately).
      ca_path: Path to the PEM certificate clients must trust to reach the
        oracle (``SSL_CERT_FILE`` for stdlib, ``verify`` for curl).

    """

    def __init__(self, *, client_timeout_sec: float = 10.0) -> None:
        self._client_timeout_sec = client_timeout_sec
        # Every acquisition registers its own release as it succeeds: a raise
        # part-way through __init__ leaves the caller no object, hence no way
        # to reach close(), so anything already acquired would be stranded.
        self._stack = ExitStack()
        try:
            directory = Path(self._stack.enter_context(tempfile.TemporaryDirectory()))
            self.ca_path = directory / "cert.pem"
            key_path = directory / "key.pem"
            self_signed_localhost_cert(self.ca_path, key_path)
            self._context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self._context.load_cert_chain(self.ca_path, key_path)
            self._context.set_alpn_protocols(["http/1.1"])
            self._sock = self._stack.enter_context(socket.socket())
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("127.0.0.1", 0))
            self._sock.listen(8)
            self._captures: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
            self._lock = threading.Lock()
            self._handlers: list[threading.Thread] = []
            self._stopped = threading.Event()
            self.port = int(self._sock.getsockname()[1])
            self.url = f"https://localhost:{self.port}/"
            self._thread = threading.Thread(
                target=self._serve, name="echo-oracle-accept", daemon=True
            )
            self._stack.callback(self._thread.join, timeout=5.0)
            self._thread.start()
        except BaseException:
            self._stack.close()
            raise

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
        """Stop the listener and release every resource acquired (idempotent)."""
        # shutdown() BEFORE the stack closes the socket: closing a listening
        # socket does not wake a thread blocked in accept() on Linux, so the
        # registered join would wait out its full timeout on every teardown and
        # then abandon the thread. shutdown() fails the accept immediately.
        # Already shut down or never connected: the close below still applies.
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._stopped.set()
        if self._thread.is_alive():
            # macOS rejects shutdown() on a listening socket without waking
            # accept(), so a loopback connection is the portable wake-up.
            with contextlib.suppress(OSError):
                wake = socket.create_connection(("127.0.0.1", self.port), timeout=1.0)
                wake.close()
            self._thread.join(timeout=1.0)
        # Handler threads hold a client socket each; a caller that asserts on
        # teardown must not observe one still inside _handle.
        for handler in tuple(self._handlers):
            handler.join(timeout=self._client_timeout_sec + 1.0)
        self._stack.close()

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
            if self._stopped.is_set():
                raw.close()
                return
            # One thread per connection: Chrome preconnects, leaving sockets
            # open and idle. Served serially, the first such socket parks this
            # loop in recv and starves every client behind it.
            handler = threading.Thread(
                target=self._handle,
                args=(raw,),
                name="echo-oracle-handler",
                daemon=True,
            )
            self._handlers.append(handler)
            handler.start()

    def _handle(self, raw: socket.socket) -> None:
        """TLS-wrap one connection, capture its header order, send a stub reply."""
        # A client that opens a socket and never completes its request must not
        # hold the connection open indefinitely.
        raw.settimeout(self._client_timeout_sec)
        try:
            conn = self._context.wrap_socket(raw, server_side=True)
        except OSError:
            # OSError, not ssl.SSLError (its subclass): a client that RSTs
            # mid-handshake raises ConnectionResetError instead, which escaped
            # _serve and killed the accept loop -- every later request in the
            # session then hung until its own timeout.
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


def _read_head(conn: _Recvable, *, max_bytes: int = 1 << 16) -> str:
    """Read a request up to and including the end of its header block.

    Returns ``""`` for a head that never terminated -- the peer hung up, stalled
    past its timeout mid-request, or ran past ``max_bytes`` without ending the
    block. Returning the PARTIAL bytes instead let ``GET / HTTP`` (Chrome's
    preconnect, and what the resilience test sends) parse as a request to ``/``
    carrying zero headers, which then overwrote the real capture as the most
    recent one.

    Bytes past the terminator are DROPPED: one ``recv`` can deliver head and
    body together, and a body line containing a colon is indistinguishable from
    a header to every downstream parser -- letting a request forge the Cookie
    lines the parity suite counts.
    """
    data = b""
    while (end := data.find(b"\r\n\r\n")) == -1 and len(data) < max_bytes:
        chunk = conn.recv(min(4096, max_bytes - len(data)))
        if not chunk:
            return ""
        data += chunk
    return data[: end + 4].decode("latin-1") if end != -1 else ""


class _Recvable(Protocol):
    """The one operation reading a head needs.

    Narrower than ``ssl.SSLSocket`` on purpose: the truncation this reader must
    distinguish is reproducible only by feeding it bytes directly, and a
    parameter that demands a real TLS socket makes that test impossible to
    write without a cast asserting something untrue.
    """

    def recv(self, bufsize: int, /) -> bytes: ...


def _requests_root(request: str) -> bool:
    """Whether the request line targets ``/`` (ignore Chrome sub-resource GETs)."""
    line = request.split("\r\n", 1)[0]
    parts = line.split(" ")
    return len(parts) >= 2 and parts[1] == "/"


def _header_names(request: str) -> tuple[str, ...]:
    """Ordered lower-cased header names from a raw HTTP/1.1 request head."""
    return tuple(line.split(":", 1)[0].lower() for line in _header_lines(request))


def _header_lines(request: str) -> tuple[str, ...]:
    """Verbatim ``name: value`` header lines from a raw HTTP/1.1 request head.

    Stops at the blank line ending the block, so a body that arrived in the
    same read cannot contribute a colon-bearing line as a header.
    """
    lines = request.split("\r\n")[1:]
    end = lines.index("") if "" in lines else len(lines)
    return tuple(line for line in lines[:end] if ":" in line)
