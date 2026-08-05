"""Tests for wesearch.fetch."""

from __future__ import annotations

from unittest.mock import patch

import gzip
import io
import socket
import zlib

import brotli
import pytest
import zstandard

from wesearch.fetch.common import (
    apply_redirect,
    bracket_ipv6,
    decompress,
    public_host,
    rewrite_origin,
)


# socket.getaddrinfo returns the canonical 5-tuple
# (family, type, proto, canonname, sockaddr); only the IP inside sockaddr
# matters here. ``AddrInfo`` names the shape once so the tests can stop
# repeating it.
type AddrInfo = tuple[int, int, int, str, tuple[str, int]]


def _addrinfo(*ips: str) -> list[AddrInfo]:
    """Build a ``socket.getaddrinfo``-shaped result over ``ips``, in order."""
    return [
        (socket.AF_INET6 if ":" in ip else socket.AF_INET, 0, 0, "", (ip, 0))
        for ip in ips
    ]


class TestPublicHost:
    def test_rejects_dns_failure(self) -> None:
        with (
            patch("socket.getaddrinfo", side_effect=socket.gaierror("nope")),
            pytest.raises(ValueError, match="DNS"),
        ):
            public_host("does-not-exist.invalid")

    def test_rejects_a_missing_host(self) -> None:
        with pytest.raises(ValueError, match="no host"):
            public_host("")

    def test_rejects_loopback(self) -> None:
        with (
            patch("socket.getaddrinfo", return_value=_addrinfo("127.0.0.1")),
            pytest.raises(ValueError, match="non-public"),
        ):
            public_host("localhost")

    def test_rejects_link_local_metadata(self) -> None:
        # The cloud metadata endpoint is the canonical SSRF target.
        with (
            patch("socket.getaddrinfo", return_value=_addrinfo("169.254.169.254")),
            pytest.raises(ValueError, match="non-public"),
        ):
            public_host("metadata.example")

    def test_rejects_an_ipv4_mapped_private_address(self) -> None:
        # ``::ffff:10.0.0.1`` is a private v4 address wearing a v6 spelling;
        # ipaddress flags it, and this pins that it stays flagged.
        with (
            patch("socket.getaddrinfo", return_value=_addrinfo("::ffff:10.0.0.1")),
            pytest.raises(ValueError, match="non-public"),
        ):
            public_host("mapped.example")

    def test_accepts_a_public_address(self) -> None:
        with patch("socket.getaddrinfo", return_value=_addrinfo("8.8.8.8")):
            assert public_host("example.com").ip == "8.8.8.8"

    def test_rejects_when_any_resolution_is_private(self) -> None:
        # One public answer does not license the name: an attacker controlling
        # the zone can serve the private one on the next lookup.
        with (
            patch("socket.getaddrinfo", return_value=_addrinfo("8.8.8.8", "127.0.0.1")),
            pytest.raises(ValueError, match="non-public"),
        ):
            public_host("example.com")

    def test_prefers_ipv4_when_resolver_lists_ipv6_first(self) -> None:
        # getaddrinfo often returns AAAA first, but many networks have no
        # working v6 route; pinning that address fails with status 0 on a page
        # that plainly serves over v4.
        with patch(
            "socket.getaddrinfo",
            return_value=_addrinfo("2606:4700:20::ac43:4403", "104.26.13.77"),
        ) as resolve:
            assert public_host("docs.astral.sh").ip == "104.26.13.77"
        assert resolve.call_count == 1

    def test_uses_ipv6_when_it_is_the_only_family(self) -> None:
        with patch("socket.getaddrinfo", return_value=_addrinfo("2606:4700:20::1")):
            assert public_host("v6only.example").ip == "2606:4700:20::1"

    def test_returns_the_bare_host_not_the_netloc(self) -> None:
        # The transport re-appends any port itself, so a port here doubles it.
        with patch("socket.getaddrinfo", return_value=_addrinfo("1.2.3.4")):
            assert public_host("example.com:8443").host == "example.com"


class TestRewriteOrigin:
    def test_cross_origin_rewrite(self) -> None:
        out = rewrite_origin({"Origin": "https://a.com"}, "https://b.com/land")
        assert out["Origin"] == "https://b.com"

    def test_no_origin_header_unchanged(self) -> None:
        h = {"Accept": "*/*"}
        assert rewrite_origin(h, "https://b.com/x") is h

    def test_ipv6_target_is_bracketed(self) -> None:
        # REV2061-003: a v6 redirect target must yield a BRACKETED Origin;
        # "https://2606:...::1" is an invalid Origin (colons unbracketed).
        out = rewrite_origin({"Origin": "https://a.com"}, "https://[2606:4700::1]/x")
        assert out["Origin"] == "https://[2606:4700::1]"

    def test_case_variant_origin_is_rewritten_not_leaked(self) -> None:
        # REVE559-003: HTTP field names are case-insensitive. A caller-supplied
        # "origin" (lowercase) must still be rewritten, not leaked verbatim.
        out = rewrite_origin({"origin": "https://a.com"}, "https://b.com/x")
        assert not any(
            v == "https://a.com" for k, v in out.items() if k.lower() == "origin"
        )
        assert any(
            v == "https://b.com" for k, v in out.items() if k.lower() == "origin"
        )


class TestApplyRedirect:
    def test_303_drops_case_variant_content_type(self) -> None:
        # REVE559-002: a 303 POST->GET must drop Content-Type regardless of case.
        headers, method, body = apply_redirect(
            "https://x/submit",
            {"content-type": "application/json", "Accept": "*/*"},
            "POST",
            body=b"{}",
            status=303,
            redirect_url="https://x/result",
        )
        assert method == "GET"
        assert body is None
        assert not any(k.lower() == "content-type" for k in headers)

    def test_302_downgrades_post_to_get(self) -> None:
        _headers, method, body = apply_redirect(
            "https://x/submit",
            {},
            "POST",
            body=b"{}",
            status=302,
            redirect_url="https://x/land",
        )
        assert method == "GET"
        assert body is None

    def test_307_preserves_method_and_body(self) -> None:
        _headers, method, body = apply_redirect(
            "https://x/submit",
            {},
            "POST",
            body=b"{}",
            status=307,
            redirect_url="https://x/land",
        )
        assert method == "POST"
        assert body == b"{}"

    def test_cross_origin_drops_cookie_and_hints(self) -> None:
        headers, _m, _b = apply_redirect(
            "https://a.com/1",
            {"Cookie": "SID=x", "sec-ch-ua-arch": '"x86"', "Accept": "*/*"},
            "GET",
            body=None,
            status=302,
            redirect_url="https://b.com/2",
        )
        assert "Cookie" not in headers
        assert "sec-ch-ua-arch" not in headers
        assert headers.get("Accept") == "*/*"  # non-origin-bound survives

    def test_same_origin_keeps_cookie_and_hints(self) -> None:
        headers, _m, _b = apply_redirect(
            "https://a.com/1",
            {"Cookie": "SID=x", "sec-ch-ua-arch": '"x86"'},
            "GET",
            body=None,
            status=302,
            redirect_url="https://a.com/2",
        )
        assert headers.get("Cookie") == "SID=x"
        assert headers.get("sec-ch-ua-arch") == '"x86"'


class TestDecompress:
    def test_gzip(self) -> None:
        data = b"hello world"
        assert decompress(gzip.compress(data), "gzip") == data

    def test_deflate(self) -> None:
        data = b"hello world"
        assert decompress(zlib.compress(data), "deflate") == data

    def test_brotli(self) -> None:
        data = b"hello world"
        assert decompress(brotli.compress(data), "br") == data

    def test_zstd(self) -> None:
        data = b"hello world"
        compressed = zstandard.ZstdCompressor().compress(data)
        assert decompress(compressed, "zstd") == data

    def test_zstd_streaming_frame_no_size(self) -> None:
        # Streaming-mode frames omit decompressed size from the header;
        # `ZstdDecompressor.decompress()` rejects them. Real servers
        # (e.g. Cloudflare) emit such frames -- we must handle them.
        data = b"hello world " * 1000
        buf = io.BytesIO()
        with zstandard.ZstdCompressor().stream_writer(buf, closefd=False) as w:
            _ = w.write(data)
        assert decompress(buf.getvalue(), "zstd") == data

    def test_identity(self) -> None:
        assert decompress(b"raw", "identity") == b"raw"

    def test_empty_encoding(self) -> None:
        assert decompress(b"raw", "") == b"raw"

    def test_unknown_encoding_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Content-Encoding"):
            decompress(b"raw", "unknown")

    def testdecompression_failure_raises(self) -> None:
        with pytest.raises(ValueError, match="Decompression failed"):
            decompress(b"not gzip", "gzip")

    def test_raw_deflate_without_zlib_header(self) -> None:
        # REV2A-002: some servers emit raw DEFLATE (no zlib wrapper); a browser
        # falls back to wbits=-MAX_WBITS. We must decode it, not raise.
        data = b"hello world"
        raw = zlib.compress(data)[2:-4]  # strip zlib header + adler checksum
        assert decompress(raw, "deflate") == data

    def test_chained_content_encoding(self) -> None:
        # REV2A-003: chained "gzip, br" is RFC-legal; apply right-to-left.
        data = b"hello world"
        chained = gzip.compress(brotli.compress(data))
        assert decompress(chained, "br, gzip") == data


class TestIPv6Bracketing:
    def test_ipv6_address_bracketed(self) -> None:
        assert bracket_ipv6("2606:4700::6810:7c60") == "[2606:4700::6810:7c60]"

    def test_already_bracketed_unchanged(self) -> None:
        assert bracket_ipv6("[::1]") == "[::1]"

    def test_ipv4_unchanged(self) -> None:
        assert bracket_ipv6("93.184.216.34") == "93.184.216.34"

    def test_hostname_unchanged(self) -> None:
        assert bracket_ipv6("example.com") == "example.com"
