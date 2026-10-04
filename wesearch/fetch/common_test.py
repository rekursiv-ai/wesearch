"""Tests for wesearch.fetch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import gzip
import socket
import zlib

import brotli
import pytest

from wesearch.fetch.common import (
    _netloc,
    apply_redirect,
    bracket_ipv6,
    decompress,
    decompress_error_body,
    default_port,
    host_header,
    join_headers,
    origin,
    pinned_host,
    public_host,
    redirect_target,
    rewrite_origin,
)
from wesearch.fetch.testing import zstd_compress
from wesearch.types.errors import FetchError


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
        with pytest.raises(ValueError, match=r"^URL has no host\.$"):
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

    @pytest.mark.parametrize("ip", ["0.0.0." + "0", "224.0.0.1", "192.0.2.1"])
    def test_rejects_unspecified_multicast_and_reserved_addresses(
        self,
        ip: str,
    ) -> None:
        with (
            patch("socket.getaddrinfo", return_value=_addrinfo(ip)),
            pytest.raises(ValueError, match="non-public"),
        ):
            public_host("special.example")

    def test_resolution_receives_host_and_unspecified_service(self) -> None:
        with patch("socket.getaddrinfo", return_value=_addrinfo("8.8.8.8")) as resolve:
            public_host("example.com")
        resolve.assert_called_once_with("example.com", None)

    def test_rejects_reserved_flag_independently(self) -> None:
        address = SimpleNamespace(
            is_loopback=False,
            is_link_local=False,
            is_private=False,
            is_multicast=False,
            is_reserved=True,
            is_unspecified=False,
        )
        with (
            patch("socket.getaddrinfo", return_value=_addrinfo("8.8.8.8")),
            patch(
                "wesearch.fetch.common.ipaddress.ip_address",
                return_value=address,
            ),
            pytest.raises(ValueError, match="non-public"),
        ):
            public_host("reserved.example")

    def test_rejects_loopback_flag_independently(self) -> None:
        address = SimpleNamespace(
            is_loopback=True,
            is_link_local=False,
            is_private=False,
            is_multicast=False,
            is_reserved=False,
            is_unspecified=False,
        )
        with (
            patch("socket.getaddrinfo", return_value=_addrinfo("8.8.8.8")),
            patch(
                "wesearch.fetch.common.ipaddress.ip_address",
                return_value=address,
            ),
            pytest.raises(ValueError, match="non-public"),
        ):
            public_host("loopback.example")

    def test_rejects_empty_resolution_with_exact_error(self) -> None:
        with (
            patch("socket.getaddrinfo", return_value=[]),
            pytest.raises(
                ValueError,
                match=r"^DNS resolution returned no address for 'empty.example'\.$",
            ),
        ):
            public_host("empty.example")

    def test_prefers_ipv4_when_resolver_lists_ipv6_first(self) -> None:
        # ``getaddrinfo`` often returns AAAA first, but many networks have no
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


class TestOrigin:
    """One URL's origin must have ONE spelling.

    The raw ``scheme://netloc`` kept host case, userinfo, and an explicit
    default port, so four equivalent URLs produced four keys. This value is the
    cookie-jar key, the Accept-CH key, and the browser leg's same-origin test:
    a second spelling fragments the jar and strips a credential from a hop that
    never left the origin.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "https://EXAMPLE.com/a",
            "https://example.com:443/a",
            "https://User:pw@example.com/a",
            "https://User:pw@EXAMPLE.com:443/a",
        ],
    )
    def test_equivalent_urls_share_one_origin(self, url: str) -> None:
        assert origin(url) == "https://example.com"

    def test_hostless_origin_keeps_empty_host(self) -> None:
        assert origin("") == "://"

    def test_a_non_default_port_is_part_of_the_origin(self) -> None:
        assert origin("https://example.com:8443/a") == "https://example.com:8443"
        assert origin("http://example.com:8080/a") == "http://example.com:8080"

    def test_an_ipv6_host_stays_bracketed(self) -> None:
        assert origin("https://[2606:4700::1]/x") == "https://[2606:4700::1]"

    def test_the_rewritten_origin_header_is_the_canonical_one(self) -> None:
        # ``apply_redirect`` decides same-origin with ``origin()`` and then
        # writes the header with ``rewrite_origin()``. Two spellings of one
        # origin means the value announced to the target disagrees with the
        # value the hop was judged against.
        out = rewrite_origin({"Origin": "https://a.com"}, "https://b.com:443/x")
        assert out["Origin"] == origin("https://b.com:443/x")

    def test_a_credential_survives_a_canonically_same_origin_hop(self) -> None:
        headers, _m, _b = apply_redirect(
            "https://User:pw@EXAMPLE.com:443/1",
            {"Authorization": "Bearer secret"},
            "GET",
            body=None,
            status=307,
            redirect_url="https://example.com/2",
        )
        assert headers.get("Authorization") == "Bearer secret"


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

    def test_301_downgrades_post_to_get(self) -> None:
        _headers, method, body = apply_redirect(
            "https://x/submit",
            {},
            "POST",
            body=b"{}",
            status=301,
            redirect_url="https://x/land",
        )
        assert method == "GET"
        assert body is None

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
        assert headers.get("Accept") == "*/*"  # non-origin-bound survives.

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

    @pytest.mark.parametrize("name", ["Authorization", "authorization"])
    def test_cross_origin_drops_authorization(self, name: str) -> None:
        """A credential is origin-bound exactly like a cookie.

        Forwarding it hands the source origin's secret to whatever host answered
        the hop. Both casings: HTTP field names are case-insensitive.
        """
        headers, _m, _b = apply_redirect(
            "https://a.com/1",
            {name: "Bearer secret", "Accept": "*/*"},
            "GET",
            body=None,
            status=307,
            redirect_url="https://evil.example/2",
        )

        assert not any(k.lower() == "authorization" for k in headers)
        assert headers.get("Accept") == "*/*"

    def test_get_methods_are_preserved_for_every_redirect_status(self) -> None:
        for status in (301, 302, 303):
            headers, method, body = apply_redirect(
                "https://a.com/1",
                {"Origin": "https://a.com"},
                "GET",
                body=b"ignored",
                status=status,
                redirect_url="https://a.com/2",
            )
            assert headers["Origin"] == "https://a.com"
            assert method == "GET"
            assert body == b"ignored"

    def test_same_origin_keeps_authorization(self) -> None:
        """A same-origin hop is still the origin the credential belongs to."""
        headers, _m, _b = apply_redirect(
            "https://a.com/1",
            {"Authorization": "Bearer secret"},
            "GET",
            body=None,
            status=307,
            redirect_url="https://a.com/2",
        )

        assert headers.get("Authorization") == "Bearer secret"


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
        assert decompress(zstd_compress(data), "zstd") == data

    def test_zstd_streaming_frame_no_size(self) -> None:
        # Streaming-mode frames omit decompressed size from the header;
        # `ZstdDecompressor.decompress()` rejects them. Real servers
        # (e.g. Cloudflare) emit such frames -- we must handle them.
        data = b"hello world " * 1000
        assert decompress(zstd_compress(data, streaming=True), "zstd") == data

    def test_zstd_garbage_raises(self) -> None:
        with pytest.raises(ValueError, match="Decompression failed"):
            decompress(b"not zstd", "zstd")

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
        raw = zlib.compress(data)[2:-4]  # Strip zlib header + adler checksum.
        assert decompress(raw, "deflate") == data

    def test_chained_content_encoding(self) -> None:
        # REV2A-003: chained "gzip, br" is RFC-legal; apply right-to-left.
        data = b"hello world"
        chained = gzip.compress(brotli.compress(data))
        assert decompress(chained, "br, gzip") == data


class TestSharedHelpers:
    def test_netloc_handles_port_and_missing_port(self) -> None:
        assert _netloc("example.com", 8443) == "example.com:8443"
        assert _netloc("example.com", None) == "example.com"

    def test_default_port_and_host_header_boundaries(self) -> None:
        assert default_port("https") == 443
        assert default_port("http") == 80
        assert host_header("example.com", 443, "https") == "example.com"
        assert host_header("example.com", 444, "https") == "example.com:444"
        assert host_header("example.com", None, "https") == "example.com"

    def test_join_headers_folds_regular_and_cookie_duplicates_differently(self) -> None:
        assert join_headers(
            [
                ("X-Test", "a"),
                ("x-test", "b"),
                ("Set-Cookie", "a=1, x"),
                ("set-cookie", "b=2"),
            ],
        ) == {"x-test": "a, b", "set-cookie": "a=1, x\nb=2"}

    def test_redirect_target_resolves_relative_location_and_requires_header(
        self,
    ) -> None:
        assert (
            redirect_target("https://example.com/a/b", 302, {"location": "../c"})
            == "https://example.com/c"
        )
        with pytest.raises(FetchError) as error:
            redirect_target("https://example.com/a", 301, {})
        assert error.value.url == "https://example.com/a"
        assert error.value.status == 301
        assert error.value.headers == {}
        assert error.value.body == b"Redirect with no Location header"

    def test_decompress_error_body_falls_back_to_raw_bytes(self) -> None:
        raw = b"not compressed"
        assert decompress_error_body(raw, {"content-encoding": "gzip"}) == raw
        assert decompress_error_body(raw, {}) == raw
        assert decompress_error_body(b"RAW", {"content-encoding": "IDENTITY"}) == b"RAW"
        assert (
            decompress_error_body(gzip.compress(b"ok"), {"content-encoding": "gzip"})
            == b"ok"
        )

    def test_decompress_error_body_passes_empty_encoding_to_decompress(self) -> None:
        with patch(
            "wesearch.fetch.common.decompress",
            return_value=b"ok",
        ) as decode:
            assert decompress_error_body(b"raw", {}) == b"ok"
        decode.assert_called_once_with(b"raw", "")

    def test_pinned_host_internal_skips_resolution(self) -> None:
        with patch("socket.getaddrinfo", side_effect=AssertionError):
            assert pinned_host("https://example.com", "internal") is None

    def test_pinned_host_untrusted_returns_public_pin(self) -> None:
        with patch("socket.getaddrinfo", return_value=_addrinfo("8.8.8.8")) as resolve:
            pin = pinned_host("https://example.com", "untrusted")
        assert pin is not None
        assert pin.ip == "8.8.8.8"
        resolve.assert_called_once_with("example.com", None)

    def test_pinned_host_rejects_hostless_untrusted_url(self) -> None:
        with pytest.raises(ValueError, match="no host"):
            pinned_host("/relative", "untrusted")


class TestIPv6Bracketing:
    def test_ipv6_address_bracketed(self) -> None:
        assert bracket_ipv6("2606:4700::6810:7c60") == "[2606:4700::6810:7c60]"

    def test_already_bracketed_unchanged(self) -> None:
        assert bracket_ipv6("[::1]") == "[::1]"

    def test_ipv4_unchanged(self) -> None:
        assert bracket_ipv6("93.184.216.34") == "93.184.216.34"

    def test_hostname_unchanged(self) -> None:
        assert bracket_ipv6("example.com") == "example.com"


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
