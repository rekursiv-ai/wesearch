"""Tests for wesearch.fetch.common."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import gzip
import socket
import zlib

import brotli
import pytest

from wesearch.chrome.headers import chrome_client_hints
from wesearch.fetch.common import (
    ORIGIN_BOUND,
    DecompressionError,
    PendingRequest,
    bracket_ipv6,
    cookie_header,
    decompress,
    decompress_error_body,
    host_header,
    join_headers,
    merge_headers,
    origin,
    pinned_host,
    public_host,
    redirect_target,
)
from wesearch.lib import zstd_compat
from wesearch.types.errors import FetchError
from wesearch.types.params import ignore, no_cookies


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


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
    cookie-jar key and the browser leg's same-origin test:
    a second spelling fragments the jar and strips a credential from a pending that
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

    def test_a_credential_survives_a_canonically_same_origin_hop(self) -> None:
        pending = _pending(
            "https://User:pw@EXAMPLE.com:443/1",
            {"Authorization": "Bearer x"},
        )
        assert _follow(pending, 307, {"location": "https://example.com/2"})
        assert pending.headers == {"Authorization": "Bearer x"}


def _pending(
    url: str,
    headers: dict[str, str],
    *,
    method: str = "GET",
    body: bytes = b"",
    remaining: int = 1,
) -> PendingRequest:
    return PendingRequest(
        url=url,
        method=method,
        headers=headers,
        body=body,
        remaining=remaining,
    )


def _follow(
    pending: PendingRequest,
    status: int,
    headers: dict[str, str],
    *,
    cookies_for_host: Callable[[str], Mapping[str, str]] = no_cookies,
) -> bool:
    return pending.follow(
        status,
        headers,
        on_response=ignore,
        on_redirect=ignore,
        cookies_for_host=cookies_for_host,
    )


class TestPendingRequestOrigin:
    """``Origin`` names the hop's new target, in its canonical spelling."""

    def test_cross_origin_rewrite(self) -> None:
        pending = _pending("https://a.com/1", {"Origin": "https://a.com"})
        assert _follow(pending, 307, {"location": "https://b.com:443/land"})
        assert pending.headers["Origin"] == origin("https://b.com:443/land")
        assert pending.headers["Origin"] == "https://b.com"

    def test_no_origin_header_none_added(self) -> None:
        pending = _pending("https://a.com/1", {"Accept": "*/*"})
        assert _follow(pending, 307, {"location": "https://b.com/x"})
        assert pending.headers == {"Accept": "*/*"}

    def test_ipv6_target_is_bracketed(self) -> None:
        # A v6 redirect target must yield a BRACKETED Origin;
        # "https://2606:...::1" is an invalid Origin (colons unbracketed).
        pending = _pending("https://a.com/1", {"Origin": "https://a.com"})
        assert _follow(pending, 307, {"location": "https://[2606:4700::1]/x"})
        assert pending.headers["Origin"] == "https://[2606:4700::1]"

    def test_case_variant_origin_is_rewritten_not_leaked(self) -> None:
        # HTTP field names are case-insensitive. A caller-supplied "origin"
        # (lowercase) must still be rewritten, not leaked verbatim.
        pending = _pending("https://a.com/1", {"origin": "https://a.com"})
        assert _follow(pending, 307, {"location": "https://b.com/x"})
        assert pending.headers == {"Origin": "https://b.com"}


class TestPendingRequestFollow:
    """``PendingRequest.follow`` is the one redirect step curl and stdlib both take."""

    def test_reports_the_response_and_the_redirect_then_advances(self) -> None:
        pending = _pending(
            "https://a.example/submit",
            {"Origin": "https://a.example", "Content-Type": "x"},
            method="POST",
            body=b"body",
        )
        responses: list[tuple[int, dict[str, str], str]] = []
        redirects: list[str] = []

        def record(status: int, headers: dict[str, str], url: str) -> None:
            responses.append((status, headers, url))

        assert pending.follow(
            303,
            {"location": "https://b.example/result"},
            on_response=record,
            on_redirect=redirects.append,
            cookies_for_host=no_cookies,
        )
        assert responses == [
            (303, {"location": "https://b.example/result"}, "https://a.example/submit"),
        ]
        assert redirects == ["https://b.example/result"]
        assert (pending.url, pending.method, pending.body, pending.remaining) == (
            "https://b.example/result",
            "GET",
            b"",
            0,
        )
        assert pending.headers == {"Origin": "https://b.example"}

    def test_does_not_follow_at_zero_but_still_reports(self) -> None:
        pending = _pending("https://a.example/", {}, remaining=0)
        seen: list[int] = []
        assert not pending.follow(
            302,
            {"location": "https://b.example/"},
            on_response=lambda status, _h, _u: seen.append(status),
            on_redirect=lambda _u: pytest.fail("followed past the budget"),
            cookies_for_host=no_cookies,
        )
        assert seen == [302]
        assert (pending.url, pending.remaining) == ("https://a.example/", 0)

    @pytest.mark.parametrize("status", [200, 304, 404])
    def test_a_non_redirect_status_is_terminal(self, status: int) -> None:
        pending = _pending("https://a.example/", {})
        assert not _follow(pending, status, {"location": "https://b.example/"})
        assert pending.url == "https://a.example/"

    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    def test_every_redirect_status_is_followed(self, status: int) -> None:
        pending = _pending("https://a.example/", {})
        assert _follow(pending, status, {"location": "https://a.example/next"})
        assert pending.url == "https://a.example/next"

    def test_a_raising_redirect_callback_leaves_the_hop_unchanged(self) -> None:
        pending = _pending("https://a.example/", {"Cookie": "a=1"})

        def refuse(url: str) -> None:
            raise ValueError(url)

        with pytest.raises(ValueError, match=r"b\.example"):
            pending.follow(
                302,
                {"location": "https://b.example/"},
                on_response=ignore,
                on_redirect=refuse,
                cookies_for_host=no_cookies,
            )
        assert (pending.url, pending.headers) == (
            "https://a.example/",
            {"Cookie": "a=1"},
        )

    def test_a_redirect_without_location_raises_with_status(self) -> None:
        pending = _pending("https://a.example/base/start", {}, method="POST", body=b"b")
        with pytest.raises(FetchError, match="302"):
            _follow(pending, 302, {})

    def test_resolves_a_relative_location_and_307_keeps_the_post(self) -> None:
        pending = _pending(
            "https://a.example/base/start",
            {"Content-Type": "x"},
            method="POST",
            body=b"body",
        )
        assert _follow(pending, 307, {"location": "next"})
        assert (pending.url, pending.method, pending.body) == (
            "https://a.example/base/next",
            "POST",
            b"body",
        )
        assert pending.headers == {"Content-Type": "x"}

    def test_303_drops_case_variant_content_type(self) -> None:
        pending = _pending(
            "https://x/submit",
            {"content-type": "application/json", "Accept": "*/*"},
            method="POST",
            body=b"{}",
        )
        assert _follow(pending, 303, {"location": "https://x/result"})
        assert (pending.method, pending.body) == ("GET", b"")
        assert pending.headers == {"Accept": "*/*"}

    @pytest.mark.parametrize("status", [301, 302, 303])
    def test_301_to_303_downgrade_a_post_to_get(self, status: int) -> None:
        pending = _pending("https://x/submit", {}, method="POST", body=b"{}")
        assert _follow(pending, status, {"location": "https://x/land"})
        assert (pending.method, pending.body) == ("GET", b"")

    @pytest.mark.parametrize("status", [307, 308])
    def test_307_and_308_preserve_method_and_body(self, status: int) -> None:
        pending = _pending("https://x/submit", {}, method="POST", body=b"{}")
        assert _follow(pending, status, {"location": "https://x/land"})
        assert (pending.method, pending.body) == ("POST", b"{}")

    @pytest.mark.parametrize("status", [301, 302, 303])
    def test_a_get_keeps_its_method_and_origin(self, status: int) -> None:
        pending = _pending("https://a.com/1", {"Origin": "https://a.com"}, body=b"kept")
        assert _follow(pending, status, {"location": "https://a.com/2"})
        assert (pending.method, pending.body) == ("GET", b"kept")
        assert pending.headers == {"Origin": "https://a.com"}

    def test_cross_origin_drops_cookie_and_hints(self) -> None:
        pending = _pending(
            "https://a.com/1",
            {"Cookie": "SID=x", "sec-ch-ua-arch": '"x86"', "Accept": "*/*"},
        )
        assert _follow(pending, 302, {"location": "https://b.com/2"})
        assert pending.headers == {"Accept": "*/*"}

    def test_same_origin_keeps_cookie_and_hints(self) -> None:
        sent = {"Cookie": "SID=x", "sec-ch-ua-arch": '"x86"'}
        pending = _pending("https://a.com/1", dict(sent))
        assert _follow(pending, 302, {"location": "https://a.com/2"})
        assert pending.headers == sent

    @pytest.mark.parametrize("name", ["Authorization", "authorization"])
    def test_cross_origin_drops_authorization(self, name: str) -> None:
        """A credential is origin-bound exactly like a cookie.

        Forwarding it hands the source origin's secret to whatever host answered
        the pending. Both casings: HTTP field names are case-insensitive.
        """
        pending = _pending("https://a.com/1", {name: "Bearer secret", "Accept": "*/*"})
        assert _follow(pending, 307, {"location": "https://evil.example/2"})
        assert pending.headers == {"Accept": "*/*"}

    def test_same_origin_keeps_authorization(self) -> None:
        """A same-origin hop is still the origin the credential belongs to."""
        pending = _pending("https://a.com/1", {"Authorization": "Bearer secret"})
        assert _follow(pending, 307, {"location": "https://a.com/2"})
        assert pending.headers == {"Authorization": "Bearer secret"}

    def test_same_origin_sends_the_cookies_the_hop_set(self) -> None:
        pending = _pending(
            "https://a.com/1",
            {"Accept": "*/*", "cookie": "old=1; keep=2"},
        )
        assert _follow(
            pending,
            302,
            {"location": "/2", "set-cookie": "old=new; Path=/\nfresh=3"},
        )
        assert pending.headers == {
            "Accept": "*/*",
            "cookie": "old=new; keep=2; fresh=3",
        }

    def test_same_origin_set_cookie_starts_a_cookie_header(self) -> None:
        pending = _pending("https://a.com/1", {"Accept": "*/*"})
        assert _follow(pending, 302, {"location": "/2", "set-cookie": "s=1"})
        assert pending.headers == {"Accept": "*/*", "Cookie": "s=1"}

    def test_cross_origin_sends_no_cookie_the_hop_set(self) -> None:
        pending = _pending("https://a.com/1", {"Accept": "*/*"})
        assert _follow(
            pending,
            302,
            {"location": "https://b.com/", "set-cookie": "s=1"},
        )
        assert pending.headers == {"Accept": "*/*"}

    def test_cross_origin_sends_the_targets_cookies_alone(self) -> None:
        asked: list[str] = []

        def stored(host: str) -> Mapping[str, str]:
            asked.append(host)
            return {"B": "1"}

        pending = _pending("https://a.com/1", {"Accept": "*/*", "cookie": "A=1"})
        assert _follow(
            pending,
            302,
            {"location": "https://b.com:8443/2"},
            cookies_for_host=stored,
        )
        assert asked == ["b.com"]
        assert pending.headers == {"Accept": "*/*", "Cookie": "B=1"}

    def test_same_origin_asks_for_no_stored_cookies(self) -> None:
        def refuse(host: str) -> Mapping[str, str]:
            raise AssertionError(host)

        pending = _pending("https://a.com/1", {"Cookie": "A=1"})
        assert _follow(pending, 302, {"location": "/2"}, cookies_for_host=refuse)
        assert pending.headers == {"Cookie": "A=1"}


class TestOriginBound:
    def test_cookie_authorization_and_every_extended_hint(self) -> None:
        hints = {name.lower() for name in chrome_client_hints(major=1)}
        assert hints
        assert {"cookie", "authorization"} | hints == ORIGIN_BOUND


class TestCookieHeader:
    def test_new_cookies_win_and_keep_the_sent_order(self) -> None:
        assert (
            cookie_header({"COOKIE": "a=1; b=2"}, {"b": "3", "c": "4"})
            == "a=1; b=3; c=4"
        )

    def test_no_cookie_header_and_no_cookies_is_empty(self) -> None:
        assert not cookie_header({"Accept": "*/*"}, {})

    def test_malformed_pairs_are_dropped(self) -> None:
        assert cookie_header({"Cookie": "a=1; junk; =v; b="}, {}) == "a=1; b="


class TestMergeHeaders:
    def test_keeps_the_default_slot_and_appends_new_names(self) -> None:
        merged = merge_headers(
            {"A": "1", "User-Agent": "chrome", "B": "2"},
            {"user-agent": "mine", "C": "3", "B": None},
        )
        assert list(merged.items()) == [("A", "1"), ("User-Agent", "mine"), ("C", "3")]

    def test_a_none_for_an_absent_name_is_dropped(self) -> None:
        assert merge_headers({"A": "1"}, {"X": None}) == {"A": "1"}

    def test_case_variant_overrides_collapse_to_the_last(self) -> None:
        # Two names for one header send it twice on the wire: a bot tell.
        assert merge_headers({}, {"cookie": "a=1", "Cookie": "a=1; b=2"}) == {
            "Cookie": "a=1; b=2",
        }

    def test_a_case_variant_override_of_a_default_keeps_its_slot(self) -> None:
        merged = merge_headers(
            {"A": "1", "Accept": "x", "B": "2"},
            {"accept": "y", "ACCEPT": "z"},
        )
        assert list(merged.items()) == [("A", "1"), ("Accept", "z"), ("B", "2")]


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
        assert decompress(zstd_compat.compress(data), "zstd") == data

    def test_zstd_streaming_frame_no_size(self) -> None:
        # Streaming-mode frames omit decompressed size from the header;
        # `zstandard.ZstdDecompressor.decompress()` rejects them. Real servers
        # (e.g. Cloudflare) emit such frames -- we must handle them. Neither
        # library's one-shot `compress` can mint one, so the frame is a golden.
        frame = bytes.fromhex(
            "28b52ffd0058a500006068656c6c6f20776f726c64200100d1ee7c4902",
        )
        assert not frame[4] & 0b1110_0000  # No content size, not single-segment.
        assert decompress(frame, "zstd") == b"hello world " * 1000

    @pytest.mark.parametrize(
        ("body", "encoding"),
        [(b"not gzip", "gzip"), (b"x", "compress")],
    )
    def test_an_undecodable_body_raises_its_own_type(
        self,
        body: bytes,
        encoding: str,
    ) -> None:
        # A corrupt response is a failure of that one fetch, which a caller tells
        # from a programming error by its type.
        with pytest.raises(DecompressionError):
            decompress(body, encoding)

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
    def test_host_header_omits_only_the_scheme_default_port(self) -> None:
        assert host_header("example.com", 443, "https") == "example.com"
        assert host_header("example.com", 80, "http") == "example.com"
        assert host_header("example.com", 444, "https") == "example.com:444"
        assert host_header("example.com", 443, "http") == "example.com:443"
        assert host_header("example.com", None, "https") == "example.com"

    def test_an_unknown_scheme_has_no_default_port(self) -> None:
        assert host_header("example.com", 80, "ftp") == "example.com:80"
        assert origin("ftp://example.com:80/x") == "ftp://example.com:80"

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

    def test_pinned_host_allowing_private_hosts_skips_resolution(self) -> None:
        with patch("socket.getaddrinfo", side_effect=AssertionError):
            assert pinned_host("https://example.com", allow_private_hosts=True) is None

    def test_pinned_host_public_only_returns_public_pin(self) -> None:
        with patch("socket.getaddrinfo", return_value=_addrinfo("8.8.8.8")) as resolve:
            pin = pinned_host("https://example.com", allow_private_hosts=False)
        assert pin is not None
        assert pin.ip == "8.8.8.8"
        resolve.assert_called_once_with("example.com", None)

    @pytest.mark.parametrize("allow_private_hosts", [False, True])
    @pytest.mark.parametrize("url", ["ftp://example.com/", "file:///etc/passwd"])
    def test_pinned_host_refuses_a_non_http_scheme(
        self,
        url: str,
        allow_private_hosts: bool,
    ) -> None:
        with (
            patch("socket.getaddrinfo", side_effect=AssertionError),
            pytest.raises(ValueError, match="Unsupported URL scheme"),
        ):
            pinned_host(url, allow_private_hosts=allow_private_hosts)

    def test_pinned_host_rejects_hostless_public_only_url(self) -> None:
        with pytest.raises(ValueError, match="no host"):
            pinned_host("https:///relative", allow_private_hosts=False)


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
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
