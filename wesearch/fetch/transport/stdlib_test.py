"""Tests for wesearch.fetch."""

from __future__ import annotations

from http import client
from unittest.mock import Mock, call, patch

import base64
import gzip
import socket
import ssl

from treekle.codec import from_plain

import pytest

from wesearch.fetch import (
    ContentParams,
    ObserveParams,
    PolicyParams,
    RequestParams,
    RetryParams,
    fetch,
)
from wesearch.fetch.common import ValidatedHost
from wesearch.fetch.testing import zstd_compress
from wesearch.fetch.transport.stdlib import (
    _open_connection,
    _ValidatedHTTPSConnection,
    _widen_after_connect,
    fetch_stdlib,
)
from wesearch.types.errors import (
    FetchError,
)


class TestFetchStdlibPath:
    def _mock_http_response(
        self,
        status: int = 200,
        body: bytes = b"hello",
        headers: list[tuple[str, str]] | None = None,
    ) -> Mock:
        resp = Mock(spec=client.HTTPResponse)
        resp.status = status
        resp.read.return_value = body
        resp.getheaders.return_value = headers or [
            ("content-encoding", "identity"),
        ]
        return resp

    def _mock_conn(self, resp: Mock) -> Mock:
        conn = Mock()
        conn.request = Mock()
        conn.getresponse.return_value = resp
        return conn

    def test_basic_get(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            result, _ = fetch(
                "https://example.com",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        assert result == b"hello"
        mock_conn.request.assert_called_once()
        assert mock_conn.request.call_args.args[0] == "GET"

    def test_gzipdecompression(self) -> None:
        compressed = gzip.compress(b"hello")
        resp = self._mock_http_response(
            body=compressed,
            headers=[("content-encoding", "gzip")],
        )
        mock_conn = self._mock_conn(resp)
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            assert (
                fetch(
                    "https://example.com",
                    request=RequestParams(policy=PolicyParams(transport="stdlib")),
                )[0]
                == b"hello"
            )

    def test_post_with_data(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    content=ContentParams(method="POST", data={"q": "test"}),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        assert mock_conn.request.call_args.args[0] == "POST"
        assert mock_conn.request.call_args.kwargs["body"] == b"q=test"

    def test_post_with_json(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    content=ContentParams(method="POST", json={"key": "value"}),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        assert mock_conn.request.call_args.kwargs["body"] == b'{"key": "value"}'
        headers = _recorded_headers(mock_conn)
        assert headers["Content-Type"] == "application/json"

    def test_data_and_json_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            fetch(
                "https://example.com",
                request=RequestParams(
                    content=ContentParams(data={"a": "1"}, json={"b": 2}),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )

    def test_cookies_serialized(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    content=ContentParams(cookies={"a": "1", "b": "2"}),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        headers = _recorded_headers(mock_conn)
        assert "a=1" in headers["Cookie"]
        assert "b=2" in headers["Cookie"]

    def test_custom_headers_override_defaults(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    content=ContentParams(headers={"User-Agent": "custom"}),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        headers = _recorded_headers(mock_conn)
        assert headers["User-Agent"] == "custom"

    def test_raw_headers_skip_defaults(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    content=ContentParams(
                        method="POST",
                        data={"q": "test"},
                        headers={"User-Agent": "custom"},
                        raw_headers=True,
                    ),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        # Host survives raw_headers: the pinned connection is opened to an IP,
        # so http.client would otherwise auto-generate "Host: 93.184.216.34" and
        # every vhost would serve the wrong site. raw_headers suppresses the
        # Chrome identity, not the authority the request is addressed to.
        assert mock_conn.request.call_args.kwargs["headers"] == {
            "Host": "example.com",
            "User-Agent": "custom",
        }

    def test_raw_headers_still_add_cookies_and_auth(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://u:p@example.com",
                request=RequestParams(
                    content=ContentParams(
                        headers={"User-Agent": "custom"},
                        cookies={"a": "1"},
                        raw_headers=True,
                    ),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        headers = _recorded_headers(mock_conn)
        assert headers == {
            "Host": "example.com",
            "User-Agent": "custom",
            "Authorization": "Basic " + base64.b64encode(b"u:p").decode(),
            "Cookie": "a=1",
        }

    def test_userinfo_url_stripped_and_basic_auth_injected(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ) as mock_open:
            fetch(
                "https://u:p@example.com:8443/x",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        # ``userinfo`` stripped: the connection opens on the bare host:port, and the
        # request path carries no credentials.
        assert mock_open.call_args.args[1] == "example.com"
        assert mock_open.call_args.kwargs["port"] == 8443
        assert mock_conn.request.call_args.args[1] == "/x"
        headers = _recorded_headers(mock_conn)
        assert headers["Authorization"] == "Basic " + base64.b64encode(b"u:p").decode()

    def test_caller_authorization_wins_over_userinfo(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://u:p@example.com/",
                request=RequestParams(
                    content=ContentParams(headers={"Authorization": "Bearer xyz"}),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        headers = _recorded_headers(mock_conn)
        assert headers["Authorization"] == "Bearer xyz"

    def test_http_error_raises_fetch_error(self) -> None:
        resp = self._mock_http_response(status=403, body=b"Forbidden")
        mock_conn = self._mock_conn(resp)
        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError, match="403"),
        ):
            fetch(
                "https://example.com",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )

    def test_timeout_passed(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ) as mock_open:
            fetch(
                "https://example.com",
                request=RequestParams(
                    retry=RetryParams(timeout_sec=60),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        assert mock_open.call_args.args[2] == 60


class TestConnectionClosedOnError:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> object:
        # Stdlib path is selected per-call via transport="stdlib", not a global.
        return

    def _mock_conn(self, status: int, body: bytes = b"nope") -> Mock:
        resp = Mock(spec=client.HTTPResponse)
        resp.status = status
        resp.read.return_value = body
        resp.getheaders.return_value = [("content-encoding", "identity")]
        conn = Mock()
        conn.request = Mock()
        conn.getresponse.return_value = resp
        return conn

    def test_conn_closed_when_error_status_raises(self) -> None:
        # A non-retryable HTTP error raises from inside the connection path,
        # BEFORE the success-path close(). The self-opened socket must not leak.
        conn = self._mock_conn(404)
        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=conn,
            ),
            pytest.raises(FetchError),
        ):
            fetch(
                "https://example.com",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        conn.close.assert_called_once()

    def test_conn_closed_on_each_retried_attempt(self) -> None:
        # A retryable 500 that then succeeds opens a fresh conn per attempt;
        # the first attempt's conn must be closed before the retry, not leaked.
        resp_500 = Mock(spec=client.HTTPResponse)
        resp_500.status = 500
        resp_500.read.return_value = b"ISE"
        resp_500.getheaders.return_value = [("content-encoding", "identity")]
        resp_ok = Mock(spec=client.HTTPResponse)
        resp_ok.status = 200
        resp_ok.read.return_value = b"ok"
        resp_ok.getheaders.return_value = [("content-encoding", "identity")]
        conn1 = Mock(request=Mock())
        conn1.getresponse.return_value = resp_500
        conn2 = Mock(request=Mock())
        conn2.getresponse.return_value = resp_ok
        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                side_effect=[conn1, conn2],
            ),
            patch("wesearch.fetch.fetch.time.sleep"),
        ):
            assert (
                fetch(
                    "https://example.com",
                    request=RequestParams(
                        retry=RetryParams(retries=1),
                        policy=PolicyParams(transport="stdlib"),
                    ),
                )[0]
                == b"ok"
            )
        conn1.close.assert_called_once()


class TestFetchStdlibBackend:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> object:
        # The stdlib backend is http.client-only; each fetch call passes
        # transport="stdlib" so these redirect/error/303/validated-host tests
        # exercise the stdlib path.
        return

    def _mock_http_response(
        self,
        status: int = 200,
        body: bytes = b"hello",
        headers: list[tuple[str, str]] | None = None,
    ) -> Mock:
        resp = Mock(spec=client.HTTPResponse)
        resp.status = status
        resp.read.return_value = body
        resp.getheaders.return_value = headers or [
            ("content-encoding", "identity"),
        ]
        return resp

    def test_redirect_followed(self) -> None:
        redir_resp = self._mock_http_response(
            status=302,
            body=b"",
            headers=[("location", "https://example.com/final")],
        )
        ok_resp = self._mock_http_response(body=b"final")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [redir_resp, ok_resp]

        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            body, _ = fetch(
                "https://example.com/start",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        assert body == b"final"

    def test_on_redirect_called(self) -> None:
        redir_resp = self._mock_http_response(
            status=302,
            body=b"",
            headers=[("location", "https://example.com/final")],
        )
        ok_resp = self._mock_http_response(body=b"done")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [redir_resp, ok_resp]

        urls: list[str] = []
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com/start",
                request=RequestParams(
                    observe=ObserveParams(on_redirect=urls.append),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        assert urls == ["https://example.com/final"]

    def test_set_cookie_value_with_comma_not_missplit(self) -> None:
        # RFC 9110 exempts Set-Cookie from comma-folding. A cookie VALUE that
        # itself contains ", " must not be split into two bogus cookies. Two
        # separate Set-Cookie headers must both survive intact.
        resp = self._mock_http_response(
            body=b"ok",
            headers=[
                ("content-encoding", "identity"),
                ("Set-Cookie", "pref=a, b, c; Path=/"),
                ("Set-Cookie", "SID=xyz; Path=/"),
            ],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            _body, session = fetch(
                "https://example.com/",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        jar = session.cookies_for("https://example.com/")
        assert jar.get("pref") == "a, b, c"
        assert jar.get("SID") == "xyz"

    def test_on_redirect_raise_aborts(self) -> None:
        redir_resp = self._mock_http_response(
            status=302,
            body=b"",
            headers=[("location", "https://bad.com/sorry")],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = redir_resp

        def reject(url: str) -> None:
            raise ValueError(f"bad redirect: {url}")

        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(ValueError, match="bad redirect"),
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    observe=ObserveParams(on_redirect=reject),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )

    def test_max_redirects_zero_returns_3xx_body(self) -> None:
        resp = self._mock_http_response(
            status=302,
            body=b"redirect body",
            headers=[
                ("content-encoding", "identity"),
                ("location", "https://example.com/other"),
            ],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            result, _ = fetch(
                "https://example.com",
                request=RequestParams(
                    retry=RetryParams(max_redirects=0),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        assert result == b"redirect body"

    def test_plain_get_curl_absent_returns_3xx_body_at_cap(self) -> None:
        # REVE559-001: a plain GET at default max_redirects, curl absent -- once
        # _fetch_simple (urllib) is gone; this routes through fetch_stdlib,
        # which returns the 3xx body at the cap. The old urllib path RAISED here
        # (None-at-cap fell through to http_error_default). No conn-triggers, so
        # this is exactly the path REVE559-001 lived on.
        resp = self._mock_http_response(
            status=302,
            body=b"cap body",
            headers=[
                ("content-encoding", "identity"),
                ("location", "https://example.com/loop"),
            ],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
        ):
            result, _ = fetch(
                "https://example.com",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )  # Default max_redirects=10.
        assert result == b"cap body"

    def test_cross_host_redirect(self) -> None:
        redir_resp = self._mock_http_response(
            status=301,
            body=b"",
            headers=[("location", "https://other.com/page")],
        )
        ok_resp = self._mock_http_response(body=b"other")
        mock_conn1 = Mock()
        mock_conn1.request = Mock()
        mock_conn1.getresponse.return_value = redir_resp
        mock_conn1.close = Mock()

        mock_conn2 = Mock()
        mock_conn2.request = Mock()
        mock_conn2.getresponse.return_value = ok_resp

        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            side_effect=[mock_conn1, mock_conn2],
        ):
            body, _ = fetch(
                "https://example.com/start",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        assert body == b"other"
        mock_conn1.close.assert_called_once()

    def test_path_relative_redirect_stays_on_host(self) -> None:
        # Location "next" (no leading slash) from /base/start must resolve to
        # /base/next on the same host, not corrupt the host to "example.comnext".
        redir_resp = self._mock_http_response(
            status=302,
            body=b"",
            headers=[("location", "next")],
        )
        ok_resp = self._mock_http_response(body=b"landed")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [redir_resp, ok_resp]

        urls: list[str] = []
        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            body, _ = fetch(
                "https://example.com/base/start",
                request=RequestParams(
                    observe=ObserveParams(on_redirect=urls.append),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        assert body == b"landed"
        assert urls == ["https://example.com/base/next"]
        # Second request stays on the same connection (same host), path /base/next.
        assert mock_conn.request.call_args_list[1].args[1] == "/base/next"

    def test_cross_origin_redirect_resets_origin_header(self) -> None:
        # A POST to a.com that redirects to b.com must NOT leak Origin: a.com;
        # the header is rewritten to the new origin (never the source).
        redir_resp = self._mock_http_response(
            status=307,
            body=b"",
            headers=[("location", "https://b.com/land")],
        )
        ok_resp = self._mock_http_response(body=b"ok")
        conn_a = Mock(request=Mock(), close=Mock())
        conn_a.getresponse.return_value = redir_resp
        conn_b = Mock(request=Mock())
        conn_b.getresponse.return_value = ok_resp

        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            side_effect=[conn_a, conn_b],
        ):
            fetch(
                "https://a.com/submit",
                request=RequestParams(
                    content=ContentParams(method="POST", data={"x": "1"}),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        sent = _recorded_headers(conn_b)
        assert sent.get("Origin") != "https://a.com"
        assert sent.get("Origin") == "https://b.com"

    def test_cross_host_redirect_drops_case_varianthost_header(self) -> None:
        # CADF-003: a caller-supplied lowercase "host" must not survive a
        # cross-host redirect (HTTP field names are case-insensitive); leaking
        # the source host to the new origin is a routing/information-leak bug.
        redir_resp = self._mock_http_response(
            status=301,
            body=b"",
            headers=[("location", "https://other.com/page")],
        )
        ok_resp = self._mock_http_response(body=b"ok")
        conn_a = Mock(request=Mock(), close=Mock())
        conn_a.getresponse.return_value = redir_resp
        conn_b = Mock(request=Mock())
        conn_b.getresponse.return_value = ok_resp

        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            side_effect=[conn_a, conn_b],
        ):
            fetch(
                "https://a.com/start",
                request=RequestParams(
                    content=ContentParams(headers={"host": "a.com"}),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        sent = _recorded_headers(conn_b)
        assert not any(k.lower() == "host" and v == "a.com" for k, v in sent.items())

    def test_303_converts_post_to_get(self) -> None:
        redir_resp = self._mock_http_response(
            status=303,
            body=b"",
            headers=[("location", "https://example.com/result")],
        )
        ok_resp = self._mock_http_response(body=b"got it")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [redir_resp, ok_resp]

        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            body, _ = fetch(
                "https://example.com/submit",
                request=RequestParams(
                    content=ContentParams(method="POST", data={"x": "1"}),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        assert body == b"got it"
        second_call = mock_conn.request.call_args_list[1]
        assert second_call.args[0] == "GET"
        assert second_call.kwargs.get("body") is None

    def test_redirect_no_location_raises(self) -> None:
        resp = self._mock_http_response(
            status=302,
            body=b"",
            headers=[("content-type", "text/html")],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError, match="302"),
        ):
            fetch(
                "https://example.com",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )

    def test_http_error_raises_fetch_error(self) -> None:
        resp = self._mock_http_response(status=404, body=b"not found")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError, match="404"),
        ):
            fetch(
                "https://example.com",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )

    def test_error_body_isdecompressed(self) -> None:
        # RED: connection-path twin of the simple-path bug. A compressed error
        # body (Cloudflare 403 challenge) was stored raw in FetchError.body while
        # the success return one line away decompressed it.
        html = b"<!DOCTYPE html><html>Just a moment...</html>"
        resp = self._mock_http_response(
            status=403,
            body=zstd_compress(html),
            headers=[("content-encoding", "zstd"), ("server", "cloudflare")],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError) as exc,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        assert exc.value.body == html

    def test_undecodable_error_body_falls_back_to_raw(self) -> None:
        # An error body whose declared encoding can't decode must NOT mask the
        # HTTP error with a decompression ValueError; surface the raw bytes so
        # the original status still propagates.
        garbage = b"this is not a valid gzip stream"
        resp = self._mock_http_response(
            status=500,
            body=garbage,
            headers=[("content-encoding", "gzip")],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError) as exc,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        assert exc.value.status == 500
        assert exc.value.body == garbage

    def test_host_validation_receives_hostname_not_netloc(self) -> None:
        # INF-002: host validation must see the bare hostname, never a
        # host:port netloc -- ``getaddrinfo`` cannot resolve "example.com:8443".
        resp = self._mock_http_response()
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp
        seen: list[str] = []

        def spy(
            host: str | None,
            port: str | int | None,
            *args: int,
        ) -> list[
            tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]
        ]:
            del port, args
            seen.append(str(host))
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

        with (
            patch("socket.getaddrinfo", spy),
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
        ):
            fetch(
                "https://example.com:8443/page",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        assert seen == ["example.com"]

    def test_validatedhost_header_carries_nondefault_port(self) -> None:
        # A2: the resolver returns the bare host (contract above), but the Host
        # HEADER must still carry a non-default port -- RFC 9110 requires the
        # port in Host when it is not the scheme default, and a vhost router
        # keys on it. Dropping it sends the wrong authority.
        resp = self._mock_http_response()
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com:8443/page",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        assert _recorded_headers(mock_conn)["Host"] == "example.com:8443"

    def test_validatedhost_header_omitsdefault_port(self) -> None:
        # The converse: a default-port URL must NOT get a ":443" in Host (a real
        # browser omits the default port), else the authority still mismatches.
        resp = self._mock_http_response()
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com/page",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        assert _recorded_headers(mock_conn)["Host"] == "example.com"

    def test_cross_host_redirecthost_header_carries_nondefault_port(self) -> None:
        # A2 also applies on the REDIRECT path: a cross-host redirect to a
        # ported URL must rebuild Host WITH the port, not just the initial hop.
        # (The initial-hop and rebuild Host logic must share one rule.)
        redir = self._mock_http_response(
            status=301,
            body=b"",
            headers=[("location", "https://other.com:8443/p")],
        )
        ok = self._mock_http_response(body=b"ok")
        conn_a = Mock(request=Mock(), close=Mock())
        conn_a.getresponse.return_value = redir
        conn_b = Mock(request=Mock())
        conn_b.getresponse.return_value = ok

        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            side_effect=[conn_a, conn_b],
        ):
            fetch(
                "https://example.com/start",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        assert _recorded_headers(conn_b)["Host"] == "other.com:8443"


class TestOpenConnection:
    def test_hostname_is_not_bracketed(self) -> None:
        captured: list[str] = []

        class _Stub:
            def __init__(
                self,
                host: str,
                *,
                port: int | None = None,
                timeout: float,
                context: object = None,
            ) -> None:
                del port, timeout, context
                captured.append(host)

        with patch(
            "wesearch.fetch.transport.stdlib.client.HTTPSConnection",
            _Stub,
        ):
            _open_connection("https", "example.com", timeout_sec=10)
        assert captured == ["example.com"]

    def test_explicit_stdlib_bypasses_curl(self) -> None:
        response = Mock(spec=client.HTTPResponse)
        response.status = 200
        response.read.return_value = b"stdlib"
        response.getheaders.return_value = [("content-encoding", "identity")]
        connection = Mock()
        connection.getresponse.return_value = response
        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=connection,
            ),
            patch("curl_cffi.requests.request") as curl_request,
        ):
            body, _ = fetch(
                "https://example.com",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        assert body == b"stdlib"
        curl_request.assert_not_called()

    def test_open_connection_http_arguments(self) -> None:
        with patch(
            "wesearch.fetch.transport.stdlib.client.HTTPConnection",
        ) as ctor:
            result = _open_connection(
                "http",
                "example.com",
                timeout_sec=30,
                connect_timeout_sec=7,
                port=8080,
                resolved_ip="192.0.2.1",
            )
        assert result is ctor.return_value
        ctor.assert_called_once_with("192.0.2.1", port=8080, timeout=7)

    def test_open_connection_https_arguments_without_pin(self) -> None:
        with (
            patch(
                "wesearch.fetch.transport.stdlib.ssl.create_default_context",
            ) as tls,
            patch(
                "wesearch.fetch.transport.stdlib.client.HTTPSConnection",
            ) as ctor,
        ):
            result = _open_connection(
                "https",
                "example.com",
                timeout_sec=30,
                port=8443,
            )
        assert result is ctor.return_value
        context = tls.return_value
        tls.assert_called_once_with()
        ctor.assert_called_once_with(
            "example.com",
            port=8443,
            timeout=30,
            context=context,
        )

    def test_open_connection_https_arguments_with_pin(self) -> None:
        with (
            patch(
                "wesearch.fetch.transport.stdlib.ssl.create_default_context",
            ) as tls,
            patch(
                "wesearch.fetch.transport.stdlib._ValidatedHTTPSConnection",
            ) as ctor,
        ):
            result = _open_connection(
                "https",
                "example.com",
                timeout_sec=30,
                connect_timeout_sec=7,
                port=8443,
                resolved_ip="2001:db8::1",
            )
        assert result is ctor.return_value
        ctor.assert_called_once_with(
            "[2001:db8::1]",
            port=8443,
            server_hostname="example.com",
            timeout=7,
            context=tls.return_value,
        )

    def test_validated_https_connection_wraps_socket(self) -> None:
        context = ssl.create_default_context()
        raw_socket = socket.socket()
        wrapped_socket = Mock()
        connection = _ValidatedHTTPSConnection(
            "192.0.2.1",
            port=443,
            server_hostname="example.com",
            timeout=7,
            context=context,
        )
        try:
            connection.sock = raw_socket
            with (
                patch.object(client.HTTPConnection, "connect"),
                patch.object(
                    context,
                    "wrap_socket",
                    return_value=wrapped_socket,
                ) as wrap,
            ):
                connection.connect()
            wrap.assert_called_once_with(
                raw_socket,
                server_hostname="example.com",
            )
        finally:
            raw_socket.close()

    def test_widen_after_connect_sets_socket_timeout(self) -> None:
        connection = client.HTTPConnection("example.com")
        sock = socket.socket()
        connection.sock = sock
        try:
            _widen_after_connect(connection, 12)
            assert sock.gettimeout() == 12
        finally:
            sock.close()

    def test_widen_after_connect_without_socket_is_noop(self) -> None:
        connection = client.HTTPConnection("example.com")
        _widen_after_connect(connection, 12)


def test_fetch_stdlib_records_exact_initial_request_arguments() -> None:
    response = Mock(spec=client.HTTPResponse)
    response.status = 200
    response.read.return_value = b"raw"
    response.getheaders.return_value = [("x-test", "yes")]
    connection = Mock()
    connection.getresponse.return_value = response
    pin = ValidatedHost(host="example.com", ip="192.0.2.1")
    observed: list[tuple[int, dict[str, str], str]] = []
    with (
        patch(
            "wesearch.fetch.transport.stdlib.pinned_host",
            return_value=pin,
        ) as validate,
        patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=connection,
        ) as open_connection,
        patch(
            "wesearch.fetch.transport.stdlib._widen_after_connect",
        ) as widen,
        patch(
            "wesearch.fetch.transport.stdlib.decompress",
            return_value=b"decoded",
        ) as decode,
    ):
        result = fetch_stdlib(
            "https://example.com:80/path?q=1",
            method="POST",
            headers={"X-Test": "yes"},
            body=b"payload",
            timeout_sec=30.0,
            connect_timeout_sec=7.0,
            max_redirects=2,
            impersonate="chrome",
            on_redirect=None,
            on_response=lambda status, headers, url: observed.append(
                (status, headers, url),
            ),
        )

    assert result == b"decoded"
    validate.assert_called_once_with("https://example.com:80/path?q=1", "untrusted")
    open_connection.assert_called_once_with(
        "https",
        "example.com",
        30.0,
        connect_timeout_sec=7.0,
        port=80,
        resolved_ip="192.0.2.1",
    )
    connection.request.assert_called_once_with(
        "POST",
        "/path?q=1",
        body=b"payload",
        headers={"Host": "example.com:80", "X-Test": "yes"},
    )
    widen.assert_called_once_with(connection, 30.0)
    decode.assert_called_once_with(b"raw", "identity")
    assert observed == [(200, {"x-test": "yes"}, "https://example.com:80/path?q=1")]


def test_fetch_stdlib_records_exact_cross_origin_redirect_arguments() -> None:
    redirect = Mock(spec=client.HTTPResponse)
    redirect.status = 307
    redirect.read.return_value = b"redirect"
    redirect.getheaders.return_value = [("location", "http://other.test/new?q=2")]
    result_response = Mock(spec=client.HTTPResponse)
    result_response.status = 200
    result_response.read.return_value = b"done"
    result_response.getheaders.return_value = []
    first_connection = Mock()
    first_connection.getresponse.return_value = redirect
    second_connection = Mock()
    second_connection.getresponse.return_value = result_response
    first_pin = ValidatedHost(host="example.com", ip="192.0.2.1")
    second_pin = ValidatedHost(host="other.test", ip="192.0.2.2")
    observed: list[str] = []
    with (
        patch(
            "wesearch.fetch.transport.stdlib.pinned_host",
            side_effect=[first_pin, second_pin],
        ) as validate,
        patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            side_effect=[first_connection, second_connection],
        ) as open_connection,
        patch(
            "wesearch.fetch.transport.stdlib.apply_redirect",
            return_value=(
                {"Host": "other.test", "Origin": "http://other.test"},
                "POST",
                b"payload",
            ),
        ) as apply,
    ):
        assert (
            fetch_stdlib(
                "https://example.com/old",
                method="POST",
                headers={"Host": "example.com"},
                body=b"payload",
                timeout_sec=20.0,
                connect_timeout_sec=4.0,
                max_redirects=1,
                impersonate="chrome",
                on_redirect=None,
                on_response=lambda _status, _headers, url: observed.append(url),
                trust="internal",
            )
            == b"done"
        )

    validate.assert_any_call("https://example.com/old", "internal")
    validate.assert_any_call("http://other.test/new?q=2", "internal")
    apply.assert_called_once_with(
        "https://example.com/old",
        {"Host": "example.com"},
        "POST",
        body=b"payload",
        status=307,
        redirect_url="http://other.test/new?q=2",
    )
    assert observed == [
        "https://example.com/old",
        "http://other.test/new?q=2",
    ]
    assert open_connection.call_args_list == [
        call(
            "https",
            "example.com",
            20.0,
            connect_timeout_sec=4.0,
            port=None,
            resolved_ip="192.0.2.1",
        ),
        call(
            "http",
            "other.test",
            20.0,
            connect_timeout_sec=4.0,
            port=None,
            resolved_ip="192.0.2.2",
        ),
    ]
    second_connection.request.assert_called_once_with(
        "POST",
        "/new?q=2",
        body=b"payload",
        headers={"Host": "other.test", "Origin": "http://other.test"},
    )


def test_fetch_stdlib_classifies_status_400_with_exact_error_inputs() -> None:
    response = Mock(spec=client.HTTPResponse)
    response.status = 400
    response.read.return_value = b"compressed"
    response.getheaders.return_value = [("content-encoding", "gzip")]
    connection = Mock()
    connection.getresponse.return_value = response
    error = RuntimeError("classified")
    with (
        patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=connection,
        ),
        patch(
            "wesearch.fetch.transport.stdlib.decompress_error_body",
            return_value=b"decoded error",
        ) as decode,
        patch(
            "wesearch.fetch.transport.stdlib.classify_http_error",
            side_effect=error,
        ) as classify,
        pytest.raises(RuntimeError, match="classified"),
    ):
        fetch_stdlib(
            "https://example.com/fail",
            method="GET",
            headers={},
            body=None,
            timeout_sec=10.0,
            max_redirects=0,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
        )
    decode.assert_called_once_with(b"compressed", {"content-encoding": "gzip"})
    classify.assert_called_once_with(
        "https://example.com/fail",
        400,
        {"content-encoding": "gzip"},
        b"decoded error",
    )


def test_validated_https_connection_passes_all_constructor_arguments() -> None:
    context = ssl.create_default_context()
    with patch.object(client.HTTPSConnection, "__init__", return_value=None) as init:
        _ValidatedHTTPSConnection(
            "192.0.2.1",
            port=8443,
            server_hostname="example.com",
            timeout=7.0,
            context=context,
        )
    init.assert_called_once_with(
        "192.0.2.1",
        port=8443,
        timeout=7.0,
        context=context,
    )


def test_fetch_stdlib_uses_root_path_for_query_only_url() -> None:
    response = Mock(spec=client.HTTPResponse)
    response.status = 200
    response.read.return_value = b"ok"
    response.getheaders.return_value = []
    connection = Mock()
    connection.getresponse.return_value = response
    with patch(
        "wesearch.fetch.transport.stdlib._open_connection",
        return_value=connection,
    ):
        fetch_stdlib(
            "https://example.com?query=yes",
            method="GET",
            headers={},
            body=None,
            timeout_sec=10.0,
            max_redirects=0,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
        )
    assert connection.request.call_args.args[1] == "/?query=yes"


def test_fetch_stdlib_reopens_for_redirect_port_and_scheme_changes() -> None:
    first = Mock(spec=client.HTTPResponse)
    first.status = 302
    first.read.return_value = b"redirect"
    first.getheaders.return_value = [("location", "http://example.com:8080/new")]
    second = Mock(spec=client.HTTPResponse)
    second.status = 302
    second.read.return_value = b"redirect again"
    second.getheaders.return_value = [
        ("location", "https://example.com:8080/final"),
    ]
    third = Mock(spec=client.HTTPResponse)
    third.status = 200
    third.read.return_value = b"ok"
    third.getheaders.return_value = []
    first_connection = Mock()
    first_connection.getresponse.return_value = first
    second_connection = Mock()
    second_connection.getresponse.return_value = second
    third_connection = Mock()
    third_connection.getresponse.return_value = third
    pin = ValidatedHost(host="example.com", ip="192.0.2.1")
    with (
        patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            side_effect=[first_connection, second_connection, third_connection],
        ) as open_connection,
        patch(
            "wesearch.fetch.transport.stdlib.pinned_host",
            return_value=pin,
        ),
    ):
        assert (
            fetch_stdlib(
                "http://example.com/start",
                method="GET",
                headers={},
                body=None,
                timeout_sec=10.0,
                max_redirects=2,
                impersonate="chrome",
                on_redirect=None,
                on_response=None,
                trust="internal",
            )
            == b"ok"
        )
    assert open_connection.call_args_list == [
        call(
            "http",
            "example.com",
            10.0,
            connect_timeout_sec=None,
            port=None,
            resolved_ip="192.0.2.1",
        ),
        call(
            "http",
            "example.com",
            10.0,
            connect_timeout_sec=None,
            port=8080,
            resolved_ip="192.0.2.1",
        ),
        call(
            "https",
            "example.com",
            10.0,
            connect_timeout_sec=None,
            port=8080,
            resolved_ip="192.0.2.1",
        ),
    ]
    assert second_connection.request.call_args == call(
        "GET",
        "/new",
        body=None,
        headers={"Host": "example.com:8080"},
    )
    assert third_connection.request.call_args == call(
        "GET",
        "/final",
        body=None,
        headers={"Host": "example.com:8080"},
    )


def test_fetch_stdlib_redirect_host_header_uses_redirect_scheme() -> None:
    redirect = Mock(spec=client.HTTPResponse)
    redirect.status = 302
    redirect.read.return_value = b"redirect"
    redirect.getheaders.return_value = [
        ("location", "https://other.test:80/final"),
    ]
    result = Mock(spec=client.HTTPResponse)
    result.status = 200
    result.read.return_value = b"ok"
    result.getheaders.return_value = []
    first_connection = Mock()
    first_connection.getresponse.return_value = redirect
    second_connection = Mock()
    second_connection.getresponse.return_value = result
    pin = ValidatedHost(host="other.test", ip="192.0.2.2")
    with (
        patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            side_effect=[first_connection, second_connection],
        ),
        patch(
            "wesearch.fetch.transport.stdlib.pinned_host",
            return_value=pin,
        ),
    ):
        fetch_stdlib(
            "https://example.com/start",
            method="GET",
            headers={},
            body=None,
            timeout_sec=10.0,
            max_redirects=1,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
            trust="internal",
        )
    assert second_connection.request.call_args == call(
        "GET",
        "/final",
        body=None,
        headers={"Host": "other.test:80"},
    )


def test_fetch_stdlib_unpinned_cross_origin_redirect_uses_empty_resolution() -> None:
    first = Mock(spec=client.HTTPResponse)
    first.status = 302
    first.read.return_value = b"redirect"
    first.getheaders.return_value = [("location", "https://other.test")]
    second = Mock(spec=client.HTTPResponse)
    second.status = 200
    second.read.return_value = b"ok"
    second.getheaders.return_value = []
    first_connection = Mock()
    first_connection.getresponse.return_value = first
    second_connection = Mock()
    second_connection.getresponse.return_value = second
    with (
        patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            side_effect=[first_connection, second_connection],
        ) as open_connection,
        patch(
            "wesearch.fetch.transport.stdlib.pinned_host",
            return_value=None,
        ),
    ):
        fetch_stdlib(
            "https://example.com/start",
            method="GET",
            headers={},
            body=None,
            timeout_sec=10.0,
            max_redirects=1,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
            trust="internal",
        )
    assert open_connection.call_args_list == [
        call(
            "https",
            "example.com",
            10.0,
            connect_timeout_sec=None,
            port=None,
            resolved_ip="",
        ),
        call(
            "https",
            "other.test",
            10.0,
            connect_timeout_sec=None,
            port=None,
            resolved_ip="",
        ),
    ]
    assert second_connection.request.call_args.args[1] == "/"


def test_fetch_stdlib_redirect_budget_stops_after_one_hop() -> None:
    first = Mock(spec=client.HTTPResponse)
    first.status = 302
    first.read.return_value = b"redirect"
    first.getheaders.return_value = [("location", "https://example.com/one")]
    second = Mock(spec=client.HTTPResponse)
    second.status = 302
    second.read.return_value = b"cap"
    second.getheaders.return_value = [("location", "https://example.com/two")]
    connection = Mock()
    connection.getresponse.side_effect = [first, second]
    with patch(
        "wesearch.fetch.transport.stdlib._open_connection",
        return_value=connection,
    ):
        result = fetch_stdlib(
            "https://example.com/start",
            method="GET",
            headers={},
            body=None,
            timeout_sec=10.0,
            max_redirects=1,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
            trust="internal",
        )
    assert result == b"cap"
    assert connection.request.call_count == 2


def _recorded_headers(mock_conn: Mock) -> dict[str, str]:
    """Return the typed headers mapping recorded by a mock connection."""
    return from_plain(mock_conn.request.call_args.kwargs["headers"], dict[str, str])


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
