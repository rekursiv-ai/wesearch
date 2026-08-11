"""Unit tests for the net-log parsing helpers in chrome.capture."""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from wesearch.chrome.capture import _load_lenient, _parse_netlog


class TestParseNetlogConstantDrift:
    def test_missing_event_type_constant_raises(self, tmp_path: Path) -> None:
        # REV2A-006: if the event-type constant name drifts, `.get()` returns
        # None; `event["type"] != None` then matches NO events (silent zero
        # capture). A missing constant must raise, not degrade silently.
        netlog = {
            "constants": {"logEventTypes": {"SOME_OTHER_NAME": 42}},
            "events": [{"type": 42, "params": {"headers": [":method: GET"]}}],
        }
        p = tmp_path / "netlog.json"
        p.write_text(json.dumps(netlog))
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            _parse_netlog(p, to_origin="https://x.com")


class TestParseNetlogProtocols:
    def test_http2_event_reconstructs_url_from_pseudo_headers(
        self, tmp_path: Path
    ) -> None:
        netlog = {
            "constants": {
                "logEventTypes": {"HTTP_TRANSACTION_HTTP2_SEND_REQUEST_HEADERS": 7}
            },
            "events": [
                {
                    "type": 7,
                    "params": {
                        "headers": [
                            ":method: GET",
                            ":scheme: https",
                            ":authority: x.com",
                            ":path: /a",
                            "user-agent: Chrome",
                            "accept: */*",
                        ]
                    },
                }
            ],
        }
        p = tmp_path / "n.json"
        p.write_text(json.dumps(netlog))
        got = _parse_netlog(p, to_origin="https://x.com")
        assert len(got) == 1
        assert got[0].url == "https://x.com/a"
        assert got[0].header_names() == ("user-agent", "accept")

    def test_http1_event_reconstructs_url_from_line_and_host(
        self, tmp_path: Path
    ) -> None:
        # An HTTP/1.1 send event carries no pseudo-headers: the URL comes from
        # the request `line` plus the `Host` header. A loopback oracle served
        # only over HTTP/1.1 would be invisible if this path were unsupported.
        netlog = {
            "constants": {
                "logEventTypes": {"HTTP_TRANSACTION_SEND_REQUEST_HEADERS": 9}
            },
            "events": [
                {
                    "type": 9,
                    "params": {
                        "line": "GET /a HTTP/1.1",
                        "headers": [
                            "Host: localhost:8443",
                            "User-Agent: Chrome",
                            "Accept: */*",
                        ],
                    },
                }
            ],
        }
        p = tmp_path / "n.json"
        p.write_text(json.dumps(netlog))
        got = _parse_netlog(p, to_origin="https://localhost:8443")
        assert len(got) == 1
        assert got[0].url == "https://localhost:8443/a"
        assert got[0].header_names() == ("host", "user-agent", "accept")


class TestLoadLenient:
    def test_parses_well_formed_json(self) -> None:
        assert _load_lenient('{"a": 1}') == {"a": 1}

    def test_repairs_truncated_tail(self) -> None:
        # A net-log killed mid-write ends after a complete event's "},"; the
        # repair closes the array + object so the completed events still parse.
        raw = '{"events": [{"x": 1}, {"y": 2},'
        got = _load_lenient(raw)
        assert got["events"] == [{"x": 1}, {"y": 2}]

    def test_no_delimiter_raises_on_original_not_repair_artifact(self) -> None:
        # RED: when the truncated body has no "}," at all, rfind returns -1 and
        # the repair parses "]}" -- raising a JSONDecodeError about the ARTIFACT,
        # not the real input, masking what actually failed. The failure must
        # reflect the original body, not the "]}" the repair fabricated.
        with pytest.raises(json.JSONDecodeError) as exc:
            _load_lenient("garbage with no delimiter")
        assert exc.value.doc != "]}", (
            "repair fabricated ']}' and raised on it, hiding the real failure"
        )


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
