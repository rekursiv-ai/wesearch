"""Tests for wesearch.paper.providers.s2 (client, backoff, record mapping)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import json

import pytest

from wesearch.fetch import FetchSession, RequestParams
from wesearch.paper import paginate as paper_paginate
from wesearch.paper.errors import BackendError, NotFoundError, RateLimitError
from wesearch.paper.paginate import Cursor
from wesearch.paper.providers import s2
from wesearch.types.errors import FetchError


if TYPE_CHECKING:
    from treekle.codec import MutablePlainTree


@pytest.fixture(autouse=True)
def mock_limiter() -> Iterator[_FakeLimiter]:
    """Inject a fake shared gate so the S2 client never waits on real time.

    Yields:
      limiter: The fake gate, exposing what the client asked of it.

    """
    limiter = _FakeLimiter()
    with patch(
        "wesearch.paper.providers.s2.cross_process_limiter",
        return_value=limiter,
    ):
        yield limiter


class TestFields:
    def test_author_fields_exclude_deprecated_aliases(self) -> None:
        assert "aliases" not in s2.AUTHOR_FIELDS_STR.split(",")


class TestGet:
    def test_parses_object(self) -> None:
        with patch(
            "wesearch.paper.providers.s2.fetch",
            _fetch_returning({"title": "X"}),
        ):
            assert s2.get("/paper/DOI:10.1/x", {"fields": "title"}) == {"title": "X"}

    def test_404_raises_not_found(self) -> None:
        err = FetchError("u", 404, {}, b"missing")
        with (
            patch("wesearch.paper.providers.s2.fetch", side_effect=err),
            pytest.raises(NotFoundError),
        ):
            s2.get("/paper/DOI:10.1/x", {})

    def test_bad_json_raises_backend_error(self) -> None:
        with (
            patch(
                "wesearch.paper.providers.s2.fetch",
                MagicMock(return_value=(b"not json", FetchSession())),
            ),
            pytest.raises(BackendError),
        ):
            s2.get("/paper/search", {})

    def test_timeout_raises_backend_error_status_zero(self) -> None:
        with (
            patch("wesearch.paper.providers.s2.fetch", side_effect=TimeoutError()),
            pytest.raises(BackendError) as ei,
        ):
            s2.get("/paper/search", {})
        assert ei.value.status == 0


class TestBackoff:
    def test_429_retries_then_raises_rate_limit(
        self,
        mock_limiter: _FakeLimiter,
    ) -> None:
        # Every attempt 429s: after the retry budget it surfaces RateLimitError,
        # and each retry records a growing backoff into the shared cooldown.
        err = FetchError("u", 429, {}, b"slow down")
        with (
            patch("wesearch.paper.providers.s2.fetch", side_effect=err),
            pytest.raises(RateLimitError),
        ):
            s2.get("/paper/search", {})
        # Two retries -> two backoff triggers (1s, 2s); acquire once per attempt.
        assert mock_limiter.cooldowns == [1.0, 2.0]
        assert mock_limiter.acquires == 3

    def test_429_then_success_recovers(self, mock_limiter: _FakeLimiter) -> None:
        err = FetchError("u", 429, {}, b"slow")
        ok = (json.dumps({"title": "ok"}).encode(), FetchSession())
        with patch("wesearch.paper.providers.s2.fetch", side_effect=[err, ok]):
            assert s2.get("/paper/x", {}) == {"title": "ok"}
        assert mock_limiter.cooldowns == [1.0]


class TestBatch:
    def test_aligns_and_nulls_misses(self) -> None:
        payload = [{"title": "A"}, None, {"title": "C"}]
        with patch("wesearch.paper.providers.s2.fetch", _fetch_returning(payload)):
            out = s2.batch(["DOI:1", "DOI:2", "DOI:3"], "title")
        assert out == [{"title": "A"}, None, {"title": "C"}]

    def test_empty_ids_no_fetch(self) -> None:
        with patch("wesearch.paper.providers.s2.fetch") as mock:
            assert s2.batch([], "title") == []
        mock.assert_not_called()

    def test_non_array_response_raises(self) -> None:
        # S2's batch endpoint must return an array; an object is a contract break.
        with (
            patch(
                "wesearch.paper.providers.s2.fetch",
                _fetch_returning({"unexpected": "object"}),
            ),
            pytest.raises(BackendError, match="non-array"),
        ):
            s2.batch(["DOI:1"], "title")


class TestPaginate:
    def test_walks_cursor_to_limit(self) -> None:
        pages = [
            (
                json.dumps({"data": [{"year": 2020}] * 3, "next": 3}).encode(),
                FetchSession(),
            ),
            (
                json.dumps({"data": [{"year": 2021}] * 3, "next": 6}).encode(),
                FetchSession(),
            ),
        ]
        with patch("wesearch.paper.providers.s2.fetch", side_effect=pages):
            page = s2.paginate("/paper/x/citations", {"fields": "year"}, limit=5)
        assert len(page.entries) == 5
        assert not page.complete

    def test_exhaustion_marks_complete(self) -> None:
        one = (
            json.dumps({"data": [{"year": 2020}], "next": None}).encode(),
            FetchSession(),
        )
        with patch("wesearch.paper.providers.s2.fetch", return_value=one):
            page = s2.paginate("/paper/x/references", {}, limit=10)
        assert page.complete

    def test_depth_ceiling_400_stops_with_results(self) -> None:
        first = (
            json.dumps({"data": [{"year": 2020}] * 3, "next": 3}).encode(),
            FetchSession(),
        )
        ceiling = FetchError("u", 400, {}, b"offset + limit < 10000")
        with patch(
            "wesearch.paper.providers.s2.fetch",
            side_effect=[first, ceiling],
        ):
            page = s2.paginate("/paper/x/citations", {}, limit=100)
        assert len(page.entries) == 3
        assert not page.complete

    def test_400_with_no_results_reraises(self) -> None:
        # A 400 before ANY page succeeded is a real error, not the depth ceiling.
        err = FetchError("u", 400, {}, b"bad request")
        with (
            patch("wesearch.paper.providers.s2.fetch", side_effect=err),
            pytest.raises(BackendError),
        ):
            s2.paginate("/paper/x/citations", {}, limit=100)

    def test_non_advancing_cursor_terminates(self) -> None:
        # A server regression where ``next`` does not advance past ``offset``
        # must terminate (not loop forever) and report incomplete.
        page_json = (
            json.dumps({"data": [{"year": 2020}] * 3, "next": 0}).encode(),
            FetchSession(),
        )
        with patch("wesearch.paper.providers.s2.fetch", return_value=page_json):
            page = s2.paginate("/paper/x/citations", {}, limit=100)
        assert len(page.entries) == 3
        assert not page.complete

    def test_author_papers_builds_endpoint(self) -> None:
        one = (
            json.dumps({"data": [{"title": "P"}], "next": None}).encode(),
            FetchSession(),
        )
        fetch = _RecordingFetch(one)
        with patch("wesearch.paper.providers.s2.fetch", fetch):
            page = s2.author_papers("42", limit=None)
        assert [e.get("title") for e in page.entries] == ["P"]
        assert fetch.urls[0].endswith("/author/42/papers")


class TestSearchPaginate:
    def test_walks_offset_to_total_and_reports_total(self) -> None:
        pages = [
            (
                json.dumps({"data": [{"title": "A"}] * 2, "total": 3}).encode(),
                FetchSession(),
            ),
            (
                json.dumps({"data": [{"title": "B"}], "total": 3}).encode(),
                FetchSession(),
            ),
        ]
        with patch("wesearch.paper.providers.s2.fetch", side_effect=pages):
            page, total = s2.search_paginate({"query": "x"}, limit=5)
        assert [e.get("title") for e in page.entries] == ["A", "A", "B"]
        assert total == 3
        assert page.complete

    def test_caps_page_size_at_search_ceiling(self) -> None:
        one = (
            json.dumps({"data": [{"title": "A"}], "total": 1}).encode(),
            FetchSession(),
        )
        fetch = _RecordingFetch(one)
        with patch("wesearch.paper.providers.s2.fetch", fetch):
            page, total = s2.search_paginate({"query": "x"}, limit=None)
        assert total == 1
        assert page.entries == [{"title": "A"}]
        params = fetch.requests[0].content.params
        assert params is not None
        assert params["limit"] == 100


class TestSearchTotal:
    def test_extracts_total(self) -> None:
        assert s2.search_total({"total": 42}) == 42

    def test_missing_total_defaults_zero(self) -> None:
        assert s2.search_total({}) == 0


class TestRecordMapping:
    def test_paper_record_from_full(self) -> None:
        data: dict[str, MutablePlainTree] = {
            "title": "Attention",
            "externalIds": {"DOI": "10.1/x", "ArXiv": "1706.03762"},
            "authors": [{"name": "A"}, {"name": "B"}],
            "year": 2017,
            "venue": "NIPS",
            "abstract": "text",
            "citationCount": 100,
            "referenceCount": 20,
            "openAccessPdf": {"url": "http://x/pdf"},
        }
        rec = s2.paper_record_from(data)
        assert rec.title == "Attention"
        assert rec.doi == "10.1/x"
        assert rec.arxiv_id == "1706.03762"
        assert rec.authors == ("A", "B")
        assert rec.year == 2017
        assert rec.open_access_pdf == "http://x/pdf"
        assert rec.sources == ("s2",)

    def test_author_record_dict_affiliations(self) -> None:
        data: dict[str, MutablePlainTree] = {
            "authorId": "42",
            "name": "Yoshua Bengio",
            "affiliations": [{"name": "MILA"}, "UdeM"],
            "hIndex": 200,
        }
        rec = s2.author_record_from(data)
        assert rec.author_id == "42"
        assert rec.affiliations == ("MILA", "UdeM")
        assert rec.h_index == 200

    def test_wrong_typed_fields_take_defaults(self) -> None:
        paper = s2.paper_record_from(
            {
                "title": "T",
                "externalIds": "oops",
                "authors": [{"name": "A"}, "stray", 3, {}],
                "openAccessPdf": [],
            },
        )
        assert paper.authors == ("A",)
        assert paper.doi == ""
        assert paper.open_access_pdf == ""
        author = s2.author_record_from(
            {"authorId": "1", "aliases": "x", "affiliations": ["U", 5, {"name": "M"}]},
        )
        assert author.aliases == ()
        assert author.affiliations == ("U", "M")
        assert s2.author_record_from({"affiliations": 7}).affiliations == ()

    def test_wrong_typed_totals_and_rows(self) -> None:
        assert s2.search_total({"total": "7"}) == 7
        assert s2.search_total({"total": [1]}) == 0
        assert s2.search_total({"total": True}) == 0
        assert s2._search_offset_advance({"data": "x", "total": 9}, 0, 5) is None
        assert s2._next_offset_advance({"data": {}, "next": 5}, 0, 5) is None


type _Fetch = Callable[..., tuple[bytes, FetchSession]]


class _FakeLimiter:
    """Records what the S2 client asks of its gate, instead of sleeping."""

    def __init__(self) -> None:
        self.acquires = 0
        self.cooldowns: list[float] = []

    def acquire(self) -> None:
        self.acquires += 1

    def trigger_cooldown(self, backoff_sec: float | None = None) -> None:
        assert backoff_sec is not None
        self.cooldowns.append(backoff_sec)


class _RecordingFetch:
    """A ``fetch`` stand-in that replays canned responses and records requests."""

    def __init__(self, *responses: tuple[bytes, FetchSession]) -> None:
        self._responses = list(responses)
        self.urls: list[str] = []
        self.requests: list[RequestParams] = []

    def __call__(
        self,
        url: str,
        *,
        session: FetchSession | None = None,
        request: RequestParams | None = None,
    ) -> tuple[bytes, FetchSession]:
        del session
        assert request is not None
        self.urls.append(url)
        self.requests.append(request)
        return self._responses.pop(0)


def _fetch_returning(payload: object) -> _Fetch:
    body = json.dumps(payload).encode()
    return lambda *_args, **_kwargs: (body, FetchSession())


class TestExactInternals:
    def test_headers_with_and_without_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        assert s2._headers() == {"Accept": "application/json"}
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "secret")
        assert s2._headers() == {"Accept": "application/json", "x-api-key": "secret"}

    def test_get_builds_exact_request(self) -> None:
        fetch = _RecordingFetch((json.dumps({"ok": True}).encode(), FetchSession()))
        with (
            patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": ""}),
            patch("wesearch.paper.providers.s2.fetch", fetch),
        ):
            assert s2.get(
                "/paper/x",
                {"fields": "title"},
                base="https://base",
                source="src",
                interval_sec=0.25,
                max_retries=0,
                backoff_base_sec=3.0,
                timeout_sec=4.5,
                transport="curl",
            ) == {"ok": True}
        request = fetch.requests[0]
        assert fetch.urls == ["https://base/paper/x"]
        assert request.content.params == {"fields": "title"}
        assert request.content.headers == {"Accept": "application/json"}
        assert request.retry.timeout_sec == 4.5
        assert request.policy.transport == "curl"

    def test_get_rejects_array_with_exact_error(self) -> None:
        with (
            patch("wesearch.paper.providers.s2.fetch", _fetch_returning([1])),
            pytest.raises(
                TypeError,
                match=r"^unexpected array from GET /paper/x$",
            ),
        ):
            s2.get("/paper/x", {})

    def test_attempt_uses_custom_backoff_and_budget(
        self,
        mock_limiter: _FakeLimiter,
    ) -> None:
        err = FetchError("u", 429, {}, b"slow")
        with pytest.raises(RateLimitError, match=r"^Semantic Scholar rate limit hit"):
            s2._attempt(
                lambda: (_ for _ in ()).throw(err),
                source="custom",
                interval_sec=2.0,
                max_retries=1,
                backoff_base_sec=3.0,
            )
        assert mock_limiter.cooldowns == [3.0]
        assert mock_limiter.acquires == 2

    def test_fetch_offset_page_adds_offset_and_limit(self) -> None:
        with patch("wesearch.paper.providers.s2.get", return_value={}) as get_mock:
            s2._fetch_offset_page("/x", {"fields": "a"}, 7, 9, transport="curl")
        get_mock.assert_called_once_with(
            "/x",
            {"fields": "a", "offset": 7, "limit": 9},
            transport="curl",
        )

    def test_next_offset_requires_integer_and_rows(self) -> None:
        assert s2._next_offset_advance({"next": 4, "data": [{"x": 1}]}, 0, 2) == 4
        assert s2._next_offset_advance({"next": 4, "data": []}, 0, 2) is None
        assert s2._next_offset_advance({"next": "4", "data": [{"x": 1}]}, 0, 2) is None

    def test_search_offset_advances_at_exact_total_boundary(self) -> None:
        assert s2._search_offset_advance({"total": 3, "data": [{}, {}]}, 0, 2) == 2
        assert s2._search_offset_advance({"total": 2, "data": [{}, {}]}, 0, 2) is None
        assert s2._search_offset_advance({"total": 3, "data": []}, 0, 2) is None

    def test_batch_builds_exact_post_request(self) -> None:
        fetch = _RecordingFetch((json.dumps([{"x": 1}]).encode(), FetchSession()))
        with patch("wesearch.paper.providers.s2.fetch", fetch):
            assert s2.batch(
                ["a"],
                "title",
                endpoint="author",
                base="https://base",
                transport="curl",
            ) == [{"x": 1}]
        request = fetch.requests[0]
        assert fetch.urls == ["https://base/author/batch"]
        assert request.content.method == "POST"
        assert request.content.params == {"fields": "title"}
        assert request.content.json == {"ids": ["a"]}
        assert request.policy.transport == "curl"

    def test_paper_record_from_preserves_all_fields_and_influential(self) -> None:
        rec = s2.paper_record_from(
            {
                "title": "t",
                "externalIds": {"DOI": "d", "ArXiv": "a"},
                "authors": [{"name": "n"}],
                "year": 2020,
                "venue": "v",
                "abstract": "x",
                "citationCount": 3,
                "referenceCount": 4,
                "openAccessPdf": {"url": "p"},
            },
            sources=("x",),
            is_influential=True,
        )
        assert (
            rec.title,
            rec.authors,
            rec.year,
            rec.venue,
            rec.doi,
            rec.arxiv_id,
            rec.abstract,
            rec.citation_count,
            rec.reference_count,
            rec.open_access_pdf,
            rec.sources,
            rec.is_influential,
        ) == ("t", ("n",), 2020, "v", "d", "a", "x", 3, 4, "p", ("x",), True)

    def test_author_record_from_preserves_optional_fields(self) -> None:
        rec = s2.author_record_from(
            {
                "authorId": "a",
                "name": "n",
                "aliases": ["old"],
                "affiliations": [" U ", {"affiliation": "V"}],
                "homepage": "h",
                "hIndex": 1,
                "citationCount": 2,
                "paperCount": 3,
            },
        )
        assert (
            rec.author_id,
            rec.name,
            rec.aliases,
            rec.affiliations,
            rec.homepage,
            rec.h_index,
            rec.citation_count,
            rec.paper_count,
        ) == ("a", "n", ("old",), ("U", "V"), "h", 1, 2, 3)

    def test_author_record_from_empty_values(self) -> None:
        rec = s2.author_record_from(
            {
                "affiliations": [{"name": ""}, {"affiliation": ""}],
                "homepage": "",
            },
        )
        assert (
            rec.author_id,
            rec.name,
            rec.aliases,
            rec.affiliations,
            rec.homepage,
        ) == ("", "(unknown)", (), (), "")

    def test_paper_record_from_empty_values(self) -> None:
        rec = s2.paper_record_from(
            {
                "externalIds": {"DOI": "", "ArXiv": ""},
                "authors": [{"name": ""}],
                "venue": "",
                "abstract": "",
                "openAccessPdf": {"url": ""},
            },
        )
        assert (
            rec.title,
            rec.authors,
            rec.venue,
            rec.doi,
            rec.arxiv_id,
            rec.abstract,
            rec.open_access_pdf,
        ) == ("", (), "", "", "", "", "")

    def test_author_record_from_ignores_non_string_homepage(self) -> None:
        assert s2.author_record_from({"homepage": 1}).homepage == ""

    def test_get_forwards_every_attempt_option(self) -> None:
        with patch.object(s2, "_attempt", return_value=b"{}") as attempt:
            s2.get(
                "/paper/x",
                {"fields": "title"},
                base="https://base",
                source="custom",
                interval_sec=2.5,
                max_retries=4,
                backoff_base_sec=3.5,
                timeout_sec=5.5,
                transport="curl",
            )
        assert attempt.call_args.kwargs["source"] == "custom"
        assert attempt.call_args.kwargs["interval_sec"] == 2.5
        assert attempt.call_args.kwargs["max_retries"] == 4
        assert attempt.call_args.kwargs["backoff_base_sec"] == 3.5

    def test_batch_forwards_every_attempt_option_and_scalar_miss(self) -> None:
        payload = json.dumps([{"x": 1}, 7, None]).encode()
        with patch.object(s2, "_attempt", return_value=payload) as attempt:
            result = s2.batch(
                ["a", "b", "c"],
                "title",
                endpoint="author",
                base="https://base",
                source="custom",
                interval_sec=2.5,
                max_retries=4,
                backoff_base_sec=3.5,
                timeout_sec=5.5,
                transport="curl",
            )
        assert result == [{"x": 1}, None, None]
        assert attempt.call_args.kwargs["source"] == "custom"
        assert attempt.call_args.kwargs["interval_sec"] == 2.5
        assert attempt.call_args.kwargs["max_retries"] == 4
        assert attempt.call_args.kwargs["backoff_base_sec"] == 3.5

    def test_get_default_request_contract(self) -> None:
        fetch = _RecordingFetch((json.dumps({"ok": True}).encode(), FetchSession()))
        with patch("wesearch.paper.providers.s2.fetch", fetch):
            assert s2.get("/paper/x", {}) == {"ok": True}
        request = fetch.requests[0]
        assert fetch.urls == ["https://api.semanticscholar.org/graph/v1/paper/x"]
        assert request.retry.timeout_sec == 10.0
        assert request.policy.transport == "auto"

    def test_get_default_attempt_options(self) -> None:
        with patch.object(s2, "_attempt", return_value=b"{}") as attempt:
            s2.get("/paper/x", {})
        assert attempt.call_args.kwargs == {
            "source": "s2",
            "interval_sec": 1.0,
            "max_retries": 2,
            "backoff_base_sec": 1.0,
        }

    def test_get_invalid_json_error_names_path(self) -> None:
        with (
            patch(
                "wesearch.paper.providers.s2.fetch",
                _RecordingFetch((b"bad", FetchSession())),
            ),
            pytest.raises(
                BackendError,
                match=r"^Semantic Scholar returned invalid JSON for /paper/x:",
            ),
        ):
            s2.get("/paper/x", {})

    def test_batch_default_attempt_options(self) -> None:
        with patch.object(s2, "_attempt", return_value=b"[]") as attempt:
            assert s2.batch(["a"], "title") == []
        assert attempt.call_args.kwargs == {
            "source": "s2",
            "interval_sec": 1.0,
            "max_retries": 2,
            "backoff_base_sec": 1.0,
        }

    def test_batch_default_request_options(self) -> None:
        fetch = _RecordingFetch((b"[]", FetchSession()))
        with (
            patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": ""}),
            patch("wesearch.paper.providers.s2.fetch", fetch),
        ):
            s2.batch(["a"], "title")
        request = fetch.requests[0]
        assert request.retry.timeout_sec == 10.0
        assert request.policy.transport == "auto"

    def test_batch_invalid_json_error_names_endpoint(self) -> None:
        with (
            patch(
                "wesearch.paper.providers.s2.fetch",
                _RecordingFetch((b"bad", FetchSession())),
            ),
            pytest.raises(
                BackendError,
                match=r"^Semantic Scholar returned invalid JSON for /paper/batch:",
            ),
        ):
            s2.batch(["a"], "title")

    def test_batch_request_forwards_timeout_and_headers(self) -> None:
        fetch = _RecordingFetch((json.dumps([{"x": 1}]).encode(), FetchSession()))
        with (
            patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": ""}),
            patch("wesearch.paper.providers.s2.fetch", fetch),
        ):
            s2.batch(["a"], "title", timeout_sec=6.5, transport="curl")
        request = fetch.requests[0]
        assert fetch.urls == ["https://api.semanticscholar.org/graph/v1/paper/batch"]
        assert request.retry.timeout_sec == 6.5
        assert request.content.headers == {"Accept": "application/json"}
        assert request.policy.transport == "curl"

    def test_attempt_default_gate_arguments(self) -> None:
        limiter = MagicMock()
        with patch.object(s2, "cross_process_limiter", return_value=limiter) as gate:
            assert s2._attempt(lambda: b"ok") == b"ok"
        gate.assert_called_once_with("s2", per_seconds=1.0)
        limiter.acquire.assert_called_once_with()

    def test_attempt_negative_budget_has_exact_terminal_error(self) -> None:
        with pytest.raises(
            AssertionError,
            match=r"^_attempt retry loop exited without returning$",
        ):
            s2._attempt(lambda: b"ok", max_retries=-1)

    def test_attempt_translates_rate_limit_with_exact_message(self) -> None:
        err = FetchError("u", 429, {}, b"slow")
        with pytest.raises(
            RateLimitError,
            match=r"^Semantic Scholar rate limit hit \(shared 1 req/sec gate\)\. Set SEMANTIC_SCHOLAR_API_KEY for a higher tier or retry shortly\.",
        ):
            s2._attempt(lambda: (_ for _ in ()).throw(err), max_retries=0)

    def test_attempt_translates_connection_error_with_exact_message(self) -> None:
        with pytest.raises(
            BackendError,
            match=r"^Semantic Scholar request failed \(timeout or connection error\): down$",
        ):
            s2._attempt(lambda: (_ for _ in ()).throw(TimeoutError("down")))

    def test_attempt_rate_limit_uses_backend_name(self) -> None:
        err = FetchError("u", 500, {}, b"broken")
        with pytest.raises(BackendError, match=r"^Semantic Scholar HTTP 500: broken$"):
            s2._attempt(lambda: (_ for _ in ()).throw(err), max_retries=0)

    def test_attempt_default_retry_budget_and_backoff(
        self,
        mock_limiter: _FakeLimiter,
    ) -> None:
        errors = iter([FetchError("u", 429, {}, b"slow")] * 3 + [None])

        def do_fetch() -> bytes:
            error = next(errors)
            if error is not None:
                raise error
            return b"ok"

        with pytest.raises(RateLimitError):
            s2._attempt(do_fetch)
        assert mock_limiter.acquires == 3
        assert mock_limiter.cooldowns == [1.0, 2.0]

    def test_loads_invalid_json_preserves_context(self) -> None:
        with pytest.raises(
            BackendError,
            match=r"^Semantic Scholar returned invalid JSON for /paper/x:.*$",
        ):
            s2._loads(b"not json", "/paper/x")

    def test_search_offset_missing_total_stops(self) -> None:
        assert s2._search_offset_advance({"data": [{}]}, 0, 1) is None

    def test_paginate_forwards_transport_and_limit(self) -> None:
        captured: list[object] = []

        def fake_paginate(cursor: object, *, limit: int | None, keep: object) -> object:
            captured.extend((cursor, limit, keep))
            assert isinstance(cursor, Cursor)
            assert cursor.page_size_max == 1000
            with patch.object(
                s2,
                "get",
                return_value={"data": [], "next": None},
            ) as get:
                cursor.fetch(0, 2)
            get.assert_called_once_with(
                "/x",
                {"fields": "title", "offset": 0, "limit": 2},
                transport="curl",
            )
            return object()

        def keep(row: dict[str, MutablePlainTree]) -> bool:
            del row
            return True

        with patch.object(paper_paginate, "paginate", side_effect=fake_paginate):
            result = s2.paginate(
                "/x",
                {"fields": "title"},
                limit=3,
                keep=keep,
                transport="curl",
            )
        assert result is not captured[0]
        assert captured[1:] == [3, keep]

    def test_paginate_default_transport(self) -> None:
        def fetch_page(cursor: Cursor, **_: object) -> object:
            return cursor.fetch(0, 1)

        with (
            patch.object(
                paper_paginate,
                "paginate",
                side_effect=fetch_page,
            ),
            patch.object(
                s2,
                "get",
                return_value={"data": [], "next": None},
            ) as get,
        ):
            s2.paginate("/x", {}, limit=1)
        get.assert_called_once_with("/x", {"offset": 0, "limit": 1}, transport="auto")

    def test_search_paginate_forwards_path_params_and_transport(self) -> None:
        captured: list[object] = []

        def fake_paginate(cursor: object, *, limit: int | None, keep: object) -> object:
            captured.extend((cursor, limit, keep))
            assert isinstance(cursor, Cursor)
            assert cursor.is_depth_ceiling(BackendError("x", status=400))
            assert not cursor.is_depth_ceiling(BackendError("x", status=401))
            with patch.object(
                s2,
                "get",
                return_value={"data": [{}], "total": 1},
            ) as get:
                cursor.fetch(0, 7)
            get.assert_called_once_with(
                "/paper/search",
                {"query": "x", "offset": 0, "limit": 7},
                transport="curl",
            )
            return object()

        with patch.object(paper_paginate, "paginate", side_effect=fake_paginate):
            result, total = s2.search_paginate(
                {"query": "x"},
                limit=4,
                transport="curl",
            )
        assert result is not captured[0]
        assert total == 1
        assert captured[1] == 4

    def test_search_paginate_default_transport(self) -> None:
        def fetch_cursor(cursor: object, **_: object) -> object:
            assert isinstance(cursor, Cursor)
            with patch.object(s2, "get", return_value={"data": [], "total": 0}) as get:
                cursor.fetch(0, 1)
            get.assert_called_once_with(
                "/paper/search",
                {"query": "x", "offset": 0, "limit": 1},
                transport="auto",
            )
            return object()

        with patch.object(paper_paginate, "paginate", side_effect=fetch_cursor):
            s2.search_paginate({"query": "x"}, limit=1)

    def test_search_paginate_missing_total_returns_zero(self) -> None:
        def fetch_page(cursor: Cursor, **_: object) -> object:
            return cursor.fetch(0, 1)

        with (
            patch.object(s2, "get", return_value={"data": [{}]}),
            patch.object(
                paper_paginate,
                "paginate",
                side_effect=fetch_page,
            ),
        ):
            _, total = s2.search_paginate({"query": "x"}, limit=1)
        assert total == 0

    def test_search_paginate_initial_total_and_default_transport(self) -> None:
        def no_fetch(cursor: object, **_: object) -> object:
            assert isinstance(cursor, Cursor)
            assert cursor.page_size_max == 100
            return object()

        with patch.object(paper_paginate, "paginate", side_effect=no_fetch):
            result, total = s2.search_paginate({"query": "x"}, limit=1)
        assert result is not None
        assert total == 0

    def test_author_papers_forwards_exact_arguments(self) -> None:
        def keep(row: dict[str, MutablePlainTree]) -> bool:
            del row
            return False

        with patch.object(s2, "paginate", return_value=object()) as paginate:
            result = s2.author_papers("a", limit=4, keep=keep, transport="curl")
        assert result is paginate.return_value
        assert paginate.call_args.args == (
            "/author/a/papers",
            {"fields": s2.S2_PAPER_FIELDS_STR},
        )
        assert paginate.call_args.kwargs["limit"] == 4
        assert paginate.call_args.kwargs["keep"] is keep
        assert paginate.call_args.kwargs["transport"] == "curl"

    def test_author_papers_default_transport(self) -> None:
        with patch.object(s2, "paginate", return_value=object()) as paginate:
            s2.author_papers("a", limit=None)
        assert paginate.call_args.kwargs["transport"] == "auto"


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
