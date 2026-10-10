"""Tests for wesearch.paper.providers.openalex (client, filter, mapping)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import json

import pytest

from wesearch.fetch import FetchSession, RequestParams
from wesearch.paper.errors import BackendError, NotFoundError, RateLimitError
from wesearch.paper.providers import openalex
from wesearch.types.errors import FetchError


if TYPE_CHECKING:
    from collections.abc import Iterator

    from treekle import MutablePlainTree


@pytest.fixture(autouse=True)
def mock_limiter() -> Iterator[_FakeLimiter]:
    """Inject a fake shared gate so the client never waits on real time.

    Yields:
      limiter: The fake gate, exposing failures requested by the test.

    """
    limiter = _FakeLimiter()
    with patch(
        "wesearch.paper.providers.openalex.cross_process_limiter",
        return_value=limiter,
    ):
        yield limiter


class TestSearch:
    def test_happy_path_maps_records_and_total(self) -> None:
        work: dict[str, MutablePlainTree] = {
            "title": "Attention",
            "publication_year": 2017,
        }
        payload = {"meta": {"count": 42}, "results": [work, {"title": "B"}]}
        with patch(
            "wesearch.paper.providers.openalex.fetch",
            _fetch_returning(payload),
        ):
            records, total, complete = openalex.search(
                "attention",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )
        assert total == 42
        assert [r.title for r in records] == ["Attention", "B"]
        assert records[0].year == 2017
        # 2 of 42 returned: the cursor was not walked to exhaustion, and
        # ``search`` must carry that through rather than claim completeness.
        assert not complete

    def test_query_sanitizes_comma_and_pipe(self) -> None:
        fetch = _search("deep, learning | attention")
        flt = _params(fetch)["filter"]
        assert isinstance(flt, str)
        assert "title_and_abstract.search:deep  learning   attention" in flt
        assert "," not in flt.split("title_and_abstract.search:")[1]
        assert "|" not in flt

    def test_limit_caps_at_per_page_max(self) -> None:
        fetch = _search(limit=500)
        assert _params(fetch)["per-page"] == 200

    def test_limit_below_max_passthrough(self) -> None:
        fetch = _search(limit=10)
        assert _params(fetch)["per-page"] == 10

    def test_limit_none_requests_full_page(self) -> None:
        # With no limit the walker fetches one full page (the ceiling), not a
        # bare default page -- so ``per-page`` is present and equals the max.
        fetch = _search(limit=None)
        assert _params(fetch)["per-page"] == 200

    def test_filter_year_bounds_and_open_access(self) -> None:
        fetch = _search(year_from=2020, year_to=2023, open_access_only=True)
        flt = _params(fetch)["filter"]
        assert isinstance(flt, str)
        assert "from_publication_date:2020-01-01" in flt
        assert "to_publication_date:2023-12-31" in flt
        assert "open_access.is_oa:true" in flt

    def test_api_key_present_when_env_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENALEX_API_KEY", "secret")
        fetch = _search()
        assert _params(fetch)["api_key"] == "secret"

    def test_api_key_absent_when_env_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        fetch = _search()
        assert "api_key" not in _params(fetch)


class TestHeaders:
    def test_mailto_in_user_agent_when_email_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENALEX_EMAIL", "me@example.com")
        headers = openalex._headers()
        assert headers["User-Agent"] == "loop-paper (mailto:me@example.com)"
        assert headers["Accept"] == "application/json"

    def test_plain_user_agent_when_email_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENALEX_EMAIL", raising=False)
        assert openalex._headers()["User-Agent"] == "loop-paper"


class TestRequestErrors:
    def test_429_raises_rate_limit_mentions_budget(self) -> None:
        err = FetchError("u", 429, {}, b"slow down")
        with (
            patch("wesearch.paper.providers.openalex.fetch", side_effect=err),
            pytest.raises(RateLimitError) as ei,
        ):
            openalex.search(
                "x",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )
        assert "daily credit budget" in str(ei.value)

    def test_non_429_raises_backend_error_with_status(self) -> None:
        err = FetchError("u", 500, {}, b"boom")
        with (
            patch("wesearch.paper.providers.openalex.fetch", side_effect=err),
            pytest.raises(BackendError) as ei,
        ):
            openalex.search(
                "x",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )
        assert ei.value.status == 500
        assert str(ei.value) == "OpenAlex HTTP 500: boom"

    def test_timeout_raises_backend_error_status_zero(self) -> None:
        with (
            patch(
                "wesearch.paper.providers.openalex.fetch",
                side_effect=TimeoutError(),
            ),
            pytest.raises(BackendError) as ei,
        ):
            openalex.search(
                "x",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )
        assert ei.value.status == 0

    def test_oserror_raises_backend_error_status_zero(self) -> None:
        with (
            patch(
                "wesearch.paper.providers.openalex.fetch",
                side_effect=OSError("conn reset"),
            ),
            pytest.raises(BackendError) as ei,
        ):
            openalex.search(
                "x",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )
        assert ei.value.status == 0

    def test_invalid_json_raises_backend_error(self) -> None:
        with (
            patch(
                "wesearch.paper.providers.openalex.fetch",
                _RecordingFetch((b"not json", FetchSession())),
            ),
            pytest.raises(BackendError) as ei,
        ):
            openalex.search(
                "x",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )
        assert "invalid JSON" in str(ei.value)

    def test_invalid_utf8_error_body_is_replaced(self) -> None:
        err = FetchError("u", 500, {}, b"bad \xff body")
        with (
            patch("wesearch.paper.providers.openalex.fetch", side_effect=err),
            pytest.raises(BackendError, match="bad \\ufffd body"),
        ):
            openalex._get("/works", {})

    @pytest.mark.parametrize("payload", [b"[]", b"null", b'"text"'])
    def test_non_object_json_raises_backend_error(self, payload: bytes) -> None:
        # A cast let these through to an AttributeError on the first ``.get``.
        # That is outside PaperError, so a FUSED search aborted entirely and
        # discarded S2's good result instead of degrading to it.
        with (
            patch(
                "wesearch.paper.providers.openalex.fetch",
                _RecordingFetch((payload, FetchSession())),
            ),
            pytest.raises(BackendError, match="expected a JSON object"),
        ):
            openalex.search(
                "x",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )

    def test_undecodable_bytes_raise_backend_error(self) -> None:
        # json.loads raises UnicodeDecodeError, not JSONDecodeError, on bytes
        # that are not valid UTF-8 -- also outside the caught type.
        with (
            patch(
                "wesearch.paper.providers.openalex.fetch",
                _RecordingFetch((b"\xff", FetchSession())),
            ),
            pytest.raises(BackendError, match="invalid JSON"),
        ):
            openalex.search(
                "x",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )

    def test_limiter_failure_raises_backend_error(self) -> None:
        # The gate is a lock file; a raw OSError escapes every caller's
        # PaperError handler and defeats fused degradation the same way.
        limiter = _FakeLimiter()
        limiter.error = OSError("read-only file system")
        with (
            patch(
                "wesearch.paper.providers.openalex.cross_process_limiter",
                return_value=limiter,
            ),
            pytest.raises(BackendError, match="rate-limit gate"),
        ):
            openalex.search(
                "x",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )


class TestFilter:
    def test_all_none_returns_none(self) -> None:
        assert (
            openalex._filter(year_from=None, year_to=None, open_access_only=False) == ""
        )

    def test_composes_parts(self) -> None:
        flt = openalex._filter(year_from=2020, year_to=2021, open_access_only=True)
        assert flt == (
            "from_publication_date:2020-01-01,"
            "to_publication_date:2021-12-31,"
            "open_access.is_oa:true"
        )


class TestReconstructAbstract:
    def test_rebuilds_in_position_order(self) -> None:
        inverted = {"learning": [1], "Deep": [0], "rocks": [2]}
        assert openalex._reconstruct_abstract(inverted) == "Deep learning rocks"

    def test_none_returns_empty(self) -> None:
        assert openalex._reconstruct_abstract(None) == ""

    def test_empty_returns_empty(self) -> None:
        assert openalex._reconstruct_abstract({}) == ""

    def test_all_empty_positions_returns_empty(self) -> None:
        assert openalex._reconstruct_abstract({"word": []}) == ""


class TestWorkToRecord:
    def test_full_work(self) -> None:
        work: dict[str, MutablePlainTree] = {
            "title": "Attention Is All You Need",
            "authorships": [
                {"author": {"display_name": "Ashish Vaswani"}},
                {"author": {"display_name": "Noam Shazeer"}},
                {"author": {}},
            ],
            "publication_year": 2017,
            "primary_location": {"source": {"display_name": "NeurIPS"}},
            "doi": "https://doi.org/10.5555/attention",
            "ids": {"arxiv": "https://arxiv.org/abs/1706.03762"},
            "abstract_inverted_index": {"a": [0], "b": [1]},
            "cited_by_count": 100,
            "referenced_works_count": 20,
            "open_access": {"oa_url": "http://x/pdf"},
        }
        rec = openalex._work_to_record(work)
        assert rec.title == "Attention Is All You Need"
        assert rec.authors == ("Ashish Vaswani", "Noam Shazeer")
        assert rec.year == 2017
        assert rec.venue == "NeurIPS"
        assert rec.doi == "10.5555/attention"
        assert rec.arxiv_id == "1706.03762"
        assert rec.abstract == "a b"
        assert rec.citation_count == 100
        assert rec.reference_count == 20
        assert rec.open_access_pdf == "http://x/pdf"
        assert rec.sources == ("openalex",)

    def test_sparse_work(self) -> None:
        work: dict[str, MutablePlainTree] = {}
        rec = openalex._work_to_record(work)
        # Empty, not a stand-in: OpenAlex reported no title, which is not the
        # same claim as the work having none, and a fabricated string is one
        # every backend emits identically -- so fusion would read unrelated
        # papers as one.
        assert rec.title == ""
        assert rec.authors == ()
        assert rec.year is None
        assert rec.doi == ""
        assert rec.arxiv_id == ""
        assert rec.abstract == ""
        assert rec.venue == ""
        assert rec.open_access_pdf == ""

    def test_display_name_fallback_for_title(self) -> None:
        work: dict[str, MutablePlainTree] = {"display_name": "Fallback Title"}
        assert openalex._work_to_record(work).title == "Fallback Title"

    def test_doi_dx_prefix_stripped(self) -> None:
        work: dict[str, MutablePlainTree] = {"doi": "http://dx.doi.org/10.1/y"}
        assert openalex._work_to_record(work).doi == "10.1/y"

    def test_arxiv_no_match_leaves_none(self) -> None:
        work: dict[str, MutablePlainTree] = {"ids": {"arxiv": "!!!"}}
        assert openalex._work_to_record(work).arxiv_id == ""

    def test_arxiv_id_recovered_from_datacite_doi(self) -> None:
        # OpenAlex indexes an arXiv preprint as its own work whose DOI is
        # arXiv's DataCite form and whose ``ids`` carries no ``arxiv`` key. The
        # id is right there in the DOI suffix, so leaving arxiv_id None throws
        # away the only identity that joins the preprint to its published twin.
        work: dict[str, MutablePlainTree] = {
            "doi": "https://doi.org/10.48550/arxiv.2210.11934",
        }
        rec = openalex._work_to_record(work)
        assert rec.arxiv_id == "2210.11934"
        assert rec.doi == "10.48550/arxiv.2210.11934"

    def test_structured_arxiv_id_wins_over_doi_suffix(self) -> None:
        work: dict[str, MutablePlainTree] = {
            "doi": "https://doi.org/10.48550/arxiv.9999.99999",
            "ids": {"arxiv": "https://arxiv.org/abs/1706.03762"},
        }
        assert openalex._work_to_record(work).arxiv_id == "1706.03762"

    def test_non_arxiv_doi_yields_no_arxiv_id(self) -> None:
        work: dict[str, MutablePlainTree] = {"doi": "https://doi.org/10.1145/3596512"}
        assert openalex._work_to_record(work).arxiv_id == ""

    def test_datacite_doi_version_suffix_is_stripped(self) -> None:
        # S2 reports the BARE arXiv id, so a recovered "2210.11934v2" is a key
        # that joins nothing -- the exact preprint/published merge this
        # recovery exists to make possible. The structured ``ids.arxiv`` path
        # already drops the version; the DOI path must agree.
        work: dict[str, MutablePlainTree] = {
            "doi": "https://doi.org/10.48550/arXiv.2210.11934v2",
        }
        assert openalex._work_to_record(work).arxiv_id == "2210.11934"

    def test_wrong_typed_nested_fields_take_defaults(self) -> None:
        work: dict[str, MutablePlainTree] = {
            "title": "T",
            "authorships": [
                {"author": "oops"},
                "stray",
                {"author": {"display_name": "A"}},
            ],
            "ids": 3,
            "primary_location": {"source": []},
            "open_access": "no",
        }
        rec = openalex._work_to_record(work)
        assert rec.authors == ("A",)
        assert rec.arxiv_id == ""
        assert rec.venue == ""

    def test_wrong_typed_paging_fields_end_the_walk(self) -> None:
        body: dict[str, MutablePlainTree] = {"results": {"x": 1}, "meta": []}
        assert openalex._works_page_advance(body, 1, 2) is None
        body = {"results": [{}, {}], "meta": {"count": "5"}}
        assert openalex._works_page_advance(body, 1, 2) == 2


class TestReferences:
    def test_resolves_then_batches(self) -> None:
        resolve: dict[str, MutablePlainTree] = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "referenced_works": [
                        "https://openalex.org/W10",
                        "https://openalex.org/W11",
                    ],
                },
            ],
        }
        batch: dict[str, MutablePlainTree] = {
            "meta": {"count": 2},
            "results": [{"title": "ref-a"}, {"title": "ref-b"}],
        }
        fetch = _RecordingFetch(
            (json.dumps(resolve).encode(), FetchSession()),
            (json.dumps(batch).encode(), FetchSession()),
        )
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            records, complete = openalex.references("doi", "10.1/x", limit=None)
        assert [r.title for r in records] == ["ref-a", "ref-b"]
        assert complete
        # Second call resolves the referenced ids via the ``openalex:`` filter.
        flt = _params(fetch)["filter"]
        assert isinstance(flt, str)
        assert "openalex:W10|W11" in flt

    def test_unresolved_ref_ids_mark_incomplete(self) -> None:
        # B1: the seed cites 2 works, but the batch resolve returns only 1 (the
        # ``openalex:`` OR-filter silently drops an id it cannot resolve). With
        # limit=None the old code reported complete=True from the REQUESTED count,
        # hiding a short reference set -- the lying-`complete` paginate.py exists
        # to prevent. `complete` must reflect the RESOLVED records, not intent.
        resolve: dict[str, MutablePlainTree] = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "referenced_works": [
                        "https://openalex.org/W10",
                        "https://openalex.org/W11",
                    ],
                },
            ],
        }
        # count=1: OpenAlex resolved only W10, dropped W11.
        batch: dict[str, MutablePlainTree] = {
            "meta": {"count": 1},
            "results": [{"title": "ref-a"}],
        }
        fetch = _RecordingFetch(
            (json.dumps(resolve).encode(), FetchSession()),
            (json.dumps(batch).encode(), FetchSession()),
        )
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            records, complete = openalex.references("doi", "10.1/x", limit=None)
        assert len(records) == 1  # Only 1 of 2 refs resolved.
        assert not complete  # Must NOT claim complete when refs went missing.

    def test_non_string_reference_ids_are_ignored(self) -> None:
        resolve: dict[str, MutablePlainTree] = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "referenced_works": [123, "https://openalex.org/W10"],
                },
            ],
        }
        batch: dict[str, MutablePlainTree] = {
            "meta": {"count": 1},
            "results": [{"title": "ref"}],
        }
        fetch = _RecordingFetch(
            (json.dumps(resolve).encode(), FetchSession()),
            (json.dumps(batch).encode(), FetchSession()),
        )
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            records, complete = openalex.references("doi", "10.1/x", limit=None)
        assert [record.title for record in records] == ["ref"]
        assert complete
        assert _params(fetch)["filter"] == "openalex:W10"

    def test_duplicate_ref_id_still_complete(self) -> None:
        # SPEC-A: referenced_works may repeat an id. The ``openalex:`` OR-filter
        # de-dups, so the batch returns fewer records than the (dup-bearing)
        # requested list -- but every DISTINCT id resolved, so this is COMPLETE.
        # A naive ``len(records) == len(capped)`` mis-reports incomplete here.
        resolve: dict[str, MutablePlainTree] = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "referenced_works": [
                        "https://openalex.org/W10",
                        "https://openalex.org/W11",
                        "https://openalex.org/W10",  # Duplicate of the first.
                    ],
                },
            ],
        }
        # Both DISTINCT ids resolved (W10, W11); OpenAlex returns 2, not 3.
        batch: dict[str, MutablePlainTree] = {
            "meta": {"count": 2},
            "results": [{"title": "ref-a"}, {"title": "ref-b"}],
        }
        fetch = _RecordingFetch(
            (json.dumps(resolve).encode(), FetchSession()),
            (json.dumps(batch).encode(), FetchSession()),
        )
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            _, complete = openalex.references("doi", "10.1/x", limit=None)
        assert complete  # All distinct refs resolved -> complete despite dup.

    def test_limit_truncates_and_marks_incomplete(self) -> None:
        resolve: dict[str, MutablePlainTree] = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "referenced_works": [
                        "https://openalex.org/W10",
                        "https://openalex.org/W11",
                        "https://openalex.org/W12",
                    ],
                },
            ],
        }
        batch: dict[str, MutablePlainTree] = {
            "meta": {"count": 1},
            "results": [{"title": "ref-a"}],
        }
        fetch = _RecordingFetch(
            (json.dumps(resolve).encode(), FetchSession()),
            (json.dumps(batch).encode(), FetchSession()),
        )
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            _, complete = openalex.references("doi", "10.1/x", limit=1)
        assert not complete  # 3 referenced, only 1 requested.
        assert _params(fetch)["filter"] == "openalex:W10"

    def test_arxiv_seed_rejected(self) -> None:
        with pytest.raises(BackendError, match="DOIs only"):
            openalex.references("arxiv", "1706.03762", limit=None)

    def test_unknown_doi_not_found(self) -> None:
        empty: dict[str, MutablePlainTree] = {"results": []}
        with (
            patch(
                "wesearch.paper.providers.openalex.fetch",
                _fetch_returning(empty),
            ),
            pytest.raises(NotFoundError),
        ):
            openalex.references("doi", "10.1/missing", limit=None)


class TestCitations:
    def test_cites_filter_and_total(self) -> None:
        resolve: dict[str, MutablePlainTree] = {
            "results": [{"id": "https://openalex.org/W1"}],
        }
        citing: dict[str, MutablePlainTree] = {
            "meta": {"count": 500},
            "results": [{"title": "citer"}],
        }
        fetch = _RecordingFetch(
            (json.dumps(resolve).encode(), FetchSession()),
            (json.dumps(citing).encode(), FetchSession()),
        )
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            records, total, complete = openalex.citations("doi", "10.1/x", limit=1)
        assert [r.title for r in records] == ["citer"]
        assert total == 500
        assert not complete  # 1 of 500 -> more remain.
        assert _params(fetch)["filter"] == "cites:W1"

    def test_year_from_added_to_filter(self) -> None:
        resolve: dict[str, MutablePlainTree] = {
            "results": [{"id": "https://openalex.org/W1"}],
        }
        citing: dict[str, MutablePlainTree] = {"meta": {"count": 0}, "results": []}
        fetch = _RecordingFetch(
            (json.dumps(resolve).encode(), FetchSession()),
            (json.dumps(citing).encode(), FetchSession()),
        )
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            openalex.citations("doi", "10.1/x", limit=None, year_from=2020)
        flt = _params(fetch)["filter"]
        assert isinstance(flt, str)
        assert "cites:W1" in flt
        assert "from_publication_date:2020-01-01" in flt

    def test_arxiv_seed_rejected(self) -> None:
        with pytest.raises(BackendError, match="DOIs only"):
            openalex.citations("arxiv", "1706.03762", limit=None)

    def test_limit_over_page_max_paginates(self) -> None:
        # BUG 1: a limit above the 200 per-page ceiling must paginate, never
        # request per-page>200 (which OpenAlex 400s). Two 200-work pages satisfy
        # a limit of 250; no single request may set per-page above 200.
        resolve: dict[str, MutablePlainTree] = {
            "results": [{"id": "https://openalex.org/W1"}],
        }
        page1: dict[str, MutablePlainTree] = {
            "meta": {"count": 500},
            "results": [{"title": f"c{i}"} for i in range(200)],
        }
        page2: dict[str, MutablePlainTree] = {
            "meta": {"count": 500},
            "results": [{"title": f"c{200 + i}"} for i in range(200)],
        }
        fetch = _RecordingFetch(
            (json.dumps(resolve).encode(), FetchSession()),
            (json.dumps(page1).encode(), FetchSession()),
            (json.dumps(page2).encode(), FetchSession()),
        )
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            records, total, complete = openalex.citations("doi", "10.1/x", limit=250)
        assert len(records) == 250
        assert total == 500
        assert not complete
        for request in fetch.requests:
            params = request.content.params
            assert params is not None
            per_page = params.get("per-page")
            assert per_page is None or isinstance(per_page, int)
            assert per_page is None or per_page <= 200

    def test_limit_none_reports_honest_completeness(self) -> None:
        # BUG 2: with no limit, a single default page of 25 against a total of
        # 500 is NOT complete. ``complete`` must mean "cursor exhausted", never
        # be True merely because ``limit is None``.
        resolve: dict[str, MutablePlainTree] = {
            "results": [{"id": "https://openalex.org/W1"}],
        }
        # One page shorter than ``count`` -> cursor not exhausted.
        citing: dict[str, MutablePlainTree] = {
            "meta": {"count": 500},
            "results": [{"title": f"c{i}"} for i in range(200)],
        }
        fetch = _RecordingFetch(
            (json.dumps(resolve).encode(), FetchSession()),
            (json.dumps(citing).encode(), FetchSession()),
        )
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            _, total, complete = openalex.citations("doi", "10.1/x", limit=None)
        assert total == 500
        assert not complete

    def test_exact_full_page_is_complete(self) -> None:
        # BUG D: a full final page whose length equals the requested size but
        # exhausts ``count`` must report complete=True. The len>=size heuristic
        # alone lies here; exhaustion must consult ``meta.count``.
        resolve: dict[str, MutablePlainTree] = {
            "results": [{"id": "https://openalex.org/W1"}],
        }
        citing: dict[str, MutablePlainTree] = {
            "meta": {"count": 200},
            "results": [{"title": f"c{i}"} for i in range(200)],
        }
        fetch = _RecordingFetch(
            (json.dumps(resolve).encode(), FetchSession()),
            (json.dumps(citing).encode(), FetchSession()),
        )
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            records, total, complete = openalex.citations("doi", "10.1/x", limit=None)
        assert total == 200
        assert len(records) == 200
        assert complete  # Cursor exhausted: 200 of 200 returned.

    def test_multi_page_exact_total_is_complete(self) -> None:
        # Two full 200-pages reaching count=400 exactly: walking to limit=400
        # exhausts the cursor and reports complete=True (no lying full-page).
        resolve: dict[str, MutablePlainTree] = {
            "results": [{"id": "https://openalex.org/W1"}],
        }
        page1: dict[str, MutablePlainTree] = {
            "meta": {"count": 400},
            "results": [{"title": f"c{i}"} for i in range(200)],
        }
        page2: dict[str, MutablePlainTree] = {
            "meta": {"count": 400},
            "results": [{"title": f"c{200 + i}"} for i in range(200)],
        }
        fetch = _RecordingFetch(
            (json.dumps(resolve).encode(), FetchSession()),
            (json.dumps(page1).encode(), FetchSession()),
            (json.dumps(page2).encode(), FetchSession()),
        )
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            records, total, complete = openalex.citations("doi", "10.1/x", limit=400)
        assert total == 400
        assert len(records) == 400
        assert complete  # 400 of 400 -> cursor exhausted.


def _fetch_returning(payload: object) -> _RecordingFetch:
    return _RecordingFetch((json.dumps(payload).encode(), FetchSession()))


def _search(
    query: str = "attention",
    *,
    limit: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
) -> _RecordingFetch:
    """Run ``search`` with a stub fetch and return that fetch recorder."""
    fetch = _fetch_returning({"meta": {"count": 0}, "results": []})
    with patch("wesearch.paper.providers.openalex.fetch", fetch):
        openalex.search(
            query,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
        )
    return fetch


def _params(fetch: _RecordingFetch) -> dict[str, str | int]:
    """Return recorded query parameters after proving the request has them."""
    params = fetch.requests[-1].content.params
    assert params is not None
    return params


class _FakeLimiter:
    """Records gate calls and can replay a requested filesystem failure."""

    def __init__(self) -> None:
        self.error: OSError | None = None

    def acquire(self) -> None:
        if self.error is not None:
            raise self.error


class _RecordingFetch:
    """Replays typed responses or exceptions and records each request."""

    def __init__(self, *outcomes: tuple[bytes, FetchSession] | BaseException) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[RequestParams] = []
        self.urls: list[str] = []

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
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class TestExactInternals:
    def test_select_exact_default_and_extra(self) -> None:
        fields = "id,doi,ids,title,display_name,authorships,publication_year,primary_location,cited_by_count,referenced_works_count,abstract_inverted_index,open_access"
        assert openalex._select() == fields
        assert openalex._select("referenced_works") == f"{fields},referenced_works"

    def test_get_builds_exact_request(self) -> None:
        fetch = _RecordingFetch((json.dumps({"ok": True}).encode(), FetchSession()))
        with (
            patch.dict("os.environ", {"OPENALEX_API_KEY": ""}),
            patch("wesearch.paper.providers.openalex.fetch", fetch),
        ):
            result = openalex._get(
                "/works",
                {"filter": "x"},
                base="https://base",
                source="src",
                interval_sec=0.25,
                timeout_sec=3.5,
                transport="curl",
            )
        assert result == {"ok": True}
        assert fetch.requests[0].content.params == {"filter": "x"}
        assert fetch.requests[0].content.headers == {
            "Accept": "application/json",
            "User-Agent": "loop-paper",
        }
        assert fetch.requests[0].retry.timeout_sec == 3.5
        assert fetch.requests[0].policy.transport == "curl"

    def test_get_defaults_and_gate_arguments_are_exact(self) -> None:
        gate = MagicMock()
        fetch = _RecordingFetch((json.dumps({"ok": True}).encode(), FetchSession()))
        with (
            patch.dict("os.environ", {"OPENALEX_API_KEY": ""}),
            patch(
                "wesearch.paper.providers.openalex.cross_process_limiter",
                return_value=gate,
            ) as limiter,
            patch("wesearch.paper.providers.openalex.fetch", fetch),
        ):
            assert openalex._get("/works", {"filter": "x"}) == {"ok": True}
        limiter.assert_called_once_with("openalex", per_seconds=0.1)
        gate.acquire.assert_called_once_with()
        assert fetch.urls == ["https://api.openalex.org/works"]
        request = fetch.requests[0]
        assert request.content.params == {"filter": "x"}
        assert request.retry.timeout_sec == 10.0
        assert request.policy.transport == "auto"
        assert fetch.requests[0].content.headers == {
            "Accept": "application/json",
            "User-Agent": "loop-paper",
        }

    def test_get_rate_limit_message_preserves_truncated_detail(self) -> None:
        detail = b"x" * 200 + b"Z"
        err = FetchError("u", 429, {}, detail)
        with (
            patch("wesearch.paper.providers.openalex.fetch", side_effect=err),
            pytest.raises(RateLimitError) as exc,
        ):
            openalex._get("/works", {})
        assert "x" * 200 in str(exc.value)
        assert "Z" not in str(exc.value)
        assert str(exc.value) == (
            "OpenAlex rate limit / daily credit budget exhausted. Set "
            "OPENALEX_API_KEY for a higher budget, or retry after the reset "
            "(midnight UTC). " + "x" * 200
        )

    def test_get_rate_limit_message_includes_body(self) -> None:
        err = FetchError("u", 429, {}, b"detail")
        with (
            patch("wesearch.paper.providers.openalex.fetch", side_effect=err),
            pytest.raises(RateLimitError, match=r"detail"),
        ):
            openalex._get("/works", {})

    def test_get_gate_failure_has_status_zero(self) -> None:
        gate = MagicMock()
        gate.acquire.side_effect = OSError("locked")
        with (
            patch(
                "wesearch.paper.providers.openalex.cross_process_limiter",
                return_value=gate,
            ),
            pytest.raises(
                BackendError,
                match=r"^OpenAlex rate-limit gate failed: locked$",
            ) as caught,
        ):
            openalex._get("/works", {})
        assert caught.value.status == 0

    def test_get_timeout_message_is_exact(self) -> None:
        with (
            patch(
                "wesearch.paper.providers.openalex.fetch",
                side_effect=TimeoutError("slow"),
            ),
            pytest.raises(
                BackendError,
                match=r"^OpenAlex request failed \(timeout or connection error\): slow$",
            ) as caught,
        ):
            openalex._get("/works", {})
        assert caught.value.status == 0

    def test_get_non_object_error_names_actual_type(self) -> None:
        with (
            patch(
                "wesearch.paper.providers.openalex.fetch",
                _RecordingFetch((b"[]", FetchSession())),
            ),
            pytest.raises(
                BackendError,
                match=r"^OpenAlex returned list, expected a JSON object\.$",
            ),
        ):
            openalex._get("/works", {})

    def test_get_404_is_backend_error(self) -> None:
        err = FetchError("u", 404, {}, b"missing")
        with (
            patch("wesearch.paper.providers.openalex.fetch", side_effect=err),
            pytest.raises(BackendError) as exc,
        ):
            openalex._get("/works", {})
        assert exc.value.status == 404
        assert str(exc.value) == "OpenAlex HTTP 404: missing"

    def test_select_extra_is_not_added_when_empty(self) -> None:
        assert not openalex._select().endswith(",")

    def test_resolve_work_exact_request_and_error_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        fetch = _fetch_returning({"results": []})
        with (
            patch("wesearch.paper.providers.openalex.fetch", fetch),
            pytest.raises(
                NotFoundError,
                match=r"^OpenAlex has no work for doi:10\.1/missing\.$",
            ),
        ):
            openalex._resolve_work("doi", "10.1/missing", extra_select="id")
        params = fetch.requests[0].content.params
        assert params == {"filter": "doi:10.1/missing", "select": "id,id"}

    def test_resolve_work_rejects_arxiv_with_status(self) -> None:
        with pytest.raises(
            BackendError,
            match=r"^OpenAlex citation graph resolves DOIs only",
        ) as exc:
            openalex._resolve_work("arxiv", "1", extra_select="id")
        assert exc.value.status == 0
        assert str(exc.value) == (
            "OpenAlex citation graph resolves DOIs only; arXiv-id resolution is "
            "unreliable. Use the S2 source for an arXiv id, or supply the DOI."
        )

    def test_work_id_tail_handles_url_and_bare_id(self) -> None:
        assert openalex._work_id_tail("https://openalex.org/W123") == "W123"
        assert openalex._work_id_tail("W123") == "W123"

    def test_works_page_advance_boundaries(self) -> None:
        assert (
            openalex._works_page_advance({"results": [1], "meta": {"count": 1}}, 1, 1)
            is None
        )
        assert (
            openalex._works_page_advance({"results": [1], "meta": {"count": 2}}, 1, 1)
            == 2
        )
        assert (
            openalex._works_page_advance({"results": [], "meta": {"count": 2}}, 1, 1)
            is None
        )

    def test_resolve_works_batches_and_forwards_transport(self) -> None:
        payload = {"meta": {"count": 2}, "results": [{"title": "a"}, {"title": "b"}]}
        fetch = _fetch_returning(payload)
        with patch("wesearch.paper.providers.openalex.fetch", fetch):
            records = openalex._resolve_works(
                ["W1", "W2"],
                per_page_max=2,
                transport="curl",
            )
        assert [record.title for record in records] == ["a", "b"]
        assert fetch.requests[0].policy.transport == "curl"
        params = fetch.requests[0].content.params
        assert params is not None
        assert params["filter"] == "openalex:W1|W2"

    def test_paginate_works_builds_page_params_and_total(self) -> None:
        body: dict[str, MutablePlainTree] = {"meta": {"count": 1}, "results": [{}]}
        with patch.object(openalex, "_get", return_value=body) as get:
            page, total = openalex._paginate_works(
                {"filter": "x"},
                limit=None,
                transport="curl",
            )
        assert total == 1
        assert page.complete
        get.assert_called_once()
        assert get.call_args.args == (
            "/works",
            {"select": openalex._select(), "filter": "x", "page": 1, "per-page": 200},
        )
        assert get.call_args.kwargs == {"transport": "curl"}

    def test_paginate_works_limit_zero_does_not_fetch(self) -> None:
        with patch.object(openalex, "_get") as get:
            page, total = openalex._paginate_works({}, limit=0)
        assert page.entries == []
        assert total == 0
        assert page.complete
        get.assert_not_called()

    def test_resolve_works_defaults_chunk_and_forwards_limit(self) -> None:
        page = MagicMock(entries=[], complete=True)
        with patch.object(
            openalex,
            "_paginate_works",
            return_value=(page, 0),
        ) as paginate_mock:
            assert openalex._resolve_works(["W1"]) == []
        paginate_mock.assert_called_once_with(
            {"filter": "openalex:W1"},
            limit=200,
            per_page_max=200,
            transport="auto",
        )

    def test_resolve_work_default_transport_and_exact_error(self) -> None:
        with (
            patch.object(openalex, "_get", return_value={"results": []}) as get,
            pytest.raises(
                NotFoundError,
                match=r"^OpenAlex has no work for doi:10\.1/x\.$",
            ),
        ):
            openalex._resolve_work("doi", "10.1/x", extra_select="id")
        get.assert_called_once_with(
            "/works",
            {"filter": "doi:10.1/x", "select": "id,id"},
            transport="auto",
        )

    def test_search_without_filters_omits_empty_filter_prefix(self) -> None:
        page = MagicMock(entries=[], complete=True)
        with patch.object(
            openalex,
            "_paginate_works",
            return_value=(page, 0),
        ) as paginate_mock:
            openalex.search(
                "q",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )
        assert paginate_mock.call_args.args[0] == {
            "filter": "title_and_abstract.search:q",
        }

    def test_citations_empty_work_id_is_preserved(self) -> None:
        page = MagicMock(entries=[], complete=True)
        with (
            patch.object(openalex, "_resolve_work", return_value={"id": None}),
            patch.object(
                openalex,
                "_paginate_works",
                return_value=(page, 0),
            ) as paginate_mock,
        ):
            openalex.citations("doi", "10.1/x", limit=None)
        assert paginate_mock.call_args.args[0] == {"filter": "cites:"}

    def test_search_forwards_exact_pagination_contract(self) -> None:
        page = MagicMock(entries=[], complete=True)
        with patch.object(
            openalex,
            "_paginate_works",
            return_value=(page, 7),
        ) as paginate_mock:
            result = openalex.search(
                "deep, learning | models",
                limit=3,
                year_from=2020,
                year_to=2022,
                open_access_only=True,
                transport="curl",
            )
        assert result == ([], 7, True)
        paginate_mock.assert_called_once_with(
            {
                "filter": "from_publication_date:2020-01-01,to_publication_date:2022-12-31,open_access.is_oa:true,title_and_abstract.search:deep  learning   models",
            },
            limit=3,
            transport="curl",
        )

    def test_citations_forwards_seed_and_transport(self) -> None:
        page = MagicMock(entries=[], complete=True)
        with (
            patch.object(
                openalex,
                "_resolve_work",
                return_value={"id": "https://openalex.org/W1"},
            ) as resolve,
            patch.object(
                openalex,
                "_paginate_works",
                return_value=(page, 0),
            ) as paginate_mock,
        ):
            assert openalex.citations(
                "doi",
                "10.1/x",
                limit=2,
                year_from=2020,
                transport="curl",
            ) == ([], 0, True)
        resolve.assert_called_once_with(
            "doi",
            "10.1/x",
            extra_select="id",
            transport="curl",
        )
        paginate_mock.assert_called_once_with(
            {"filter": "cites:W1,from_publication_date:2020-01-01"},
            limit=2,
            transport="curl",
        )

    def test_references_forwards_seed_batch_and_transport(self) -> None:
        with (
            patch.object(
                openalex,
                "_resolve_work",
                return_value={"referenced_works": ["https://openalex.org/W1"]},
            ) as resolve,
            patch.object(openalex, "_resolve_works", return_value=[]) as batch,
        ):
            records, complete = openalex.references(
                "doi",
                "10.1/x",
                limit=2,
                transport="curl",
            )
        assert records == []
        assert not complete
        resolve.assert_called_once_with(
            "doi",
            "10.1/x",
            extra_select="referenced_works",
            transport="curl",
        )
        batch.assert_called_once_with(["W1"], transport="curl")

    def test_work_id_tail_handles_multiple_slashes(self) -> None:
        assert openalex._work_id_tail("https://openalex.org/works/W123") == "W123"

    def test_paginate_works_default_transport_is_auto(self) -> None:
        with patch.object(
            openalex,
            "_get",
            return_value={"meta": {"count": 0}, "results": []},
        ) as get:
            openalex._paginate_works({}, limit=None)
        assert get.call_args.kwargs == {"transport": "auto"}

    def test_public_default_transports_are_forwarded(self) -> None:
        page = MagicMock(entries=[], complete=True)
        with patch.object(
            openalex,
            "_paginate_works",
            return_value=(page, 0),
        ) as paginate_mock:
            openalex.search(
                "q",
                limit=None,
                year_from=None,
                year_to=None,
                open_access_only=False,
            )
        assert paginate_mock.call_args.kwargs == {"limit": None, "transport": "auto"}
        with (
            patch.object(
                openalex,
                "_resolve_work",
                return_value={"id": "https://openalex.org/W1"},
            ) as resolve,
            patch.object(
                openalex,
                "_paginate_works",
                return_value=(page, 0),
            ) as paginate_mock,
        ):
            openalex.citations("doi", "10.1/x", limit=None)
        resolve.assert_called_once_with(
            "doi",
            "10.1/x",
            extra_select="id",
            transport="auto",
        )
        assert paginate_mock.call_args.kwargs["transport"] == "auto"
        with (
            patch.object(
                openalex,
                "_resolve_work",
                return_value={"referenced_works": []},
            ) as resolve,
            patch.object(openalex, "_resolve_works", return_value=[]) as batch,
        ):
            assert openalex.references("doi", "10.1/x", limit=None) == ([], True)
        resolve.assert_called_once_with(
            "doi",
            "10.1/x",
            extra_select="referenced_works",
            transport="auto",
        )
        batch.assert_called_once_with([], transport="auto")

    def test_references_exact_limit_is_complete(self) -> None:
        with (
            patch.object(
                openalex,
                "_resolve_work",
                return_value={"referenced_works": ["W1"]},
            ),
            patch.object(
                openalex,
                "_resolve_works",
                return_value=[openalex._work_to_record({"title": "x"})],
            ),
        ):
            records, complete = openalex.references("doi", "10.1/x", limit=1)
        assert len(records) == 1
        assert complete

    def test_work_to_record_preserves_http_doi_prefixes(self) -> None:
        assert (
            openalex._work_to_record({"doi": "http://doi.org/10.1/a"}).doi == "10.1/a"
        )
        assert (
            openalex._work_to_record({"doi": "https://dx.doi.org/10.1/b"}).doi
            == "10.1/b"
        )
        assert (
            openalex._work_to_record({"doi": "http://dx.doi.org/10.1/c"}).doi
            == "10.1/c"
        )

    def test_work_to_record_skips_non_string_arxiv_id(self) -> None:
        assert openalex._work_to_record({"ids": {"arxiv": 123}}).arxiv_id == ""

    def test_work_to_record_preserves_empty_optional_fields(self) -> None:
        rec = openalex._work_to_record(
            {"doi": "", "ids": {"arxiv": ""}, "open_access": {"oa_url": ""}},
        )
        assert (rec.doi, rec.arxiv_id, rec.open_access_pdf) == ("", "", "")


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
