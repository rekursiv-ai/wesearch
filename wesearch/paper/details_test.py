"""Tests for wesearch.paper.details (metadata + citation graph)."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from unittest.mock import patch

from treekle import MutablePlainTree

import pytest

from wesearch.fetch import FetchSession
from wesearch.paper.details import (
    citations,
    metadata,
    metadata_batch,
    references,
)
from wesearch.paper.errors import PaperError
from wesearch.paper.paginate import Page
from wesearch.paper.providers import (
    openalex,
    s2,
)
from wesearch.paper.types import PaperRecord


class TestCitations:
    def test_year_filter_keeps_recent(self) -> None:
        entries: list[dict[str, MutablePlainTree]] = [
            {"isInfluential": True, "citingPaper": {"title": "new", "year": 2024}},
            {"isInfluential": False, "citingPaper": {"title": "old", "year": 2010}},
        ]

        def fake(
            path: str,
            params: dict[str, str | int],
            *,
            limit: int | None,
            keep: Callable[[dict[str, MutablePlainTree]], bool],
            session: FetchSession | None = None,
        ) -> Page:
            del path, params, limit, session
            return Page(entries=[e for e in entries if keep(e)], complete=True)

        with patch.object(s2, "paginate", side_effect=fake):
            listing = citations("doi", "10.1/x", limit=None, year_from=2020)
        assert [r.title for r in listing.records] == ["new"]

    def test_year_filter_includes_the_boundary_year(self) -> None:
        entry: dict[str, MutablePlainTree] = {
            "isInfluential": True,
            "citingPaper": {"title": "boundary", "year": 2020},
        }
        with patch.object(
            s2,
            "paginate",
            return_value=Page(entries=[entry], complete=True),
        ):
            listing = citations("doi", "10.1/x", limit=None, year_from=2020)

        assert [r.title for r in listing.records] == ["boundary"]

    def test_year_filter_excludes_a_bool_year(self) -> None:
        # ``isinstance(True, int)`` is true, so a JSON ``true`` year passed the
        # filter and was compared as the value 1.
        entries: list[dict[str, MutablePlainTree]] = [
            {"isInfluential": True, "citingPaper": {"title": "real", "year": 2024}},
            {"isInfluential": True, "citingPaper": {"title": "bogus", "year": True}},
        ]

        def fake(
            path: str,
            params: dict[str, str | int],
            *,
            limit: int | None,
            keep: Callable[[dict[str, MutablePlainTree]], bool],
            session: FetchSession | None = None,
        ) -> Page:
            del path, params, limit, session
            return Page(entries=[e for e in entries if keep(e)], complete=True)

        with patch.object(s2, "paginate", side_effect=fake):
            listing = citations("doi", "10.1/x", limit=None, year_from=1)
        assert [r.title for r in listing.records] == ["real"]

    def test_influential_only_filters(self) -> None:
        entries: list[dict[str, MutablePlainTree]] = [
            {"isInfluential": True, "citingPaper": {"title": "keep"}},
            {"isInfluential": False, "citingPaper": {"title": "drop"}},
        ]

        def fake(
            path: str,
            params: dict[str, str | int],
            *,
            limit: int | None,
            keep: Callable[[dict[str, MutablePlainTree]], bool],
            session: FetchSession | None = None,
        ) -> Page:
            del path, params, limit, session
            return Page(entries=[e for e in entries if keep(e)], complete=False)

        with patch.object(s2, "paginate", side_effect=fake):
            listing = citations("doi", "10.1/x", limit=5, influential_only=True)
        assert [r.title for r in listing.records] == ["keep"]
        assert not listing.complete  # Cursor not exhausted -> more may exist.

    def test_s2_requests_exact_citation_path_params_and_options(self) -> None:
        page = Page(entries=[], complete=True)
        with patch.object(s2, "paginate", return_value=page) as paginate:
            citations(
                "doi",
                "10.1/x",
                limit=7,
                influential_only=True,
                year_from=2020,
            )

        assert paginate.call_args.args == (
            "/paper/DOI:10.1/x/citations",
            {
                "fields": ",".join(
                    (
                        "isInfluential",
                        *(f"citingPaper.{field}" for field in s2.S2_PAPER_FIELDS),
                    ),
                ),
            },
        )
        assert paginate.call_args.kwargs["limit"] == 7
        keep = cast(
            Callable[[dict[str, MutablePlainTree]], bool],
            paginate.call_args.kwargs["keep"],
        )
        assert keep({"isInfluential": True, "citingPaper": {"year": 2020}})

    def test_openalex_citations_forwards_every_argument(self) -> None:
        records = [PaperRecord(title="citer", sources=("openalex",))]
        with patch.object(
            openalex,
            "citations",
            return_value=(records, 1, True),
        ) as citations_call:
            citations("doi", "10.1/x", limit=7, source="openalex", year_from=2020)

        assert citations_call.call_args.args == ("doi", "10.1/x")
        assert citations_call.call_args.kwargs == {
            "limit": 7,
            "year_from": 2020,
            "session": None,
        }


class TestMetadata:
    def test_single(self) -> None:
        payload: dict[str, MutablePlainTree] = {
            "title": "T",
            "externalIds": {"DOI": "10.1/x"},
        }
        with patch.object(s2, "get", return_value=payload) as get:
            rec = metadata("doi", "10.1/x")
        assert rec.title == "T"
        assert get.call_args.args[0] == "/paper/DOI:10.1/x"

    def test_single_requests_full_s2_fields(self) -> None:
        payload: dict[str, MutablePlainTree] = {"title": "T"}
        with patch.object(s2, "get", return_value=payload) as get:
            metadata("arxiv", "2312.00000")

        assert get.call_args.args == (
            "/paper/ARXIV:2312.00000",
            {"fields": s2.S2_PAPER_FIELDS_STR},
        )

    def test_batch_aligns_and_nulls(self) -> None:
        with patch.object(s2, "batch", return_value=[{"title": "A"}, None]):
            recs = metadata_batch(["DOI:1", "DOI:2"])
        assert recs[0] is not None
        assert recs[0].title == "A"
        assert recs[1] is None

    def test_batch_requests_exact_ids_fields_and_endpoint(self) -> None:
        with patch.object(s2, "batch", return_value=[]) as batch:
            metadata_batch(["DOI:1", "ARXIV:2"])

        assert batch.call_args.args == (["DOI:1", "ARXIV:2"], s2.S2_PAPER_FIELDS_STR)
        assert batch.call_args.kwargs == {"endpoint": "paper", "session": None}


class TestReferences:
    def test_maps_edges(self) -> None:
        page = Page(
            entries=[{"citedPaper": {"title": "cited"}, "isInfluential": True}],
            complete=True,
        )
        with patch.object(s2, "paginate", return_value=page) as paginate:
            listing = references("doi", "10.1/x", limit=7)
        assert [r.title for r in listing.records] == ["cited"]
        assert listing.records[0].is_influential is True
        assert paginate.call_args.args == (
            "/paper/DOI:10.1/x/references",
            {
                "fields": ",".join(
                    (
                        "isInfluential",
                        *(f"citedPaper.{field}" for field in s2.S2_PAPER_FIELDS),
                    ),
                ),
            },
        )
        assert paginate.call_args.kwargs == {"limit": 7, "session": None}

    def test_skips_empty_inner_edge(self) -> None:
        # An edge row with no inner paper object is skipped, not mapped to a stub.
        page = Page(
            entries=[{"citedPaper": {}}, {"citedPaper": {"title": "real"}}],
            complete=True,
        )
        with patch.object(s2, "paginate", return_value=page):
            listing = references("doi", "10.1/x", limit=None)
        assert [r.title for r in listing.records] == ["real"]

    def test_preserves_incomplete_page(self) -> None:
        page = Page(entries=[{"citedPaper": {"title": "real"}}], complete=False)
        with patch.object(s2, "paginate", return_value=page):
            listing = references("doi", "10.1/x", limit=None)

        assert [r.title for r in listing.records] == ["real"]
        assert listing.complete is False


class TestOpenAlexGraphSource:
    def test_references_dispatches_to_openalex(self) -> None:
        recs = [PaperRecord(title="ref", sources=("openalex",))]
        with patch.object(openalex, "references", return_value=(recs, True)) as oa_refs:
            listing = references("doi", "10.1/x", limit=7, source="openalex")
        assert [r.title for r in listing.records] == ["ref"]
        assert listing.complete
        assert oa_refs.call_args.args == ("doi", "10.1/x")
        assert oa_refs.call_args.kwargs == {"limit": 7, "session": None}

    def test_citations_dispatches_to_openalex(self) -> None:
        recs = [PaperRecord(title="citer", sources=("openalex",))]
        with patch.object(
            openalex,
            "citations",
            return_value=(recs, 500, False),
        ) as oa_cites:
            listing = citations("doi", "10.1/x", limit=1, source="openalex")
        assert [r.title for r in listing.records] == ["citer"]
        assert not listing.complete  # Total 500 > 1 returned.
        assert oa_cites.call_args.args == ("doi", "10.1/x")
        assert oa_cites.call_args.kwargs == {
            "limit": 1,
            "year_from": None,
            "session": None,
        }

    def test_openalex_citations_complete_when_all_returned(self) -> None:
        recs = [PaperRecord(title="c", sources=("openalex",))]
        with patch.object(openalex, "citations", return_value=(recs, 1, True)):
            listing = citations("doi", "10.1/x", limit=None, source="openalex")
        assert listing.complete

    def test_an_underreported_total_cannot_claim_completeness(self) -> None:
        # OpenAlex's meta.count is an estimate and can come back BELOW what the
        # walk returned. Re-deriving `complete or total <= len(records)` turned
        # the paginator's honest "not exhausted" into a claim of completeness.
        recs = [PaperRecord(title=f"c{i}", sources=("openalex",)) for i in range(200)]
        with patch.object(openalex, "citations", return_value=(recs, 150, False)):
            listing = citations("doi", "10.1/x", limit=200, source="openalex")
        assert not listing.complete

    def test_an_unknown_graph_source_is_rejected(self) -> None:
        # The dispatch is a single `if source == "openalex"`, so anything else
        # fell through and silently ran S2 -- the wrong backend, reported as if
        # it were the requested one.
        for verb in (references, citations):
            with pytest.raises(PaperError, match="Unknown citation-graph source"):
                verb(
                    "doi",
                    "10.1/x",
                    limit=1,
                    source="bogus",  # ty: ignore[invalid-argument-type] -- The negative test passes an invalid source to verify validation.  # pyright: ignore[reportArgumentType] -- The negative test passes an invalid source to verify validation.  # pyrefly: ignore[bad-argument-type] -- The negative test passes an invalid source to verify validation.
                )

    def test_influential_only_rejected_for_openalex(self) -> None:
        with pytest.raises(
            PaperError,
            match=r"^'influential_only' is S2-only; OpenAlex has no influence flag\.$",
        ):
            citations(
                "doi",
                "10.1/x",
                limit=None,
                source="openalex",
                influential_only=True,
            )


class _SessionFetch:
    """A ``fetch`` returning one fixed body and recording each call's session."""

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.sessions: list[FetchSession | None] = []

    def __call__(
        self,
        url: str,
        *,
        request: object = None,
        session: FetchSession | None = None,
    ) -> bytes:
        del url, request
        self.sessions.append(session)
        return self._body


_S2_PAGE = b'{"data": [], "next": null}'
_OA_RESOLVE = (
    b'{"results": [{"id": "W1", "referenced_works": []}], "meta": {"count": 0}}'
)


class TestSessionForwarding:
    """Every lookup reuses the caller's session for each fetch it makes."""

    def _sessions(self, owner: object, body: bytes, call: _Call) -> list[bool]:
        session = FetchSession()
        fetch = _SessionFetch(body)
        with patch.object(owner, "fetch", fetch):
            call(session)
        return [s is session for s in fetch.sessions]

    def test_metadata(self) -> None:
        def call(s: FetchSession) -> object:
            return metadata("doi", "10.1/x", session=s)

        assert self._sessions(s2, b"{}", call) == [True]

    def test_metadata_batch(self) -> None:
        def call(s: FetchSession) -> object:
            return metadata_batch(["DOI:1"], session=s)

        assert self._sessions(s2, b"[]", call) == [True]

    def test_s2_references(self) -> None:
        def call(s: FetchSession) -> object:
            return references("doi", "10.1/x", limit=1, session=s)

        assert self._sessions(s2, _S2_PAGE, call) == [True]

    def test_s2_citations(self) -> None:
        def call(s: FetchSession) -> object:
            return citations("doi", "10.1/x", limit=1, session=s)

        assert self._sessions(s2, _S2_PAGE, call) == [True]

    def test_openalex_references(self) -> None:
        def call(s: FetchSession) -> object:
            return references("doi", "10.1/x", limit=1, source="openalex", session=s)

        assert self._sessions(openalex, _OA_RESOLVE, call) == [True]

    def test_openalex_citations(self) -> None:
        def call(s: FetchSession) -> object:
            return citations("doi", "10.1/x", limit=1, source="openalex", session=s)

        assert self._sessions(openalex, _OA_RESOLVE, call) == [True, True]


type _Call = Callable[[FetchSession], object]


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
