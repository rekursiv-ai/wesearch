"""Tests for wesearch.paper.authors (author search / metadata / papers)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from wesearch.fetch import FetchSession
from wesearch.paper.authors import author_metadata, author_papers, search_authors
from wesearch.paper.paginate import Page
from wesearch.paper.providers import s2
from wesearch.paper.types import AuthorRecord


if TYPE_CHECKING:
    from collections.abc import Callable

    from treekle import MutablePlainTree


class TestAuthors:
    def test_search_sorts_by_h_index_and_caps(self) -> None:
        payload: dict[str, MutablePlainTree] = {
            "total": 3,
            "data": [
                {"authorId": "1", "name": "Low", "hIndex": 5},
                {"authorId": "2", "name": "High", "hIndex": 90},
                {"authorId": "3", "name": "Mid", "hIndex": 40},
            ],
        }
        with patch.object(s2, "get", return_value=payload):
            result = search_authors("x", limit=2)
        assert [r.name for r in result.records] == [
            "High",
            "Mid",
        ]  # h-index desc, capped.
        assert result.total == 3

    def test_author_metadata_batch(self) -> None:
        with patch.object(
            s2,
            "batch",
            return_value=[{"authorId": "1", "name": "A"}, None],
        ) as batch:
            recs = author_metadata(["1", "2"])
        assert isinstance(recs[0], AuthorRecord)
        assert recs[1] is None
        assert batch.call_args.args == (["1", "2"], s2.AUTHOR_FIELDS_STR)
        assert batch.call_args.kwargs == {"endpoint": "author", "session": None}

    def test_search_authors_requests_exact_path_and_params(self) -> None:
        payload: dict[str, MutablePlainTree] = {
            "total": 0,
            "data": [],
        }
        with patch.object(s2, "get", return_value=payload) as get:
            search_authors("Ada", limit=None)
        assert get.call_args.args == (
            "/author/search",
            {"query": "Ada", "fields": s2.AUTHOR_FIELDS_STR},
        )
        assert get.call_args.kwargs == {"session": None}

    def test_search_authors_sorts_unknown_h_index_last(self) -> None:
        payload: dict[str, MutablePlainTree] = {
            "total": 3,
            "data": [
                {"authorId": "1", "name": "Unknown"},
                {"authorId": "2", "name": "Zero", "hIndex": 0},
                {"authorId": "3", "name": "One", "hIndex": 1},
            ],
        }
        with patch.object(s2, "get", return_value=payload):
            result = search_authors("x", limit=None)
        assert [record.name for record in result.records] == ["One", "Zero", "Unknown"]

    @pytest.mark.parametrize(
        ("limit", "year_from", "year_to", "expected"),
        [
            (7, 2020, None, ["new", "boundary"]),
            (None, 1, 9999, ["new", "boundary", "old"]),
            (None, None, None, ["new", "boundary", "old", "bogus", "undated"]),
            (None, None, 2020, ["boundary", "old"]),
        ],
    )
    def test_author_papers_year_filter(
        self,
        limit: int | None,
        year_from: int | None,
        year_to: int | None,
        expected: list[str],
    ) -> None:
        entries: list[dict[str, MutablePlainTree]] = [
            {"title": "new", "year": 2024},
            {"title": "boundary", "year": 2020},
            {"title": "old", "year": 2000},
            {"title": "bogus", "year": True},
            {"title": "undated"},
        ]

        def fake(
            author_id: str,
            *,
            limit: int | None,
            keep: Callable[[dict[str, MutablePlainTree]], bool],
            session: FetchSession | None = None,
        ) -> Page:
            del author_id, limit, session
            return Page(entries=[e for e in entries if keep(e)], complete=True)

        with patch.object(s2, "author_papers", side_effect=fake) as papers:
            listing = author_papers(
                "1",
                limit=limit,
                year_from=year_from,
                year_to=year_to,
            )
        assert [r.title for r in listing.records] == expected
        assert papers.call_args.args == ("1",)
        assert papers.call_args.kwargs["limit"] == limit
        assert papers.call_args.kwargs["keep"] is not None

    def test_author_papers_preserves_completeness(self) -> None:
        with patch.object(
            s2,
            "author_papers",
            return_value=Page(entries=[], complete=False),
        ):
            listing = author_papers("1", limit=2)
        assert listing.complete is False


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


class TestSessionForwarding:
    """Every author call reuses the caller's session for its fetch."""

    def _sessions(
        self,
        body: bytes,
        call: Callable[[FetchSession], object],
    ) -> list[bool]:
        session = FetchSession()
        fetch = _SessionFetch(body)
        with patch.object(s2, "fetch", fetch):
            call(session)
        return [s is session for s in fetch.sessions]

    def test_search_authors(self) -> None:
        def call(s: FetchSession) -> object:
            return search_authors("ada", limit=None, session=s)

        assert self._sessions(b'{"data": []}', call) == [True]

    def test_author_metadata(self) -> None:
        def call(s: FetchSession) -> object:
            return author_metadata(["1"], session=s)

        assert self._sessions(b"[]", call) == [True]

    def test_author_papers(self) -> None:
        def call(s: FetchSession) -> object:
            return author_papers("1", session=s)

        assert self._sessions(b'{"data": [], "next": null}', call) == [True]


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
