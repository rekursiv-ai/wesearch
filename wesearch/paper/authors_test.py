"""Tests for wesearch.paper.authors (author search / metadata / papers)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from wesearch.paper.authors import author_metadata, author_papers, search_authors
from wesearch.paper.custom_types import AuthorRecord
from wesearch.paper.paginate import Page
from wesearch.paper.providers import s2


if TYPE_CHECKING:
    from collections.abc import Callable

    from wesearch.lib.codec import MutablePlainTree


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
        assert batch.call_args.kwargs == {"endpoint": "author"}

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

    def test_author_papers_year_filter(self) -> None:
        entries: list[dict[str, MutablePlainTree]] = [
            {"title": "new", "year": 2024},
            {"title": "boundary", "year": 2020},
            {"title": "old", "year": 2000},
        ]

        def fake(
            author_id: str,
            *,
            limit: int | None,
            keep: Callable[[dict[str, MutablePlainTree]], bool],
        ) -> Page:
            del author_id, limit
            return Page(entries=[e for e in entries if keep(e)], complete=True)

        with patch.object(s2, "author_papers", side_effect=fake) as papers:
            listing = author_papers("1", limit=7, year_from=2020)
        assert [r.title for r in listing.records] == ["new", "boundary"]
        assert papers.call_args.args == ("1",)
        assert papers.call_args.kwargs["limit"] == 7
        assert papers.call_args.kwargs["keep"] is not None

    def test_author_papers_preserves_completeness(self) -> None:
        def fake(
            author_id: str,
            *,
            limit: int | None,
            keep: Callable[[dict[str, MutablePlainTree]], bool],
        ) -> Page:
            del author_id, limit, keep
            return Page(entries=[], complete=False)

        with patch.object(s2, "author_papers", side_effect=fake):
            listing = author_papers("1", limit=2)
        assert listing.complete is False

    def test_author_papers_excludes_a_bool_year(self) -> None:
        # ``bool`` subclasses ``int``, so ``isinstance(True, int)`` admitted a
        # JSON ``true`` as a publication year and then compared it as 1.
        entries: list[dict[str, MutablePlainTree]] = [
            {"title": "real", "year": 2024},
            {"title": "bogus", "year": True},
        ]

        def fake(
            author_id: str,
            *,
            limit: int | None,
            keep: Callable[[dict[str, MutablePlainTree]], bool],
        ) -> Page:
            del author_id, limit
            return Page(entries=[e for e in entries if keep(e)], complete=True)

        with patch.object(s2, "author_papers", side_effect=fake):
            listing = author_papers("1", limit=None, year_from=1, year_to=9999)
        assert [r.title for r in listing.records] == ["real"]

    def test_author_papers_no_filter_keeps_all(self) -> None:
        entries: list[dict[str, MutablePlainTree]] = [{"title": "a"}, {"title": "b"}]

        def fake(
            author_id: str,
            *,
            limit: int | None,
            keep: Callable[[dict[str, MutablePlainTree]], bool],
        ) -> Page:
            del author_id, limit
            # No bounds -> keep predicate returns True for every entry.
            return Page(entries=[e for e in entries if keep(e)], complete=True)

        with patch.object(s2, "author_papers", side_effect=fake):
            listing = author_papers("1", limit=None)
        assert len(listing.records) == 2

    def test_author_papers_year_to_and_undated(self) -> None:
        # year_to upper bound + an undated (non-int year) work excluded.
        entries: list[dict[str, MutablePlainTree]] = [
            {"title": "in", "year": 2018},
            {"title": "boundary", "year": 2020},
            {"title": "toolate", "year": 2024},
            {"title": "undated"},
        ]

        def fake(
            author_id: str,
            *,
            limit: int | None,
            keep: Callable[[dict[str, MutablePlainTree]], bool],
        ) -> Page:
            del author_id, limit
            return Page(entries=[e for e in entries if keep(e)], complete=True)

        with patch.object(s2, "author_papers", side_effect=fake):
            listing = author_papers("1", limit=None, year_to=2020)
        assert [r.title for r in listing.records] == ["in", "boundary"]


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
