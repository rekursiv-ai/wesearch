"""Tests for wesearch.paper.paginate (the generic cursor walker)."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from treekle import MutablePlainTree, from_plain

import pytest

from wesearch.paper.errors import BackendError
from wesearch.paper.paginate import Cursor, _cap, paginate


def _rows(body: dict[str, MutablePlainTree]) -> list[dict[str, MutablePlainTree]]:
    """Read the ``data`` row list from a fixture page body."""
    return cast(list[dict[str, MutablePlainTree]], body.get("data") or [])


def _offset_cursor(
    pages: list[list[dict[str, MutablePlainTree]]],
    *,
    page_size_max: int = 2,
    fetch: MagicMock | None = None,
) -> Cursor:
    """Return a 0-based offset cursor over pre-baked ``pages`` (short page = last)."""

    def do_fetch(offset: int, size: int) -> dict[str, MutablePlainTree]:
        del size
        idx = offset  # `pages` are indexed by page number for simplicity.
        rows = pages[idx] if idx < len(pages) else []
        body: dict[str, MutablePlainTree] = {"data": [*rows]}
        return body

    def advance(
        body: dict[str, MutablePlainTree],
        position: int,
        size: int,
    ) -> int | None:
        rows = body.get("data") or []
        assert isinstance(rows, list)
        return position + 1 if len(rows) >= size else None

    return Cursor(
        fetch=fetch or do_fetch,
        rows=_rows,
        advance=advance,
        page_size_max=page_size_max,
    )


class TestPaginate:
    def test_cap_none_preserves_list_identity(self) -> None:
        entries: list[dict[str, MutablePlainTree]] = [{"n": 1}]
        assert _cap(entries, None) is entries

    def test_single_page_under_limit_is_complete(self) -> None:
        cursor = _offset_cursor([[{"n": 1}, {"n": 2}]], page_size_max=3)
        page = paginate(cursor, limit=None)
        assert page.entries == [{"n": 1}, {"n": 2}]
        assert page.complete

    def test_limit_clamps_page_size_never_exceeds_max(self) -> None:
        sizes: list[int] = []

        def do_fetch(offset: int, size: int) -> dict[str, MutablePlainTree]:
            sizes.append(size)
            rows: list[MutablePlainTree] = [{"n": offset}] * size  # Always full.
            body: dict[str, MutablePlainTree] = {"data": rows}
            return body

        cursor = _offset_cursor(
            [],
            page_size_max=2,
            fetch=MagicMock(side_effect=do_fetch),
        )
        page = paginate(cursor, limit=5)
        assert sizes == [2, 2, 2]
        assert page.entries == [{"n": 0}, {"n": 0}, {"n": 1}, {"n": 1}, {"n": 2}]
        assert not page.complete

    def test_walks_multiple_pages_to_limit_incomplete(self) -> None:
        pages: list[list[dict[str, MutablePlainTree]]] = [
            [{"n": 1}, {"n": 2}],
            [{"n": 3}, {"n": 4}],
        ]
        cursor = _offset_cursor(pages, page_size_max=2)
        page = paginate(cursor, limit=3)
        assert page.entries == [{"n": 1}, {"n": 2}, {"n": 3}]
        assert not page.complete

    def test_full_page_exactly_at_limit_is_incomplete(self) -> None:
        pages: list[list[dict[str, MutablePlainTree]]] = [
            [{"n": 1}, {"n": 2}],
            [{"n": 3}],
        ]
        cursor = _offset_cursor(pages, page_size_max=2)
        page = paginate(cursor, limit=2)
        assert page.entries == [{"n": 1}, {"n": 2}]
        assert not page.complete

    def test_keep_filter_does_not_understate_completeness(self) -> None:
        pages: list[list[dict[str, MutablePlainTree]]] = [
            [{"n": 1}, {"n": 2}],
            [{"n": 3}],
        ]
        cursor = _offset_cursor(pages, page_size_max=2)
        page = paginate(
            cursor,
            limit=None,
            keep=lambda r: from_plain(r["n"], int) % 2 == 0,
        )
        assert page.entries == [{"n": 2}]
        assert not page.complete

    def test_zero_limit_returns_empty_page(self) -> None:
        cursor = _offset_cursor([[{"n": 1}]], page_size_max=2)
        page = paginate(cursor, limit=0)
        assert page.entries == []
        assert page.complete is False

    def test_depth_ceiling_caps_overlong_backend_page(self) -> None:
        def do_fetch(offset: int, size: int) -> dict[str, MutablePlainTree]:
            if offset == 0:
                return {"data": [{"n": i} for i in range(size + 1)]}
            raise BackendError("too deep", status=400)

        cursor = Cursor(
            fetch=MagicMock(side_effect=do_fetch),
            rows=_rows,
            advance=lambda _b, pos, _s: pos + 1,
            page_size_max=2,
            is_depth_ceiling=lambda e: e.status == 400,
        )
        page = paginate(cursor, limit=4)
        assert page.entries == [{"n": 0}, {"n": 1}, {"n": 2}]
        assert page.complete is False

    def test_depth_ceiling_stops_incomplete_with_results(self) -> None:
        def do_fetch(offset: int, size: int) -> dict[str, MutablePlainTree]:
            if offset == 0:
                return {"data": [{"n": i} for i in range(size)]}
            raise BackendError("too deep", status=400)

        cursor = Cursor(
            fetch=MagicMock(side_effect=do_fetch),
            rows=_rows,
            advance=lambda _b, pos, _s: pos + 1,  # Always claims more.
            page_size_max=2,
            is_depth_ceiling=lambda e: e.status == 400,
        )
        page = paginate(cursor, limit=3)
        assert page.entries == [{"n": 0}, {"n": 1}]
        assert page.complete is False  # Ceiling hit -> more may exist.

    def test_depth_ceiling_without_results_reraises(self) -> None:
        def do_fetch(offset: int, size: int) -> dict[str, MutablePlainTree]:
            del offset, size
            raise BackendError("bad", status=400)

        cursor = Cursor(
            fetch=MagicMock(side_effect=do_fetch),
            rows=_rows,
            advance=lambda _b, pos, _s: pos + 1,
            page_size_max=2,
            is_depth_ceiling=lambda e: e.status == 400,
        )
        with pytest.raises(BackendError):
            paginate(cursor, limit=3)

    def test_non_depth_ceiling_error_reraises(self) -> None:
        cursor = Cursor(
            fetch=MagicMock(side_effect=BackendError("boom", status=500)),
            rows=_rows,
            advance=lambda _b, _pos, _s: None,
            page_size_max=2,
        )
        with pytest.raises(BackendError):
            paginate(cursor, limit=None)

    def test_non_advancing_cursor_caps_overlong_page(self) -> None:
        cursor = Cursor(
            fetch=MagicMock(return_value={"data": [{"n": 0}, {"n": 1}, {"n": 2}]}),
            rows=_rows,
            advance=lambda _b, _pos, _s: 0,
            page_size_max=2,
        )
        page = paginate(cursor, limit=4)
        assert page.entries == [{"n": 0}, {"n": 1}, {"n": 2}]
        assert page.complete is False

    def test_non_advancing_cursor_terminates(self) -> None:
        # Advance returns a position <= current -> stop, do not loop forever.
        cursor = Cursor(
            fetch=MagicMock(return_value={"data": [{"n": 0}, {"n": 1}]}),
            rows=_rows,
            advance=lambda _b, _pos, _s: 0,  # Never advances.
            page_size_max=2,
        )
        page = paginate(cursor, limit=3)
        assert page.entries == [{"n": 0}, {"n": 1}]
        assert page.complete is False

    def test_start_position_respected(self) -> None:
        seen: list[int] = []

        def do_fetch(position: int, size: int) -> dict[str, MutablePlainTree]:
            del size
            seen.append(position)
            return {"data": []}

        cursor = Cursor(
            fetch=MagicMock(side_effect=do_fetch),
            rows=_rows,
            advance=lambda _b, _pos, _s: None,
            page_size_max=2,
            start=1,  # 1-based page APIs (OpenAlex)
        )
        paginate(cursor, limit=None)
        assert seen == [1]

    def test_negative_limit_rejected(self) -> None:
        # O-WEB-006: a negative limit flows into _page_size -> min(-1, max) = -1,
        # sending a negative page size on the wire and slicing entries[:-1].
        # Reject it at the boundary instead.
        cursor = _offset_cursor([[]], page_size_max=2)
        with pytest.raises(ValueError, match="limit"):
            paginate(cursor, limit=-1)


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
