"""Tests for the host-wide search result cache."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import os
import time

from wesearch.lib.userdirs import cache_dir
from wesearch.search import cache
from wesearch.search.cache import cached_search
from wesearch.search.types import PaperResult, SearchResult


if TYPE_CHECKING:
    import pytest


_ROWS = [
    PaperResult(
        url="https://a.example/",
        title="a",
        snippet="",
        authors=("x",),
        published=datetime(2020, 1, 1, tzinfo=UTC),
        citations=3,
    ),
    SearchResult(url="https://b.example/", title="b", snippet=""),
]


def test_a_hit_skips_the_backend_and_keeps_result_subclasses() -> None:
    calls = [0]

    def compute() -> list[SearchResult]:
        calls[0] += 1
        return list(_ROWS)

    first = cached_search({"q": "x"}, compute)
    second = cached_search({"q": "x"}, compute)

    assert calls == [1]
    assert second == first == _ROWS
    assert type(second[0]) is PaperResult


def test_distinct_keys_do_not_share_an_entry() -> None:
    calls = [0]

    def compute() -> list[SearchResult]:
        calls[0] += 1
        return list(_ROWS)

    cached_search({"q": "x", "backend": "a"}, compute)
    cached_search({"q": "x", "backend": "b"}, compute)

    assert calls == [2]


def test_an_empty_answer_is_not_cached() -> None:
    """An empty page is often a soft block; pinning it would starve every agent."""
    calls = [0]

    def compute() -> list[SearchResult]:
        calls[0] += 1
        return []

    cached_search({"q": "x"}, compute)
    cached_search({"q": "x"}, compute)

    assert calls == [2]


def test_a_stale_entry_is_recomputed() -> None:
    calls = [0]

    def compute() -> list[SearchResult]:
        calls[0] += 1
        return list(_ROWS)

    cached_search({"q": "x"}, compute)
    base = cache_dir() / "rekursiv-ai" / "wesearch" / "search"
    for entry in base.glob("*.json"):
        old = time.time() - 7200
        os.utime(entry, (old, old))
    cached_search({"q": "x"}, compute, ttl_sec=3600)

    assert calls == [2]


def test_a_zero_ttl_bypasses_read_and_write() -> None:
    calls = [0]

    def compute() -> list[SearchResult]:
        calls[0] += 1
        return list(_ROWS)

    cached_search({"q": "x"}, compute)
    cached_search({"q": "x"}, compute, ttl_sec=0)
    cached_search({"q": "y"}, compute, ttl_sec=0)

    assert calls == [3]
    stored = list((cache_dir() / "rekursiv-ai" / "wesearch" / "search").glob("*.json"))
    assert len(stored) == 1


def test_a_write_prunes_expired_entries_and_idle_locks() -> None:
    base = cache_dir() / "rekursiv-ai" / "wesearch" / "search"
    base.mkdir(parents=True)
    old = time.time() - 2 * 86_400
    for name in ("stale.json", "stale.lock", "idle.lock"):
        (base / name).write_text("x")
        os.utime(base / name, (old, old))
    (base / "recent.lock").write_text("x")

    cached_search({"q": "x"}, lambda: list(_ROWS))

    survivors = sorted(path.name for path in base.iterdir())
    assert "stale.json" not in survivors
    assert "stale.lock" not in survivors
    assert "idle.lock" not in survivors
    assert "recent.lock" in survivors


def test_prune_tolerates_an_entry_removed_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = cache_dir() / "rekursiv-ai" / "wesearch" / "search"
    base.mkdir(parents=True)
    gone = base / "gone.json"
    gone.write_text("x")
    real = Path.stat
    probed: list[Path] = []

    # Only the entry vanishes: 3.12's ``glob`` stats the directory itself first.
    def vanished(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self.name != gone.name:
            return real(self, follow_symlinks=follow_symlinks)
        probed.append(self)
        raise FileNotFoundError

    monkeypatch.setattr(Path, "stat", vanished)
    cache._prune(base, ttl_sec=1.0)
    assert probed == [gone]


def test_a_corrupt_entry_is_recomputed() -> None:
    calls = [0]

    def compute() -> list[SearchResult]:
        calls[0] += 1
        return list(_ROWS)

    cached_search({"q": "x"}, compute)
    for entry in (cache_dir() / "rekursiv-ai" / "wesearch" / "search").glob("*.json"):
        entry.write_text("{not json")
    assert cached_search({"q": "x"}, compute) == _ROWS
    assert calls == [2]


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
