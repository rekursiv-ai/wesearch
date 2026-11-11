"""Tests for the host-wide search result cache."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import logging
import os
import stat
import time

import pytest

from wesearch.lib.userdirs import cache_dir
from wesearch.search import cache
from wesearch.search.cache import cached_search
from wesearch.search.types import PaperResult, SearchResult


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


@pytest.mark.parametrize("ttl_sec", [3600.0, 0.5])
def test_a_hit_skips_the_backend_and_keeps_result_subclasses(
    ttl_sec: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    calls = [0]

    def compute() -> list[SearchResult]:
        calls[0] += 1
        return list(_ROWS)

    first = cached_search({"q": "x", "backend": "a"}, compute, ttl_sec=ttl_sec)
    second = cached_search({"backend": "a", "q": "x"}, compute, ttl_sec=ttl_sec)

    assert calls == [1]
    assert second == first == _ROWS
    assert type(second[0]) is PaperResult
    locks = (cache_dir() / "rekursiv-ai" / "wesearch" / "search").glob("*.lock")
    assert [stat.S_IMODE(lock.stat().st_mode) for lock in locks] == [0o600]


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


@pytest.mark.parametrize(
    ("ttl_sec", "age", "expected_calls"),
    [(2.0, 2.0, 1), (2.0, 2.5, 2), (None, 3600.0, 1), (None, 3600.5, 2)],
)
def test_an_entry_is_recomputed_only_after_its_ttl(
    monkeypatch: pytest.MonkeyPatch,
    ttl_sec: float | None,
    age: float,
    expected_calls: int,
) -> None:
    calls = [0]
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)

    def compute() -> list[SearchResult]:
        calls[0] += 1
        return list(_ROWS)

    options = {} if ttl_sec is None else {"ttl_sec": ttl_sec}
    cached_search({"q": "x"}, compute, **options)
    base = cache_dir() / "rekursiv-ai" / "wesearch" / "search"
    for entry in base.glob("*.json"):
        os.utime(entry, (now - age, now - age))
    assert cached_search({"q": "x"}, compute, **options) == _ROWS
    assert calls == [expected_calls]


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


@pytest.mark.parametrize(
    ("lock_ttl_sec", "lock_age"),
    [(4.0, 4.0), (None, 86_400.0)],
)
def test_prune_keeps_entries_until_their_ttl_is_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_ttl_sec: float | None,
    lock_age: float,
) -> None:
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    for name, age in (("answer.json", 2.0), ("answer.lock", lock_age)):
        entry = tmp_path / name
        entry.write_text("x")
        os.utime(entry, (now - age, now - age))
    options = {} if lock_ttl_sec is None else {"lock_ttl_sec": lock_ttl_sec}
    cache._prune(tmp_path, ttl_sec=2.0, **options)
    assert {entry.name for entry in tmp_path.iterdir()} == {
        "answer.json",
        "answer.lock",
    }
    now += 0.5
    cache._prune(tmp_path, ttl_sec=2.0, **options)
    assert list(tmp_path.iterdir()) == []


def test_prune_tolerates_an_entry_removed_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = cache_dir() / "rekursiv-ai" / "wesearch" / "search"
    base.mkdir(parents=True)
    entries = {base / "gone.json", base / "stale.json"}
    for entry in entries:
        entry.write_text("x")
        old = time.time() - 2
        os.utime(entry, (old, old))
    real = Path.stat
    probed: list[Path] = []

    # Only the entry vanishes: 3.12's ``glob`` stats the directory itself first.
    def vanished(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self not in entries:
            return real(self, follow_symlinks=follow_symlinks)
        probed.append(self)
        if len(probed) == 1:
            self.unlink()
            raise FileNotFoundError
        return real(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", vanished)
    cache._prune(base, ttl_sec=1.0)
    assert set(probed) == entries
    assert list(base.iterdir()) == []


def test_a_corrupt_entry_is_recomputed(caplog: pytest.LogCaptureFixture) -> None:
    """Preserve recovery, warning level, cause and path, not case or decoration."""
    calls = [0]

    def compute() -> list[SearchResult]:
        calls[0] += 1
        return list(_ROWS)

    cached_search({"q": "x"}, compute)
    (entry,) = (cache_dir() / "rekursiv-ai" / "wesearch" / "search").glob("*.json")
    entry.write_text("{not json")
    assert cached_search({"q": "x"}, compute) == _ROWS
    assert calls == [2]
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert str(entry) in record.getMessage()
    assert "unreadable search cache entry" in record.getMessage().casefold()


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
