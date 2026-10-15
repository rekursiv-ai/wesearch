"""A host-wide disk cache for search results.

Agents on one host ask overlapping questions; without a shared cache each pays
the full backend cost, and the egress IP's budget, for an answer another agent
already holds. Entries live under the user cache directory and are shared by
every process; a per-key ``flock`` makes concurrent identical queries wait for
the first one instead of all going to the backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import fcntl
import hashlib
import json
import logging
import os
import time

from treekle import ReadError, from_plain, json_loads, to_plain

from wesearch.lib.userdirs import cache_dir
from wesearch.search.custom_types import SearchResult


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path


__all__ = ["cached_search"]

logger = logging.getLogger(__name__)


def cached_search(
    key: Mapping[str, object],
    compute: Callable[[], Sequence[SearchResult]],
    *,
    ttl_sec: float = 3600.0,
) -> Sequence[SearchResult]:
    """Return the cached results for ``key``, computing and storing them on a miss.

    Empty results are not stored: an empty page is as often a parser gap or a
    soft block as a true "nothing found", and caching it would pin that answer
    for every agent. Entries round-trip through ``treekle``, so a category's
    result subclass (a paper's DOI, an image's URL) comes back as that subclass.

    Args:
      key: Everything that determines the answer (query, backend, count, ...).
      compute: Produces the results on a miss.
      ttl_sec: Seconds an entry stays valid; ``0`` bypasses the cache entirely.

    Returns:
      results: The cached or freshly computed results.

    """
    if ttl_sec <= 0:
        return compute()
    base = cache_dir() / "rekursiv-ai" / "wesearch" / "search"
    base.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps(to_plain(dict(key)), sort_keys=True).encode(),
    ).hexdigest()
    path = base / f"{digest}.json"
    lock = os.open(
        base / f"{digest}.lock",
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
        0o600,
    )
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hit = _read(path, ttl_sec=ttl_sec)
        if hit is not None:
            return hit
        results = compute()
        if results:
            _write(path, results)
            _prune(base, ttl_sec=ttl_sec)
        return results
    finally:
        os.close(lock)


def _read(path: Path, *, ttl_sec: float) -> list[SearchResult] | None:
    """Return a fresh entry, or ``None`` when absent, stale, or unreadable."""
    try:
        if time.time() - path.stat().st_mtime > ttl_sec:
            return None
        return from_plain(
            json_loads(path.read_bytes()),
            list[SearchResult],
            allow_imports=True,
        )
    except FileNotFoundError:
        return None
    except (OSError, ReadError, ValueError):
        logger.warning("Ignoring unreadable search cache entry %s.", path)
        return None


def _write(path: Path, results: Sequence[SearchResult]) -> None:
    """Atomically replace ``path`` with ``results``."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(to_plain(list(results))))
    tmp.replace(path)


# A lock file is kept a day past its entry: unlinking one another process still holds
# lets a third lock a fresh inode and run the same query concurrently, which costs one
# duplicate request -- harmless only once nobody has asked that query for a day.
def _prune(base: Path, *, ttl_sec: float, lock_ttl_sec: float = 86_400.0) -> None:
    """Delete expired entries and idle locks so the cache does not grow unbounded."""
    now = time.time()
    for pattern, ttl in (("*.json", ttl_sec), ("*.lock", max(ttl_sec, lock_ttl_sec))):
        for entry in base.glob(pattern):
            try:
                if entry.stat().st_mtime < now - ttl:
                    entry.unlink()
            except FileNotFoundError:
                continue
