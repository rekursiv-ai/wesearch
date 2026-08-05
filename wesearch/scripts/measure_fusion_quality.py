#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")/../../.." run --frozen \
  python3 -m wesearch.scripts.measure_fusion_quality "$@"
Report live fused-search duplication and limit adherence.

Two properties the unit tests cannot observe, because both depend on what the
backends actually return: how many surviving records name the same paper, and
whether the returned count honors ``limit``. Run before and after a change to
``fuse`` or ``_fused`` to see the effect on real result sets.

Live network calls to Semantic Scholar and OpenAlex.
'''
# fmt: on

from __future__ import annotations

from collections import defaultdict

import time

from wesearch.paper.custom_types import PaperRecord
from wesearch.paper.errors import PaperError
from wesearch.paper.fuse import _normalize_title
from wesearch.paper.search import search


def _residual_duplicates(records: list[PaperRecord]) -> int:
    """Records naming a paper another record already named, by title."""
    by_title: defaultdict[str, int] = defaultdict(int)
    for rec in records:
        by_title[_normalize_title(rec.title)] += 1
    return sum(count - 1 for count in by_title.values() if count > 1)


def main(
    queries: tuple[str, ...] = (
        "attention",
        "bayesian inference",
        "editorial introduction",
        "graph neural network",
        "discussion",
        "reciprocal rank fusion",
        "test-time training",
        "sparse autoencoder",
    ),
    *,
    limit: int = 40,
) -> int:
    """Print per-query duplication and limit adherence for fused search.

    Args:
      queries: Search queries to sample. The generic titles ("discussion",
        "editorial introduction") are deliberate: they are where a
        title-similarity rule collapses genuinely distinct papers, so a
        regression there shows up as a LOSS of records, not a gain.
      limit: Hits requested per query; the returned count must not exceed it.

    Returns:
      status: Process exit status.

    """
    total = duplicates = overruns = 0
    for query in queries:
        result = None
        for _attempt in range(4):
            try:
                result = search(query, limit=limit)
                break
            except PaperError:
                time.sleep(6.0)
        if result is None:
            print(f"{query!r:26s} SKIPPED (backends unavailable)")  # noqa: T201 -- CLI probe.
            continue
        residual = _residual_duplicates(result.records)
        over = max(0, len(result.records) - limit)
        total += len(result.records)
        duplicates += residual
        overruns += over
        print(  # noqa: T201 -- CLI probe output.
            f"{query!r:26s} records={len(result.records):3d} "
            f"limit={limit} overrun={over:2d} "
            f"same-title-residual={residual:2d} total={result.total}"
        )
        time.sleep(1.0)
    print(  # noqa: T201 -- CLI probe output.
        f"\nTOTAL records={total} residual-duplicates={duplicates} "
        f"limit-overruns={overruns}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
