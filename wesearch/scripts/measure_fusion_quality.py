#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script; CLI output is its product.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync \
  python3 -m wesearch.scripts.measure_fusion_quality "$@"
Report live fused-search duplication and limit adherence.

Two properties the unit tests cannot observe, because both depend on what the
backends actually return: how many surviving records name the same paper, and
whether the returned count honors ``limit``. Run before and after a change to
the IDENTITY rule in ``fuse`` -- both metrics are invariant to the rank
weights, which only reorder a set identity has already fixed.

Live network calls to Semantic Scholar and OpenAlex.
'''
# fmt: on

from __future__ import annotations

import time

from wesearch.paper.custom_types import PaperRecord
from wesearch.paper.errors import PaperError
from wesearch.paper.fuse import normalize_title
from wesearch.paper.search import SearchResult, search


def _residual_duplicates(records: list[PaperRecord]) -> int:
    """Records naming a paper another record already named.

    Identity, not title: two records sharing a normalized title are the same
    paper only when they also share an identifier, since distinct papers really
    do share titles (``Discussion``). Counting every same-title pair scored
    correct behavior as a defect -- it is the very collapsing this measurement
    exists to show the absence of.
    """
    seen: set[str] = set()
    duplicates = 0
    for rec in records:
        keys: set[str] = set()
        if rec.doi:
            keys.add(f"doi:{rec.doi.lower()}")
        if rec.arxiv_id:
            keys.add(f"arxiv:{rec.arxiv_id.lower()}")
        if not keys:
            keys = {f"title:{normalize_title(rec.title)}"}
        if keys & seen:
            duplicates += 1
        seen |= keys
    return duplicates


def _sampled(query: str, *, limit: int, attempts: int = 4) -> SearchResult | None:
    """Search ``query``, or return ``None`` when no usable sample came back.

    A degraded result is not a sample of FUSED quality: with one backend lost
    there is nothing to fuse, so counting it would report the duplication rate
    of a single backend as fusion's.
    """
    for _attempt in range(attempts):
        try:
            result = search(query, limit=limit)
        except PaperError:
            time.sleep(6.0)
            continue
        return result if result.complete or result.records else None
    return None


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
      status: Process exit status; non-zero when no query yielded a sample,
        so an unavailable backend cannot read as a clean measurement.

    """
    total = duplicates = overruns = sampled = 0
    for query in queries:
        result = _sampled(query, limit=limit)
        if result is None:
            print(f"{query!r:26s} SKIPPED (backends unavailable)")
            continue
        sampled += 1
        residual = _residual_duplicates(result.records)
        over = max(0, len(result.records) - limit)
        total += len(result.records)
        duplicates += residual
        overruns += over
        print(
            f"{query!r:26s} records={len(result.records):3d} "
            f"limit={limit} overrun={over:2d} "
            f"duplicate-identities={residual:2d} total={result.total}"
        )
        time.sleep(1.0)
    print(
        f"\nTOTAL queries={sampled}/{len(queries)} records={total} "
        f"residual-duplicates={duplicates} limit-overruns={overruns}"
    )
    return 0 if sampled else 1


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
