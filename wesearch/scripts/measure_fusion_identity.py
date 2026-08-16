#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script; CLI output is its product.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync \
  python3 -m wesearch.scripts.measure_fusion_identity "$@"
Measure which identifier namespaces actually join records across backends.

Answers one question before ``PaperRecord`` grows a field: does adding an
identifier namespace merge record pairs that DOI + arXiv id do not already
merge? A namespace that buys zero incremental joins is dead weight.

Live network calls to Semantic Scholar and OpenAlex; S2 throttles at one
request/second, so a 429 is retried with a fixed backoff.
'''
# fmt: on

from __future__ import annotations

from collections.abc import Callable

import re
import time

from wesearch.lib.custom_json import dict_val, list_val, str_val
from wesearch.paper.custom_types import PaperRecord
from wesearch.paper.errors import PaperError
from wesearch.paper.providers import openalex, s2
from wesearch.paper.search import search


_ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.(.+?)(?:v\d+)?$", re.IGNORECASE)


def _doi_keys(rec: PaperRecord) -> list[str]:
    """Today's identity: DOI only (fuse falls back to title when absent)."""
    return [f"doi:{rec.doi.lower()}"] if rec.doi else []


def _arxiv_keys(rec: PaperRecord) -> list[str]:
    """DOI plus arXiv id, including one recovered from a 10.48550 DOI."""
    keys = _doi_keys(rec)
    arxiv = rec.arxiv_id
    if arxiv is None:
        match = _ARXIV_DOI_RE.match(rec.doi or "")
        arxiv = match.group(1) if match else None
    if arxiv:
        keys.append(f"arxiv:{arxiv.lower()}")
    return keys


def _raw_records(
    query: str, *, attempts: int = 4
) -> tuple[list[PaperRecord], list[PaperRecord]]:
    """Fetch one page from each backend, retrying S2's shared-gate throttle.

    Raises:
      PaperError: When S2 stays throttled. A backend that never answered is an
        unavailable sample, not zero joins -- reporting it as data would
        publish a measurement nobody took.

    """
    for _attempt in range(attempts):
        try:
            s2_hits = search(query, source="s2", limit=40).records
            break
        except PaperError:
            time.sleep(6.0)
    else:
        raise PaperError(f"Semantic Scholar unavailable for {query!r}")
    oa_hits, _oa_total, _oa_complete = openalex.search(
        query, limit=40, year_from=None, year_to=None, open_access_only=False
    )
    return s2_hits, oa_hits


def _cross_backend_joins(
    s2_hits: list[PaperRecord],
    oa_hits: list[PaperRecord],
    key_fn: Callable[[PaperRecord], list[str]],
) -> set[str]:
    """Keys present on BOTH backends -- the pairs this namespace would merge."""
    left = {key for rec in s2_hits for key in key_fn(rec)}
    right = {key for rec in oa_hits for key in key_fn(rec)}
    return left & right


def main(
    queries: tuple[str, ...] = (
        "attention",
        "bayesian inference",
        "graph neural network",
        "reciprocal rank fusion",
        "test-time training",
        "sparse autoencoder",
    ),
) -> int:
    """Report incremental cross-backend joins per identifier namespace.

    Args:
      queries: Search queries to sample. Broad and narrow terms both matter --
        a broad query surfaces near-duplicate titles, a narrow one surfaces
        preprint/published pairs.

    Returns:
      status: Process exit status; non-zero when no query yielded a sample, so
        an unavailable backend cannot read as a clean measurement.

    """
    totals = {"doi": 0, "arxiv": 0, "mag_extra": 0}
    sampled = 0
    for query in queries:
        try:
            s2_hits, oa_hits = _raw_records(query)
            s2_mag = _raw_mag_s2(query)
        except PaperError as e:
            print(f"{query!r:26s} SKIPPED ({e})")
            continue
        sampled += 1
        by_doi = _cross_backend_joins(s2_hits, oa_hits, _doi_keys)
        by_arxiv = _cross_backend_joins(s2_hits, oa_hits, _arxiv_keys)
        # MAG lives on the raw payloads, not on PaperRecord, so read it there.
        oa_mag = _raw_mag_openalex(query)
        mag_pairs = set(s2_mag) & set(oa_mag)
        # A MAG join is incremental only when the pair shares NO key the
        # existing identity already uses. Comparing DOIs alone counted a pair
        # that DOI+arXiv already merges (publisher DOI vs DataCite DOI, one
        # arXiv id) as new, overstating what a MAG field would buy.
        extra = {
            mag
            for mag in mag_pairs
            if not _identity_of(s2_mag[mag]) & _identity_of(oa_mag[mag])
        }
        totals["doi"] += len(by_doi)
        totals["arxiv"] += len(by_arxiv)
        totals["mag_extra"] += len(extra)
        print(
            f"{query!r:26s} doi={len(by_doi):3d} "
            f"doi+arxiv={len(by_arxiv):3d} mag_pairs={len(mag_pairs):3d} "
            f"mag_beyond_doi={len(extra):2d}"
        )
        time.sleep(1.0)
    print(
        f"\nTOTAL queries={sampled}/{len(queries)} doi={totals['doi']} "
        f"doi+arxiv={totals['arxiv']} "
        f"(+{totals['arxiv'] - totals['doi']} from arXiv) "
        f"mag_beyond_doi={totals['mag_extra']}"
    )
    return 0 if sampled else 1


def _identity_of(doi: str) -> set[str]:
    """Keys a bare DOI string contributes under the shipped identity rule."""
    if not doi:
        return set()
    keys = {f"doi:{doi.lower()}"}
    match = _ARXIV_DOI_RE.match(doi)
    if match:
        keys.add(f"arxiv:{match.group(1).lower()}")
    return keys


def _raw_mag_s2(query: str, *, attempts: int = 4) -> dict[str, str]:
    """MAG id -> DOI (``""`` when absent) from a raw S2 search page.

    Raises:
      PaperError: When S2 stays throttled, for the reason in
        :func:`_raw_records`.

    """
    for _attempt in range(attempts):
        try:
            body = s2.get(
                "/paper/search",
                {"query": query, "fields": s2.S2_PAPER_FIELDS_STR, "limit": 40},
            )
            break
        except PaperError:
            time.sleep(6.0)
    else:
        raise PaperError(f"Semantic Scholar unavailable for {query!r}")
    out: dict[str, str] = {}
    for row in list_val(body.get("data")):
        ids = dict_val(dict_val(row).get("externalIds"))
        mag = str_val(ids.get("MAG"))
        if mag:
            out[mag] = str_val(ids.get("DOI"))
    return out


def _raw_mag_openalex(query: str) -> dict[str, str]:
    """MAG id -> DOI (``""`` when absent) from a raw OpenAlex works page."""
    out: dict[str, str] = {}
    body = openalex._get(  # noqa: SLF001 -- probe reads the raw payload.
        "/works",
        {
            "filter": f"title_and_abstract.search:{query}",
            "select": "id,doi,ids,title",
            "per-page": 40,
            "page": 1,
        },
    )
    for work in list_val(body.get("results")):
        work_obj = dict_val(work)
        mag = str_val(dict_val(work_obj.get("ids")).get("mag"))
        if mag:
            doi = str_val(work_obj.get("doi"))
            out[mag.rsplit("/", 1)[-1]] = doi.rsplit("doi.org/", 1)[-1]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
