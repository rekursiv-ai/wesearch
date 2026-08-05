"""Reciprocal-rank fusion of results from two backends.

Merges Semantic Scholar and OpenAlex hit lists into one ranked list: each
backend contributes ``weight / (offset + rank)`` to a paper's score, summed
across backends, so cross-backend agreement outranks either backend's lone top
hit.

One paper carries several identifiers -- a publisher DOI, arXiv's DataCite DOI,
an arXiv id -- and the two backends rarely report the same subset, so identity
is a SET of keys, not one key. Records are grouped by connected components over
those keys (union-find): two records merge when they share ANY identifier, even
transitively through a third record. A record with no identifier at all falls
back to its normalized title.
"""

from __future__ import annotations

import re

from wesearch.paper.custom_types import PaperRecord


__all__ = ["fuse", "normalize_title"]

_WORD_PUNCT_RE = re.compile(r"[^\w\s]+")
_WS_RE = re.compile(r"\s+")


def fuse(s2_hits: list[PaperRecord], oa_hits: list[PaperRecord]) -> list[PaperRecord]:
    """Reciprocal-rank-fuse S2 and OpenAlex hits into one ranked list.

    A paper both backends rank well floats above either backend's lone top hit;
    an OpenAlex-only paper still scores by its single rank, so a throttled S2
    degrades to OpenAlex-ranked results rather than nothing.

    Args:
      s2_hits: Semantic Scholar results in rank order (best first).
      oa_hits: OpenAlex results in rank order (best first).

    Returns:
      fused: Papers ordered by descending fused score, one entry per paper.

    References:
      https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
        Cormack, Clarke, Büttcher. "Reciprocal Rank Fusion Outperforms
        Condorcet and Individual Rank Learning Methods." SIGIR 2009.

    """
    # RRF rank offset: ``weight / (offset + rank)`` adds ``offset`` phantom
    # slots before the real ranks (score half-life around ``rank == offset``).
    # The canonical 60 over-flattens with only two backends -- all of S2's top
    # ~27 would outrank an OpenAlex-only #1, burying the cross-pollinated hits
    # fusion exists to surface. At 10, a strong single-backend hit interleaves
    # into the other's top while respecting S2's lead.
    offset = 10.0
    # S2's relevance ranking is more precise than OpenAlex's broad text match,
    # so an S2 rank counts for more; an OpenAlex-only paper still scores.
    # Not kwargs: both metrics measure_fusion_quality reports are invariant to
    # these -- weights only reorder a set the identity rule already fixed.
    weights = ((s2_hits, "s2", 1.0), (oa_hits, "openalex", 0.7))

    groups = _group_by_identity([rec for hits, _label, _w in weights for rec in hits])
    merged: dict[int, PaperRecord] = {}
    score: dict[int, float] = {}
    # A backend that returns one paper twice (OpenAlex indexes a preprint and
    # its published version as separate works) must contribute ONE reciprocal
    # rank, not the sum of both rows -- otherwise a self-collision at ranks
    # 11-12 outranks that backend's own #1. Keep each backend's best rank only.
    best_rank: dict[tuple[int, str], int] = {}
    index = 0
    for hits, label, weight in weights:
        for rank, rec in enumerate(hits, start=1):
            root = groups[index]
            index += 1
            merged[root] = merged[root].merge(rec) if root in merged else rec
            if best_rank.setdefault((root, label), rank) == rank:
                score[root] = score.get(root, 0.0) + weight / (offset + rank)
    # Stable sort by descending score over insertion-ordered roots; the S2 loop
    # runs first so a coincidental score tie keeps S2's paper first.
    return [
        merged[root] for root in sorted(merged, key=lambda r: score[r], reverse=True)
    ]


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace in a title.

    Args:
      title: A backend-reported title.

    Returns:
      normalized: The comparison form used as a last-resort identity key.

    """
    return _WS_RE.sub(" ", _WORD_PUNCT_RE.sub(" ", title.lower())).strip()


def _identity_keys(rec: PaperRecord) -> list[str]:
    """Every identifier naming this paper, or its title when it has none.

    A DOI and an arXiv id are separate namespaces for the SAME paper, so both
    are emitted: a record carrying only the publisher DOI and one carrying only
    arXiv's DataCite DOI join through whichever key they share. Title is a last
    resort -- two distinct papers can share one (``Discussion``, ``Editorial
    introduction``), so it is used only when no identifier exists at all.

    A record with no identifier AND no title gets NO key: it names nothing the
    fusion can recognize, so it merges with nothing rather than with every
    other nameless record.
    """
    keys: list[str] = []
    if rec.doi:
        keys.append(f"doi:{rec.doi.lower()}")
    if rec.arxiv_id:
        keys.append(f"arxiv:{rec.arxiv_id.lower()}")
    if keys:
        return keys
    title = normalize_title(rec.title)
    return [f"title:{title}"] if title else []


def _group_by_identity(records: list[PaperRecord]) -> list[int]:
    """Return each record's component root under union-find over its keys.

    Args:
      records: Records to group, in the order their ranks will be scored.

    Returns:
      roots: One component root per input record, positionally aligned.

    """
    parent: dict[int, int] = {}
    owner: dict[str, int] = {}
    for index, rec in enumerate(records):
        parent[index] = index
        for key in _identity_keys(rec):
            other = owner.setdefault(key, index)
            _union(parent, index, other)
    return [_find(parent, index) for index in range(len(records))]


def _find(parent: dict[int, int], node: int) -> int:
    """Return ``node``'s component root, compressing the path behind it."""
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != root:
        parent[node], node = root, parent[node]
    return root


def _union(parent: dict[int, int], left: int, right: int) -> None:
    """Merge two components, keeping the earlier-seen record as the root."""
    left_root, right_root = _find(parent, left), _find(parent, right)
    if left_root != right_root:
        parent[max(left_root, right_root)] = min(left_root, right_root)
