"""Tests for wesearch.paper.fuse (reciprocal-rank fusion + dedup)."""

from __future__ import annotations

from dataclasses import fields

from wesearch.paper.custom_types import PaperRecord
from wesearch.paper.fuse import fuse


def _rec(title: str, *, doi: str | None = None, source: str = "s2") -> PaperRecord:
    return PaperRecord(title=title, doi=doi, sources=(source,))


class TestMergeCompleteness:
    def test_merge_preserves_every_field_on_dedup(self) -> None:
        # A-WEB-005: _merge enumerated fields by hand and forgot several optional
        # ones (e.g. is_influential), zeroing them when two records dedup. Any
        # field populated only on the first record must survive the merge --
        # assert per-field so a NEWLY added field can't be silently dropped.
        first = PaperRecord(
            title="X",
            doi="10.1/a",
            is_influential=True,
            sources=("s2",),
        )
        second = PaperRecord(title="X", doi="10.1/a", sources=("openalex",))
        (merged,) = fuse([first], [second])
        for f in fields(PaperRecord):
            if f.name == "sources":
                continue  # sources are unioned, asserted elsewhere
            assert getattr(merged, f.name) == getattr(first, f.name), (
                f"_merge dropped field {f.name!r}"
            )


class TestFuse:
    def test_agreement_outranks_lone_top(self) -> None:
        # A paper both backends rank (#2 S2, #1 OpenAlex) must beat S2's lone #1.
        s2 = [_rec("solo", doi="10.1/solo"), _rec("shared", doi="10.1/shared")]
        oa = [_rec("shared", doi="10.1/shared", source="openalex")]
        out = fuse(s2, oa)
        assert out[0].doi == "10.1/shared"
        assert set(out[0].sources) == {"s2", "openalex"}

    def test_dedup_by_doi_merges_sources(self) -> None:
        s2 = [_rec("t", doi="10.1/x")]
        oa = [_rec("t", doi="10.1/x", source="openalex")]
        out = fuse(s2, oa)
        assert len(out) == 1
        assert set(out[0].sources) == {"s2", "openalex"}

    def test_dedup_by_title_when_no_doi(self) -> None:
        s2 = [_rec("Deep Learning!")]
        oa = [_rec("deep  learning", source="openalex")]
        out = fuse(s2, oa)
        assert len(out) == 1

    def test_dedup_by_arxiv_id_across_differing_dois(self) -> None:
        # A preprint and its published version are ONE paper carrying two DOIs
        # (10.48550/arxiv.* and the publisher's). Keying on DOI alone splits
        # them; the shared arXiv id is the identity that joins them. Measured
        # live: 10 such pairs across 6 queries, the whole of the real fused
        # duplication.
        s2 = [
            PaperRecord(
                title="RRF",
                doi="10.1145/3596512",
                arxiv_id="2210.11934",
                sources=("s2",),
            )
        ]
        oa = [
            PaperRecord(
                title="RRF",
                doi="10.48550/arxiv.2210.11934",
                arxiv_id="2210.11934",
                sources=("openalex",),
            )
        ]
        out = fuse(s2, oa)
        assert len(out) == 1
        assert set(out[0].sources) == {"s2", "openalex"}
        # The publisher DOI wins: S2 is merged first and its values take priority.
        assert out[0].doi == "10.1145/3596512"

    def test_same_backend_duplicate_does_not_double_score(self) -> None:
        # A backend that returns ONE paper twice must not out-rank a distinct
        # paper it ranked far higher. Summing both occurrences' reciprocal-rank
        # contributions lets a duplicate pair at ranks 11-12 beat the backend's
        # own #1, so a contribution is per BACKEND per paper, not per row.
        # Measured live: 21 of 22 intra-backend collisions are OpenAlex's own
        # preprint/published twins, so this is the common case, not a corner.
        oa = [PaperRecord(title="top", doi="10.1/top", sources=("openalex",))]
        oa += [
            PaperRecord(title=f"filler{i}", doi=f"10.1/f{i}", sources=("openalex",))
            for i in range(9)
        ]
        oa += [
            PaperRecord(title="dup", doi="10.1/dup", sources=("openalex",)),
            PaperRecord(title="dup", doi="10.1/dup", sources=("openalex",)),
        ]
        out = fuse([], oa)
        assert out[0].title == "top"

    def test_openalex_only_still_ranked(self) -> None:
        # A throttled S2 (empty) degrades to OpenAlex-ranked results, not nothing.
        out = fuse([], [_rec("a", source="openalex"), _rec("b", source="openalex")])
        assert [r.title for r in out] == ["a", "b"]

    def test_s2_wins_equal_rank_tie(self) -> None:
        # Same-rank single-backend papers break in S2's favor (higher weight).
        out = fuse(
            [_rec("s2top", doi="10.1/s")],
            [_rec("oatop", doi="10.1/o", source="openalex")],
        )
        assert out[0].doi == "10.1/s"


class TestAMissingTitleIsNotIdentity:
    def test_two_untitled_papers_do_not_collapse(self) -> None:
        # A backend reporting no title says nothing about the paper, so two
        # such records are not the same paper. Keying on the absence unions
        # every id-less untitled record into ONE component and destroys all
        # but one -- the same failure the server-side title dedup was removed
        # for.
        out = fuse(
            [
                PaperRecord(title="", year=1990, sources=("s2",)),
                PaperRecord(title="", year=2020, sources=("s2",)),
            ],
            [],
        )
        assert len(out) == 2

    def test_whitespace_is_not_a_title(self) -> None:
        out = fuse(
            [PaperRecord(title="   ", sources=("s2",))],
            [PaperRecord(title="\t", sources=("openalex",))],
        )
        assert len(out) == 2

    def test_a_real_shared_title_still_joins(self) -> None:
        # The refusal is scoped to an ABSENT title: a genuine one is still the
        # last-resort identity for id-less records.
        out = fuse(
            [_rec("Deep Learning!")], [_rec("deep  learning", source="openalex")]
        )
        assert len(out) == 1


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
