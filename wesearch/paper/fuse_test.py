"""Tests for wesearch.paper.fuse (reciprocal-rank fusion + dedup)."""

from __future__ import annotations

from dataclasses import fields

from wesearch.paper.custom_types import PaperRecord
from wesearch.paper.fuse import _find, fuse, normalize_title


def _rec(title: str, *, doi: str = "", source: str = "s2") -> PaperRecord:
    return PaperRecord(title=title, doi=doi, sources=(source,))


def test_normalize_title_exactly_lowercases_punctuation_and_whitespace() -> None:
    assert normalize_title(" A\tB--C! ") == "a b c"


def test_find_compresses_a_multi_level_path() -> None:
    parent = {0: 0, 1: 0, 2: 1}
    assert _find(parent, 2) == 0
    assert parent[2] == 0


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
                continue  # `sources` are unioned, asserted elsewhere.
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

    def test_identifiers_match_case_insensitively(self) -> None:
        out = fuse(
            [PaperRecord(title="T", doi="10.1/X", arxiv_id="2106.ABC")],
            [
                PaperRecord(
                    title="T",
                    doi="10.1/x",
                    arxiv_id="2106.abc",
                    sources=("openalex",),
                ),
            ],
        )
        assert len(out) == 1

    def test_arxiv_identity_without_doi(self) -> None:
        out = fuse(
            [PaperRecord(title="one", arxiv_id="2106.00001")],
            [PaperRecord(title="other", arxiv_id="2106.00002", sources=("openalex",))],
        )
        assert len(out) == 2

    def test_transitive_identity_group(self) -> None:
        out = fuse(
            [
                PaperRecord(title="a", doi="10.1/a"),
                PaperRecord(title="b", doi="10.1/b", arxiv_id="2106.00001"),
                PaperRecord(title="c", doi="10.1/a", arxiv_id="2106.00001"),
            ],
            [],
        )
        assert len(out) == 1

    def test_fusion_consumes_every_ranked_record(self) -> None:
        out = fuse(
            [_rec("a", doi="10.1/a"), _rec("b", doi="10.1/b"), _rec("c", doi="10.1/c")],
            [
                _rec("d", doi="10.1/d", source="openalex"),
                _rec("e", doi="10.1/e", source="openalex"),
            ],
        )
        assert [record.title for record in out] == ["a", "b", "c", "d", "e"]

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
            ),
        ]
        oa = [
            PaperRecord(
                title="RRF",
                doi="10.48550/arxiv.2210.11934",
                arxiv_id="2210.11934",
                sources=("openalex",),
            ),
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

    def test_rank_offset_changes_a_boundary_ordering(self) -> None:
        s2_hits = [_rec(f"s{i}", doi=f"10.1/s{i}") for i in range(1, 6)]
        s2_hits.append(_rec("s6", doi="10.1/s6"))
        oa_hits = [_rec("o1", doi="10.1/o1", source="openalex")]
        out = fuse(s2_hits, oa_hits)
        assert [record.title for record in out[-2:]] == ["o1", "s6"]


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
            [_rec("Deep Learning!")],
            [_rec("deep  learning", source="openalex")],
        )
        assert len(out) == 1


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
