#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script; CLI output is its product.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync \
  python3 -m wesearch.scripts.compare_extractors "$@"
Compare HTML-to-text converters on real pages: how much they cut, what they lose.

``fetch_web`` renders a page for a language model, so a converter is good when it
is small AND keeps the words. Those pull in opposite directions, and one number
hides which one a change traded away -- so every converter is scored twice:

  compression: output characters over the RAW page. Lower is cheaper. The raw
    page is the denominator because that is what a caller pays to send, and
    because it is independent of every candidate -- scoring against the visible
    text made the text dump its own contestant, unbeatable by construction.
  recall: share of the page's visible word 5-grams the output still contains.
    Lower means content was dropped. This is the honest cost of compression.

The reference for both is every text node in the document, uncleaned. That is
the ceiling of "the textual elements of a page" -- the thing the tool promises.

Article extractors (trafilatura's default, readability) score well on
compression and badly on recall for any page that is not article-shaped: a
dictionary entry, a Q&A thread, a profile timeline. Rule-based converters
(html2text, markdownify) invert that. Which trade is right is a judgement about
the corpus, so this script reports both and writes each output to disk for
reading -- it deliberately does not crown a winner.

Pages are cached so a rerun replays bytes instead of re-fetching: the corpus
includes bot-walled sites whose response varies by IP, hour, and challenge
state, and a benchmark whose input moves cannot compare two runs. The cache
lives under /opt/scratch/caches (never the checkout) and is safe to delete.

``--probe`` additionally asserts per-page strings that must survive: the check
that catches a converter silently truncating a page into plausible-looking
output, which no aggregate score reveals.
'''
# fmt: on

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import argparse
import hashlib
import re
import time

from trafilatura.baseline import html2txt
from trafilatura.external import try_readability

import html2text
import lxml.html

from wesearch.fetch import PolicyParams, RequestParams, Transport, fetch
from wesearch.fetch.extractor.html2text import extract_html2text
from wesearch.fetch.extractor.markdownify import extract_markdownify
from wesearch.fetch.extractor.trafilatura import extract_trafilatura


@dataclass(frozen=True, slots=True, kw_only=True)
class Page:
    """One corpus entry: what to fetch and what must survive conversion.

    Attributes:
      url: Page to fetch.
      transport: Retrieval transport; bot-walled hosts need ``"zendriver"``.
      probes: Strings visible on the rendered page. A converter that drops one
        has lost content a reader would notice, however good its scores.

    """

    url: str
    transport: Transport = "auto"
    probes: tuple[str, ...] = ()

    @property
    def slug(self) -> str:
        """Filesystem-safe cache key, unique per URL."""
        host = re.sub(r"[^a-z0-9]+", "-", self.url.split("//", 1)[-1].lower())
        return f"{host[:60].strip('-')}-{hashlib.sha256(self.url.encode()).hexdigest()[:8]}"


@dataclass(frozen=True, slots=True, kw_only=True)
class Score:
    """One converter's result on one page.

    Attributes:
      name: Converter name.
      chars: Output length in characters.
      compression: ``chars`` over RAW page length; lower is cheaper. The raw
        page is the denominator because that is what a caller pays to send, and
        because it is independent of every candidate -- a denominator one
        candidate produces makes that candidate unbeatable.
      distortion: ``1 - F1`` over word 5-grams against the page's visible text;
        lower is truer. Both halves of the F1 are load-bearing. Recall alone
        rewarded emitting the page's stylesheet: markdownify returned 516_564
        characters of CSS on one dictionary entry and scored 0.99, because
        every real n-gram was in there somewhere. Precision alone rewards
        returning one correct sentence and nothing else. Reported as ``1 - F1``
        so this column and ``compression`` both read smaller-is-better.
      missing_probes: Required strings absent from the output. Hand-picked
        CONTENT, so this is the one fidelity signal blind to boilerplate: the
        n-gram distortion above counts dropping a nav menu as loss, which
        flatters a converter that keeps chrome and punishes an extractor for
        doing its job. A converter is only trustworthy when this is empty.
      seconds: Wall-clock conversion time.

    """

    name: str
    chars: int
    compression: float
    distortion: float
    missing_probes: tuple[str, ...] = ()
    seconds: float = 0.0


# The corpus spans the shapes that break extractors, not a sample of the web:
# article (wikipedia, pep), dictionary entry (merriam-webster, britannica),
# Q&A thread (stackoverflow), profile timeline (x), forum (reddit), product
# listing (amazon), link index (pragmaticengineer). Probes are strings read off
# the rendered page, so a converter cannot pass by returning boilerplate.
CORPUS: tuple[Page, ...] = (
    Page(
        url="https://www.merriam-webster.com/dictionary/agent",
        transport="zendriver",
        probes=(
            "\u02c8\u0101-j\u0259nt",
            "one that acts or exerts power",
            "an oxidizing agent",
            "a theatrical agent",
            "crown agent",
            "a means or instrument by which a guiding intelligence",
        ),
    ),
    Page(
        url="https://www.britannica.com/dictionary/agent",
        transport="zendriver",
        probes=(
            "e\u026a\u02a4\u0259nt",
            "a person who does business for another",
            "real estate agent",
            "a person who tries",
        ),
    ),
    Page(
        url="https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        probes=("attention mechanism", "feedforward", "RMSNorm", "LayerNorm"),
    ),
    Page(
        url="https://peps.python.org/pep-0008/",
        probes=(
            "Use 4 spaces per indentation level",
            "Limit all lines to a maximum of 79 characters",
            "no attribute is really private",
        ),
    ),
    Page(
        url="https://arxiv.org/abs/1706.03762",
        probes=(
            "Attention Is All You Need",
            "dominant sequence transduction models",
            "15 pages, 5 figures",
        ),
    ),
    Page(
        url=(
            "https://stackoverflow.com/questions/11227809/"
            "why-is-processing-a-sorted-array-faster-than-processing-an-unsorted-array"
        ),
        probes=(
            "branch predictor",
            "Answers",
            "Bit shifting is",
            "The processor will get its",
        ),
    ),
    Page(
        url="https://x.com/elonmusk",
        transport="zendriver",
        probes=("Followers", "Terafab", "Joined June 2009"),
    ),
    Page(
        url="https://www.reddit.com/r/Python/top/?t=year",
        transport="zendriver",
        probes=("r/Python", "Pycon US 2025"),
    ),
    Page(
        url="https://www.amazon.com/Echo-Dot-5th-Gen-Charcoal/dp/B09B8V1LZ3",
        transport="zendriver",
        probes=("Echo Dot", "Alexa", "About this item"),
    ),
    Page(
        url="https://blog.pragmaticengineer.com/",
        probes=("The Pragmatic Engineer", "Section 174"),
    ),
    Page(
        url="https://news.ycombinator.com/item?id=1",
        probes=(
            "Y Combinator",
            "the rising star of venture capital",
            "Sandhill Road",
        ),
    ),
)


def reference_text(html: str) -> str:
    """Return every visible text node: the ceiling a converter is scored against.

    ``clean=False`` deliberately: the cleaned variant prunes cookie banners and
    footers by class name, and on arXiv that swallows the whole abstract page --
    it returns nothing at all. A denominator missing content scores every
    converter above 1.0 and hides the loss this script exists to measure.

    Compression above 1.0 is therefore expected and not a defect: a markdown
    converter emits link targets and table pipes that carry no words, so it can
    exceed the reference in characters while matching it in content.
    """
    return html2txt(html, clean=False) or ""


def converters() -> dict[str, Callable[[str], str]]:
    """Return every converter under comparison, by name.

    Names are kept short because they become paired column headers: twelve
    columns of ``trafilatura-html2txt`` render as ``tra...`` in a terminal,
    which makes the table unreadable exactly where it must be compared.
    """
    return {
        "traf": extract_trafilatura,
        "traf-txt": reference_text,
        # The shipped implementations, not re-parameterized copies: this script
        # scored markdownify with ``strip=["script", "style"]`` for several runs,
        # which keeps a stylesheet's TEXT and inflated one page to 567_708
        # characters. A benchmark that configures its own candidates measures
        # something the product never runs.
        "h2t": extract_html2text,
        "mdfy": extract_markdownify,
        "read-md": _readability_markdown,
    }


def score_page(
    page: Page, html: str, *, names: Sequence[str] = ()
) -> tuple[int, list[Score]]:
    """Score every converter on one page.

    Compression is measured against the RAW page, which is what a caller pays
    to hand a model. Distortion is measured against every visible text node,
    which is the most content any converter could return. The two denominators
    are deliberately different: scoring both against the visible text made the
    text dump its own contestant, scoring 1.00/0.00 by construction and topping
    a ranking it was supposed to define.

    Args:
      page: Corpus entry supplying the probes.
      html: Raw page source.
      names: Converter subset to run; empty runs all available.

    Returns:
      reference: Length of the page's full visible text, the distortion
        denominator.
      scores: One :class:`Score` per converter, in converter-registry order.

    """
    reference = reference_text(html)
    grams = _word_grams(reference)
    scores: list[Score] = []
    for name, convert in converters().items():
        if names and name not in names:
            continue
        start = time.perf_counter()
        output = convert(html)
        elapsed = time.perf_counter() - start
        scores.append(
            Score(
                name=name,
                chars=len(output),
                compression=len(output) / len(html) if html else 0.0,
                distortion=_gram_distortion(grams, output),
                missing_probes=tuple(p for p in page.probes if p not in output),
                seconds=elapsed,
            )
        )
    return len(reference), scores


def cached_html(page: Page, *, cache_dir: Path, refresh: bool = False) -> str:
    """Return the page source, fetching and caching it on first use.

    Raises:
      FetchError: When the page has no cache entry and cannot be fetched.

    """
    path = cache_dir / f"{page.slug}.html"
    if path.exists() and not refresh:
        return path.read_text(errors="replace")
    body, _ = fetch(
        page.url, request=RequestParams(policy=PolicyParams(transport=page.transport))
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body.decode("utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison and print one table per page.

    Returns:
      status: Non-zero when no page could be scored, so an unreachable corpus
        cannot read as a clean result.

    """
    args = _parse_args(argv)
    # Validated before any fetch: an unrecognized name used to filter down to
    # nothing, fetch the whole corpus anyway, print an empty table, and exit 0
    # -- a typo read as "every converter is perfect".
    unknown = sorted(set(args.converter) - set(converters()))
    if unknown:
        print(f"Unknown converter(s): {', '.join(unknown)}.")
        print(f"Available: {', '.join(converters())}.")
        return 2
    missing_urls = sorted(set(args.url) - {p.url for p in CORPUS})
    if missing_urls:
        print(f"URL(s) not in the corpus: {', '.join(missing_urls)}.")
        return 2
    pages = [p for p in CORPUS if not args.url or p.url in args.url]
    names = [n for n in converters() if not args.converter or n in args.converter]
    rows: list[tuple[Page, list[Score]]] = []
    for page in pages:
        try:
            html = cached_html(
                page, cache_dir=args.cache_dir, refresh=bool(args.refresh)
            )
        except Exception as error:  # noqa: BLE001 -- one unreachable page must not end the run.
            print(f"\n{page.slug}: UNAVAILABLE: {error}")
            continue
        rows.append((page, score_page(page, html, names=names)[1]))
        _write_samples(page, html=html, out_dir=args.samples, names=names)
    if rows:
        _print_table(rows, names=names, samples=args.samples)
    return 0 if rows else 1


def _html2text(html: str) -> str:
    """Convert with html2text, unwrapped: a wrapped line breaks probe matching."""
    converter = html2text.HTML2Text()
    converter.ignore_images = True
    converter.body_width = 0
    return converter.handle(html)


def _readability_markdown(html: str) -> str:
    """Extract with readability, then convert -- the common two-step agent recipe.

    Uses trafilatura's vendored readability fork so the comparison needs no
    separate install; it is the same algorithm behind Firefox Reader View.
    """
    article = try_readability(lxml.html.fromstring(html))
    return _html2text(lxml.html.tostring(article, encoding="unicode"))


def _word_grams(text: str, size: int = 5) -> frozenset[tuple[str, ...]]:
    """Return the text's word n-grams, ignoring case, markup, and whitespace.

    Markdown link targets are stripped first. Left in, a URL's words interleave
    with the prose and break every n-gram spanning the link, so a markdown
    converter that lost nothing scores as if it had: markdownify measured 0.72
    on arXiv while visibly containing the whole page.
    """
    words = [
        str(v) for v in re.findall(r"[\w']+", re.sub(r"\]\([^)]*\)", "]", text).lower())
    ]
    return frozenset(
        tuple(words[i : i + size]) for i in range(max(0, len(words) - size + 1))
    )


def _gram_distortion(reference: frozenset[tuple[str, ...]], output: str) -> float:
    """Return ``1 - F1`` of the output's n-grams against the reference's."""
    grams = _word_grams(output)
    if not reference or not grams:
        return 1.0
    shared = len(reference & grams)
    if not shared:
        return 1.0
    precision, recall = shared / len(grams), shared / len(reference)
    return 1.0 - 2 * precision * recall / (precision + recall)


def _print_table(
    rows: list[tuple[Page, list[Score]]], *, names: Sequence[str], samples: Path
) -> None:
    """Print one markdown table: page rows, compression then distortion columns.

    Both metric groups span the same converters in the same order, so a page
    reads left-to-right as "what each cost" then "what each lost". A converter
    is only good when its column is low in BOTH halves; either half alone ranks
    an extractor that returned nothing first.
    """
    header = [
        "page",
        *(f"{n} c" for n in names),
        *(f"{n} d" for n in names),
        *(f"{n} p" for n in names),
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for page, scores in rows:
        by_name = {s.name: s for s in scores}
        cells = [_page_label(page)]
        cells += [f"{by_name[n].compression:.2f}" for n in names]
        cells += [f"{by_name[n].distortion:.2f}" for n in names]
        cells += [
            f"**{len(by_name[n].missing_probes)}/{len(page.probes)}**"
            if by_name[n].missing_probes
            else f"0/{len(page.probes)}"
            for n in names
        ]
        print("| " + " | ".join(cells) + " |")
    print(
        f"\n`c` = compression: output chars over RAW page chars; lower is cheaper.\n"
        f"`d` = distortion: `1 - F1` of word 5-grams against the page's visible\n"
        f"text; lower is truer. `traf-txt` IS that text, so its `d` is 0.00 by\n"
        f"construction -- it is the yardstick, not a contestant.\n"
        f"`p` = content probes LOST over probes checked; **bold** = any loss.\n"
        f"Prefer `p` over `d` when they disagree: `d` cannot tell a dropped nav\n"
        f"menu from a dropped paragraph, and the probes are hand-picked content.\n"
        f"{len(rows)} pages, {len(names)} converters.\n"
        f"\n**These numbers do not decide anything on their own. READ THE OUTPUT.**\n"
        f"Every converter's text is in `{samples}`\n"
        f"as `<page>.<converter>.txt`. Two scoring bugs survived several runs of\n"
        f"this table and died on first reading of those files: a converter whose\n"
        f"91% of output was one CSS rule scored near-perfect fidelity, and the\n"
        f"visible-text dump ranked first against a yardstick that was itself.\n"
        f"An n-gram cannot tell a stylesheet from a paragraph. You can."
    )


def _page_label(page: Page) -> str:
    """Return the host, plus a path hint when the host alone is ambiguous."""
    host = page.url.split("//", 1)[-1].split("/", 1)[0].removeprefix("www.")
    return host.removesuffix(".com").removesuffix(".org")


def _write_samples(
    page: Page, *, html: str, out_dir: Path, names: Sequence[str]
) -> None:
    """Write each converter's output so a reader can judge the content itself."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, convert in converters().items():
        if name in names:
            (out_dir / f"{page.slug}.{name}.txt").write_text(convert(html))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/opt/scratch/caches/wesearch-extractors"),
        help="Where fetched pages are replayed from (default: %(default)s).",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Re-fetch pages, replacing the cache."
    )
    parser.add_argument(
        "--url", action="append", default=[], help="Limit to this corpus URL (repeat)."
    )
    parser.add_argument(
        "--converter",
        action="append",
        default=[],
        help="Limit to this converter (repeat).",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("/opt/scratch/artifacts/wesearch-extractors"),
        help="Where converter outputs are written for reading (default: %(default)s).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
