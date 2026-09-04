"""Tests for wesearch.search.search."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock, MagicMock, patch

import json
import os
import urllib.error

import bs4
import pytest

from wesearch.fetch import FetchSession
from wesearch.search.custom_types import (
    CodeResult,
    FileResult,
    ImageResult,
    MapResult,
    MediaResult,
    PackageResult,
    PaperResult,
    SearchError,
    SearchResult,
    TorrentResult,
    VideoResult,
    gsa_headers_for_query,
    strip_scripts,
)
from wesearch.search.duckduckgo import (
    _duckduckgo_check_captcha,
    _duckduckgo_extract_url,
    _duckduckgo_parse,
    _duckduckgo_quote_bangs,
    _duckduckgo_validate_body,
    duckduckgo,
)
from wesearch.search.search import search
from wesearch.search.searxng import _searxng_url, searxng
from wesearch.types.errors import (
    BotDetectionError,
    FetchError,
    PuzzleChallengeError,
)


def _patch_fetch(
    return_value: bytes = b"{}",
    side_effect: Any = None,
    *,
    module: str = "search",
) -> AbstractContextManager[MagicMock | AsyncMock]:
    # fetch returns (body, session); wrap the byte-valued test inputs so the
    # mock matches that shape (an exception side_effect still raises).
    kwargs: dict[str, Any] = {}
    if side_effect is not None:
        kwargs["side_effect"] = _tuple_side_effect(side_effect)
    else:
        kwargs["return_value"] = (return_value, FetchSession())
    return patch(f"wesearch.search.{module}.fetch", **kwargs)  # ty: ignore[unsound-return-statement] -- dynamic **kwargs into patch's overloads erases the return type


def _tuple_side_effect(side_effect: Any) -> Any:
    """Wrap a byte-valued side_effect so each byte result becomes (bytes, session)."""
    if isinstance(side_effect, list):
        items = cast(list[object], side_effect)
        return [
            item if isinstance(item, BaseException) else (item, FetchSession())
            for item in items
        ]
    return side_effect  # an exception class/instance: raised as-is.


@contextmanager
def _patch_searxng_fetch(
    payload: dict[str, Any],
) -> Generator[MagicMock | AsyncMock]:
    body = json.dumps(payload).encode()
    with (
        patch.dict(
            os.environ,
            {"SEARXNG_URL": "https://search.example.test/"},
        ),
        _patch_fetch(module="searxng", return_value=body) as mock,
    ):
        yield mock


class TestStripScripts:
    def test_removes_script_tags(self) -> None:
        soup = bs4.BeautifulSoup(
            "<div><p>Keep</p><script>remove()</script></div>",
            "html.parser",
        )
        strip_scripts(soup)
        assert "remove" not in soup.get_text()
        assert "Keep" in soup.get_text()


class TestSearchDispatch:
    def test_network_error_normalized(self) -> None:
        err = urllib.error.URLError(ConnectionResetError(104, "reset"))
        with (
            _patch_fetch(module="duckduckgo", side_effect=err),
            pytest.raises(SearchError, match="duckduckgo"),
        ):
            search("cats", backend="duckduckgo")

    def test_fetch_error_normalized(self) -> None:
        # INF-027: a backend's FetchError (e.g. HTTP 503) must surface as a
        # SearchError, not escape raw to the caller.
        err = FetchError("https://duckduckgo.com", 503, {}, b"unavailable")
        with (
            _patch_fetch(module="duckduckgo", side_effect=err),
            pytest.raises(SearchError, match="duckduckgo"),
        ):
            search("cats", backend="duckduckgo")

    def test_bot_detection_error_propagates_with_guidance(self) -> None:
        # A-WEB-002: a BotDetectionError is-a FetchError, so the broad
        # `except FetchError` flattened it into a generic SearchError, discarding
        # the actionable `.guidance` and specific type. It must propagate intact
        # so a caller can tell "solve captcha / rotate IP" from a plain failure.
        err = PuzzleChallengeError("DuckDuckGo returned a challenge form.")
        with (
            _patch_fetch(module="duckduckgo", side_effect=err),
            pytest.raises(BotDetectionError) as exc,
        ):
            search("cats", backend="duckduckgo")
        assert "captcha" in exc.value.guidance.lower()

    def test_transport_forwarded_to_backend(self) -> None:
        with patch("wesearch.search.search.duckduckgo", return_value=[]) as provider:
            search("cats", backend="duckduckgo", transport="stdlib")
        assert provider.call_args.kwargs["transport"] == "stdlib"

    def test_negative_num_results_rejected_duckduckgo(self) -> None:
        # O-WEB-003: negative num_results was accepted inconsistently (ddg had no
        # guard, searxng sliced items[:-1], scholar raised). Reject uniformly.
        with pytest.raises(ValueError, match="num_results"):
            search("cats", backend="duckduckgo", num_results=-1)

    def test_negative_num_results_rejected_searxng(self) -> None:
        with (
            patch.dict("os.environ", {"SEARXNG_URL": "https://searx.example"}),
            pytest.raises(ValueError, match="num_results"),
        ):
            search("cats", backend="searxng", num_results=-1)


class TestGsaHeaders:
    def test_query_selects_stable_user_agent(self) -> None:
        with patch(
            "wesearch.search.custom_types.user_agent_pool",
            return_value=("ua0", "ua1", "ua2"),
        ):
            assert gsa_headers_for_query("same") == gsa_headers_for_query("same")
            assert gsa_headers_for_query("same")["User-Agent"].endswith(" NSTNWV")


class TestSearchSearxng:
    def test_requires_env_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        # SearchError, not the RuntimeError it inherits: asserting the base
        # class passes even if the specific type regresses away.
        with pytest.raises(SearchError, match="SEARXNG_URL"):
            _searxng_url()

    def test_parses_results(self) -> None:
        payload = {
            "results": [
                {
                    "url": "https://example.com/page",
                    "title": "Example Title",
                    "content": "Snippet text.",
                },
            ],
        }
        with _patch_searxng_fetch(payload):
            results = searxng("test query")
        assert results == [
            SearchResult(
                url="https://example.com/page",
                title="Example Title",
                snippet="Snippet text.",
            ),
        ]

    def test_num_results_caps(self) -> None:
        payload = {
            "results": [
                {
                    "url": f"https://e{i}.com",
                    "title": f"T{i}",
                    "content": "s",
                }
                for i in range(5)
            ],
        }
        with _patch_searxng_fetch(payload):
            results = searxng("test", num_results=2)
        assert len(results) == 2

    def test_missing_results_key(self) -> None:
        with _patch_searxng_fetch({}):
            assert searxng("test") == []

    def test_non_dict_json_body_yields_empty(self) -> None:
        # A well-formed but non-object JSON response must not raise
        # AttributeError on ``.get`` -- it yields an empty result.
        with (
            patch.dict(os.environ, {"SEARXNG_URL": "https://search.example.test/"}),
            _patch_fetch(module="searxng", return_value=b"[]"),
        ):
            assert searxng("test") == []

    def test_non_list_results_field_yields_empty(self) -> None:
        with _patch_searxng_fetch({"results": "oops"}):
            assert searxng("test") == []

    def test_html_body_raises_search_error(self) -> None:
        """A 200 that carries HTML must not reach ``json.loads``.

        Cloudflare serves its rate-limit interstitial ("error code: 1015") as
        HTTP 200 with an HTML body, so no ``FetchError`` is raised and the
        page flowed into the JSON parser -- surfacing as a bare
        ``JSONDecodeError: line 1 column 1 (char 0)`` that names neither the
        backend nor the cause. Measured against the live instance: 12 rapid
        requests all returned 200 with ``<!DOCTYPE html>``.
        """
        with (
            patch.dict(os.environ, {"SEARXNG_URL": "https://search.example.test/"}),
            _patch_fetch(
                module="searxng",
                return_value=b"<!DOCTYPE html><html><body>Bad Gateway</body></html>",
            ),
            pytest.raises(SearchError, match="HTML page instead of JSON"),
        ):
            _ = searxng("test")

    def test_html_error_body_names_the_rate_limit(self) -> None:
        """The message must name the cause, not just the malformed shape."""
        with (
            patch.dict(os.environ, {"SEARXNG_URL": "https://search.example.test/"}),
            _patch_fetch(
                module="searxng",
                return_value=b"error code: 1015",
            ),
            pytest.raises(SearchError, match="rate-limit"),
        ):
            _ = searxng("test")

    def test_missing_env_raises_search_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        with pytest.raises(SearchError, match="SEARXNG_URL"):
            searxng("test")

    def test_missing_fields_default_to_empty(self) -> None:
        with _patch_searxng_fetch({"results": [{}]}):
            results = searxng("test")
        assert results == [
            SearchResult(url="", title="", snippet=""),
        ]

    def test_url_includes_query(self) -> None:
        with _patch_searxng_fetch({"results": []}) as mock:
            searxng("hello world")
        url = mock.call_args.args[0]
        assert "q=hello+world" in url
        assert "format=json" in url
        assert url.startswith("https://search.example.test/search?")

    def test_fetch_error_propagates(self) -> None:
        err = FetchError("https://x.com", 500, {}, b"")
        with (
            patch.dict(
                os.environ,
                {"SEARXNG_URL": "https://search.example.test/"},
            ),
            _patch_fetch(module="searxng", side_effect=err),
            pytest.raises(FetchError),
        ):
            searxng("test")

    @pytest.mark.parametrize("value", [True, False])
    def test_json_bool_is_not_a_coordinate(self, value: bool) -> None:
        # ``bool`` subclasses ``int``, so a numeric isinstance guard admits a
        # JSON ``true``; ``float_val`` then rejects it to its 0.0 DEFAULT. That
        # is exactly the "Gulf of Guinea" failure the parser's own comment says
        # it prevents -- reintroduced by the guard meant to prevent it.
        payload = {"results": [{"url": "https://m", "latitude": value}]}
        with _patch_searxng_fetch(payload):
            (result,) = searxng("q", categories="map")
        assert isinstance(result, MapResult)
        assert result.latitude is None

    @pytest.mark.parametrize("field", ["latitude", "longitude"])
    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_coordinate_is_unknown(self, field: str, literal: str) -> None:
        body = (
            f'{{"results": [{{"url": "https://m", "{field}": {literal}}}]}}'
        ).encode()
        with (
            patch.dict(os.environ, {"SEARXNG_URL": "https://search.example.test/"}),
            _patch_fetch(module="searxng", return_value=body),
        ):
            (result,) = searxng("q", categories="map")
        assert isinstance(result, MapResult)
        coordinate = result.latitude if field == "latitude" else result.longitude
        assert coordinate is None

    def test_non_finite_seed_does_not_escape_as_an_exception(self) -> None:
        # int(nan) raises ValueError and int(inf) OverflowError, neither of
        # which the dispatcher catches, so a hostile payload crashed the parse.
        body = (
            b'{"results": [{"template": "torrent.html", "url": "https://t",'
            b' "seed": NaN, "leech": Infinity}]}'
        )
        with (
            patch.dict(os.environ, {"SEARXNG_URL": "https://search.example.test/"}),
            _patch_fetch(module="searxng", return_value=body),
        ):
            (result,) = searxng("q", categories="files")
        assert isinstance(result, TorrentResult)
        assert result.seed is None
        assert result.leech is None

    def test_fractional_seed_is_not_silently_truncated(self) -> None:
        payload = {
            "results": [{"template": "torrent.html", "url": "https://t", "seed": 1.9}]
        }
        with _patch_searxng_fetch(payload):
            (result,) = searxng("q", categories="files")
        assert isinstance(result, TorrentResult)
        assert result.seed is None

    def test_absurd_citation_count_degrades_to_none(self) -> None:
        # CPython refuses int() on a >4300-digit string, so an unbounded digit
        # run raised out of a parse whose contract is "else None".
        payload = {
            "results": [
                {
                    "template": "paper.html",
                    "url": "https://p",
                    "comments": "9" * 5000 + " citations",
                }
            ]
        }
        with _patch_searxng_fetch(payload):
            (result,) = searxng("q", categories="science")
        assert isinstance(result, PaperResult)
        assert result.citations is None

    def test_retries_a_throttled_instance(self) -> None:
        # A Cloudflare-fronted instance rate-limits a burst of queries with a
        # 429 carrying Retry-After. That is transient, and fetch's retry loop
        # already honors Retry-After -- but only if given a budget to spend.
        with _patch_searxng_fetch({"results": []}) as mock:
            searxng("q")
        assert mock.call_args.kwargs["request"].retry.retries >= 1

    def test_a_caller_can_bound_the_retry_budget(self) -> None:
        # ``retries`` multiplies ``timeout_sec``: this hardcodes 1, so a caller
        # that lowered the ceiling to bound a wedged egress still pays twice
        # what it asked for. ``duckduckgo`` exposes the knob; this did not, so
        # the two backends answer the same question differently.
        with _patch_searxng_fetch({"results": []}) as mock:
            searxng("q", retries=0)
        assert mock.call_args.kwargs["request"].retry.retries == 0

    def test_default_category_is_general(self) -> None:
        with _patch_searxng_fetch({"results": []}) as mock:
            searxng("q")
        assert "categories=general" in mock.call_args.args[0]

    def test_social_media_category_url_encodes_space(self) -> None:
        with _patch_searxng_fetch({"results": []}) as mock:
            results = searxng("q", categories="social media")
        # SearXNG's tab value carries a space; it must be percent-encoded.
        assert "categories=social+media" in mock.call_args.args[0]
        assert results == []


class TestSearchSearxngScience:
    _PAPER_PAYLOAD: ClassVar[dict[str, Any]] = {
        "results": [
            {
                "template": "paper.html",
                "url": "https://doi.org/10.1/x",
                "title": "Attention Is All You Need",
                "content": "We propose the Transformer.",
                "authors": ["Vaswani", "Shazeer", 42],
                "journal": "NeurIPS",
                "doi": "10.1/x",
                "pdf_url": "https://arxiv.org/pdf/1706.03762",
                "publishedDate": "2017-06-12T00:00:00",
                "tags": ["Computer Science"],
                "comments": "100,000 citations from the year 2017 to 2024",
            }
        ]
    }

    def test_science_returns_paper_results(self) -> None:
        with _patch_searxng_fetch(self._PAPER_PAYLOAD):
            results = searxng("transformer", categories="science")
        assert results == [
            PaperResult(
                url="https://doi.org/10.1/x",
                title="Attention Is All You Need",
                snippet="We propose the Transformer.",
                authors=("Vaswani", "Shazeer"),
                journal="NeurIPS",
                doi="10.1/x",
                pdf_url="https://arxiv.org/pdf/1706.03762",
                # SearXNG emits naive ISO timestamps; fromisoformat keeps them
                # naive, so the expected value is naive too.
                published=datetime(2017, 6, 12),  # noqa: DTZ001
                tags=("Computer Science",),
                citations=100_000,
            )
        ]

    def test_paper_result_is_a_search_result(self) -> None:
        # LSB: a web-result consumer reading url/title/snippet works unchanged.
        with _patch_searxng_fetch(self._PAPER_PAYLOAD):
            (paper,) = searxng("transformer", categories="science")
        assert isinstance(paper, SearchResult)
        assert paper.url
        assert paper.title
        assert paper.snippet

    def test_science_category_sent(self) -> None:
        with _patch_searxng_fetch({"results": []}) as mock:
            searxng("q", categories="science")
        assert "categories=science" in mock.call_args.args[0]

    def test_sparse_paper_defaults_empty(self) -> None:
        with _patch_searxng_fetch({"results": [{"template": "paper.html"}]}):
            (paper,) = searxng("q", categories="science")
        assert paper == PaperResult(url="", title="", snippet="")
        assert paper.authors == ()
        assert paper.citations is None

    def test_malformed_published_date_is_none(self) -> None:
        payload = {"results": [{"publishedDate": "not-a-date"}]}
        with _patch_searxng_fetch(payload):
            (paper,) = searxng("q", categories="science")
        assert paper.published is None

    def test_unparseable_comments_yields_no_citations(self) -> None:
        payload = {"results": [{"comments": "many citations"}]}
        with _patch_searxng_fetch(payload):
            (paper,) = searxng("q", categories="science")
        assert paper.citations is None

    def test_search_science_passthrough_types_as_papers(self) -> None:
        with _patch_searxng_fetch(self._PAPER_PAYLOAD):
            results = search("q", backend="searxng", categories="science")
        assert isinstance(results[0], PaperResult)

    def test_categories_rejected_for_non_searxng_backend(self) -> None:
        # A non-default category on a non-searxng backend is rejected at
        # runtime; the type system permits the call (HTML backends share the
        # web-result overload), so this guard is the only enforcement point.
        with pytest.raises(ValueError, match="only supported by the 'searxng'"):
            search("q", backend="duckduckgo", categories="science")


class TestSearchSearxngStructuredCategories:
    def test_images(self) -> None:
        payload = {
            "results": [
                {
                    "url": "https://page",
                    "title": "Cat",
                    "content": "a cat",
                    "img_src": "https://img/cat.png",
                    "thumbnail_src": "https://img/cat_t.png",
                    "resolution": "1920x1080",
                    "img_format": "png",
                    "source": "example.com",
                    "filesize": "1MB",
                }
            ]
        }
        with _patch_searxng_fetch(payload):
            (r,) = searxng("cat", categories="images")
        assert r == ImageResult(
            url="https://page",
            title="Cat",
            snippet="a cat",
            image_url="https://img/cat.png",
            thumbnail_url="https://img/cat_t.png",
            resolution="1920x1080",
            img_format="png",
            source="example.com",
            filesize="1MB",
        )

    def test_video_extends_media(self) -> None:
        payload = {
            "results": [
                {
                    "url": "https://v",
                    "title": "Clip",
                    "content": "desc",
                    "length": "3:21",
                    "views": "1.2M",
                    "author": "Channel",
                    "iframe_src": "https://embed",
                    "thumbnail": "https://t",
                    "publishedDate": "2020-01-02T00:00:00",
                }
            ]
        }
        with _patch_searxng_fetch(payload):
            (r,) = searxng("clip", categories="videos")
        assert isinstance(r, MediaResult)  # VideoResult is-a MediaResult
        assert r.length == "3:21"
        assert r.views == "1.2M"
        assert r.author == "Channel"
        assert r.iframe_url == "https://embed"
        assert r.published == datetime(2020, 1, 2)  # noqa: DTZ001 -- naive ISO

    def test_news_is_media_result(self) -> None:
        payload = {
            "results": [
                {
                    "url": "https://n",
                    "title": "Headline",
                    "content": "story",
                    "publishedDate": "2026-06-22T00:00:00",
                    "thumbnail": "https://t",
                }
            ]
        }
        with _patch_searxng_fetch(payload):
            (r,) = searxng("news", categories="news")
        assert isinstance(r, MediaResult)
        assert not isinstance(r, VideoResult)
        assert r.published == datetime(2026, 6, 22)  # noqa: DTZ001 -- naive ISO

    def test_music_is_media_result(self) -> None:
        payload = {"results": [{"url": "https://s", "audio_src": "https://a"}]}
        with _patch_searxng_fetch(payload):
            (r,) = searxng("song", categories="music")
        assert isinstance(r, MediaResult)
        assert r.audio_url == "https://a"

    def test_map(self) -> None:
        payload = {
            "results": [
                {
                    "url": "https://m",
                    "title": "Eiffel Tower",
                    "content": "tower",
                    "latitude": 48.8584,
                    "longitude": 2.2945,
                    "address": {"road": "Champ de Mars", "country": "France", "x": 9},
                }
            ]
        }
        with _patch_searxng_fetch(payload):
            (r,) = searxng("eiffel", categories="map")
        assert isinstance(r, MapResult)
        assert r.latitude == 48.8584
        assert r.longitude == 2.2945
        # Non-string address values are dropped, not coerced.
        assert dict(r.address) == {"road": "Champ de Mars", "country": "France"}

    def test_map_missing_coords_are_none(self) -> None:
        with _patch_searxng_fetch({"results": [{"url": "https://m"}]}):
            (r,) = searxng("x", categories="map")
        assert isinstance(r, MapResult)
        assert r.latitude is None
        assert r.longitude is None
        assert dict(r.address) == {}

    def test_it_dispatches_by_template(self) -> None:
        payload = {
            "results": [
                {
                    "template": "packages.html",
                    "url": "https://pkg",
                    "title": "numpy",
                    "content": "arrays",
                    "package_name": "numpy",
                    "version": "2.0",
                    "license_name": "BSD",
                    "tags": ["python", 7],
                },
                {
                    "template": "code.html",
                    "url": "https://code",
                    "title": "main.py",
                    "content": "def main()",
                    "repository": "org/repo",
                    "filename": "main.py",
                    "code_language": "python",
                },
                {"template": "default.html", "url": "https://w", "title": "wiki"},
            ]
        }
        with _patch_searxng_fetch(payload):
            pkg, code, web = searxng("numpy", categories="it")
        assert isinstance(pkg, PackageResult)
        assert pkg.package_name == "numpy"
        assert pkg.version == "2.0"
        assert pkg.tags == ("python",)
        assert isinstance(code, CodeResult)
        assert code.repository == "org/repo"
        assert code.code_language == "python"
        assert type(web) is SearchResult

    def test_files_dispatches_by_template(self) -> None:
        payload = {
            "results": [
                {
                    "template": "torrent.html",
                    "url": "https://t",
                    "title": "ISO",
                    "content": "",
                    "magnetlink": "magnet:?xt=1",
                    "seed": 10,
                    "leech": 2,
                    "filesize": "700MB",
                },
                {
                    "template": "file.html",
                    "url": "https://f",
                    "title": "doc",
                    "abstract": "a document",
                    "filename": "doc.pdf",
                    "size": "1MB",
                    "mimetype": "application/pdf",
                },
            ]
        }
        with _patch_searxng_fetch(payload):
            torrent, file = searxng("iso", categories="files")
        assert isinstance(torrent, TorrentResult)
        assert torrent.magnet_url == "magnet:?xt=1"
        assert torrent.seed == 10
        assert torrent.leech == 2
        assert isinstance(file, FileResult)
        assert file.filename == "doc.pdf"
        # file.html abstract feeds snippet.
        assert file.snippet == "a document"

    def test_search_passthrough_images(self) -> None:
        payload = {"results": [{"url": "https://p", "img_src": "https://i"}]}
        with _patch_searxng_fetch(payload):
            results = search("q", backend="searxng", categories="images")
        assert isinstance(results[0], ImageResult)


_SINGLE_DDG = """
<html><body>
<div id="links">
  <div class="result results_links results_links_deep web-result">
    <h2 class="result__title">
      <a class="result__a" href="https://example.com/page">Example Title</a>
    </h2>
    <a class="result__snippet">This is the snippet text.</a>
  </div>
</div>
</body></html>
"""

_TWO_DDG = """
<html><body>
<div id="links">
  <div class="result web-result">
    <h2><a class="result__a" href="https://first.com">First</a></h2>
    <a class="result__snippet">First snippet.</a>
  </div>
  <div class="result web-result">
    <h2><a class="result__a" href="https://second.com">Second</a></h2>
    <a class="result__snippet">Second snippet.</a>
  </div>
</div>
</body></html>
"""

_NO_RESULTS_DDG = "<html><body><div>No results.</div></body></html>"

_SCRIPT_DDG = """
<html><body>
<div id="links">
  <div class="result web-result">
    <h2><a class="result__a" href="https://example.com">Title</a></h2>
    <a class="result__snippet">Snippet.<script>noise()</script></a>
  </div>
</div>
</body></html>
"""

_MISSING_LINK_DDG = """
<html><body>
<div id="links"><div class="result web-result"><span>No link here</span></div></div>
</body></html>
"""


class TestQuoteBangsDdg:
    def test_quotes_leading_bang_token(self) -> None:
        assert _duckduckgo_quote_bangs("!w python") == "'!w' python"

    def test_quotes_embedded_bang_token(self) -> None:
        assert _duckduckgo_quote_bangs("python !gh repo") == "python '!gh' repo"

    def test_preserves_normal_query(self) -> None:
        assert _duckduckgo_quote_bangs("python web search") == "python web search"


class TestCheckCaptchaDdg:
    def test_detects_challenge_form(self) -> None:
        with pytest.raises(PuzzleChallengeError):
            _duckduckgo_check_captcha(
                '<html><body><form id="challenge-form"></form></body></html>',
            )

    def test_normal_page(self) -> None:
        _duckduckgo_check_captcha("<html><body>ok</body></html>")


class TestExtractUrlDdg:
    def test_direct_url(self) -> None:
        assert (
            _duckduckgo_extract_url("https://example.com/page")
            == "https://example.com/page"
        )

    def test_protocol_relative_url(self) -> None:
        assert (
            _duckduckgo_extract_url("//example.com/page") == "https://example.com/page"
        )

    def test_rejects_non_http_url(self) -> None:
        assert _duckduckgo_extract_url("httpx2://example.com/page") is None

    def test_wrapped_url(self) -> None:
        assert (
            _duckduckgo_extract_url(
                "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%2520b",
            )
            == "https://example.com/a%20b"
        )

    def test_protocol_relative_wrapped_url(self) -> None:
        assert (
            _duckduckgo_extract_url(
                "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage",
            )
            == "https://example.com/page"
        )

    def test_empty_url(self) -> None:
        assert _duckduckgo_extract_url("") is None


class TestParseDdg:
    def test_single(self) -> None:
        results = _duckduckgo_parse(_SINGLE_DDG, 10)
        assert len(results) == 1
        assert results[0].url == "https://example.com/page"
        assert results[0].title == "Example Title"
        assert "snippet text" in results[0].snippet

    def test_two(self) -> None:
        results = _duckduckgo_parse(_TWO_DDG, 10)
        assert [r.title for r in results] == ["First", "Second"]

    def test_max_results_caps(self) -> None:
        assert len(_duckduckgo_parse(_TWO_DDG, 1)) == 1

    def test_zero_max_returns_empty(self) -> None:
        # O-WEB-009: max_results=0 must yield [] (append-before-cap bug).
        assert _duckduckgo_parse(_TWO_DDG, 0) == []

    def test_no_results(self) -> None:
        assert _duckduckgo_parse(_NO_RESULTS_DDG, 10) == []

    def test_scripts_stripped(self) -> None:
        results = _duckduckgo_parse(_SCRIPT_DDG, 10)
        assert len(results) == 1
        assert "noise" not in results[0].snippet

    def test_missing_link_skipped(self) -> None:
        assert _duckduckgo_parse(_MISSING_LINK_DDG, 10) == []

    def test_long_snippet_not_truncated(self) -> None:
        long = "x" * 500
        html = f"""
        <html><body>
        <div id="links"><div class="result web-result">
          <h2><a class="result__a" href="https://example.com">T</a></h2>
          <a class="result__snippet">{long}</a>
        </div></div>
        </body></html>
        """
        results = _duckduckgo_parse(html, 10)
        assert len(results) == 1
        assert len(results[0].snippet) >= 500

    def test_no_snippet_ok(self) -> None:
        html = """
        <html><body>
        <div id="links"><div class="result web-result">
          <h2><a class="result__a" href="https://example.com">Title</a></h2>
        </div></div>
        </body></html>
        """
        results = _duckduckgo_parse(html, 10)
        assert len(results) == 1
        assert results[0].snippet == ""

    def test_wrapped_result_url(self) -> None:
        html = """
        <html><body>
        <div id="links"><div class="result web-result">
          <h2><a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">Wrapped</a></h2>
        </div></div>
        </body></html>
        """
        results = _duckduckgo_parse(html, 10)
        assert results[0].url == "https://example.com/page"

    def test_ignores_ad_results(self) -> None:
        html = """
        <html><body>
        <div id="links">
          <div class="result result--ad result--ad--small">
            <h2><a class="result__a" href="https://ad.example.com">Ad</a></h2>
          </div>
          <div class="result web-result">
            <h2><a class="result__a" href="https://example.com">Organic</a></h2>
          </div>
        </div>
        </body></html>
        """
        results = _duckduckgo_parse(html, 10)
        assert [r.title for r in results] == ["Organic"]

    def test_requires_links_container(self) -> None:
        html = """
        <html><body>
          <div class="result web-result">
            <h2><a class="result__a" href="https://example.com">Loose</a></h2>
          </div>
        </body></html>
        """
        assert _duckduckgo_parse(html, 10) == []


class TestDuckduckgoChallengeReachesFallback:
    def test_challenge_is_a_body_validator_not_a_post_hoc_check(self) -> None:
        """The challenge must be detected INSIDE fetch, not after it returns.

        DuckDuckGo serves its puzzle with HTTP 200. Checked on the returned
        body, the error is raised past fetch's own ``except BotDetectionError``
        -- the hook that learns the domain and retries through Zendriver -- so
        an automatic-transport caller just failed instead of escalating.
        """
        challenge = b'<html><body><form id="challenge-form"></form></body></html>'
        with pytest.raises(PuzzleChallengeError):
            _duckduckgo_validate_body(challenge)

    def test_a_normal_body_passes_the_validator(self) -> None:
        _duckduckgo_validate_body(b"<html><body><div>results</div></body></html>")


class TestSearchDuckduckgo:
    def test_delegates_to_parse(self) -> None:
        with _patch_fetch(module="duckduckgo", return_value=_SINGLE_DDG.encode()):
            results = duckduckgo("test query")
        assert len(results) == 1
        assert results[0].title == "Example Title"

    def test_uses_normal_browser_request_contract(self) -> None:
        with (
            patch(
                "wesearch.search.duckduckgo.user_agent_pool",
                return_value=("ddg-test-ua",),
            ),
            _patch_fetch(
                module="duckduckgo", return_value=_NO_RESULTS_DDG.encode()
            ) as mock,
        ):
            duckduckgo("test")
        # The query rides in the URL, not a POST body: a POSTed query is dropped
        # and DDG serves its empty homepage. GET with q= returns real results.
        req = mock.call_args.kwargs["request"]
        assert req.content.method == "GET"
        assert req.content.data is None
        url = mock.call_args.args[0]
        assert url.startswith("https://html.duckduckgo.com/html/?")
        assert "q=test" in url
        assert "kl=wt-wt" in url
        assert req.content.headers == {
            "User-Agent": "ddg-test-ua NSTNWV",
            "Accept": "*/*",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Accept-Language": "all,all-ALL;q=0.7",
            "Referer": "https://html.duckduckgo.com/html/",
        }
        assert not req.content.cookies
        # INF-025: the GSA mobile UA must be sent verbatim, without fetch's
        # default desktop Chrome sec-ch-ua headers.
        assert req.content.raw_headers is True
        assert req.retry.retries == 2

    def test_user_agent_is_process_stable_across_queries(self) -> None:
        # DDG's vqd anti-bot token is keyed to (query, UA); a UA that shifts
        # between requests is read as a bot. Unlike the per-query Google UA, the
        # DDG UA must be the SAME for every query in the process.
        with _patch_fetch(
            module="duckduckgo", return_value=_NO_RESULTS_DDG.encode()
        ) as mock:
            duckduckgo("alpha")
            duckduckgo("beta")
        ua_a = mock.call_args_list[0].kwargs["request"].content.headers["User-Agent"]
        ua_b = mock.call_args_list[1].kwargs["request"].content.headers["User-Agent"]
        assert ua_a == ua_b
        assert ua_a.endswith("NSTNWV")

    def test_quotes_bangs_before_request(self) -> None:
        with _patch_fetch(
            module="duckduckgo",
            return_value=_NO_RESULTS_DDG.encode(),
        ) as mock:
            duckduckgo("!w python")
        # Bang tokens are quoted, then percent-encoded into the query string.
        assert "q=%27%21w%27+python" in mock.call_args.args[0]

    def test_rejects_too_long_query(self) -> None:
        # INF-026: an over-length query must raise, not silently return [].
        with (
            _patch_fetch(module="duckduckgo") as mock,
            pytest.raises(SearchError, match="exceeds"),
        ):
            duckduckgo("x" * 500)
        mock.assert_not_called()

    def test_fetch_error_propagates(self) -> None:
        err = FetchError("https://x.com", 500, {}, b"")
        with (
            _patch_fetch(module="duckduckgo", side_effect=err),
            pytest.raises(FetchError),
        ):
            duckduckgo("test")


class TestLiveQueryAvailabilitySkips:
    """``_query_once`` must classify a wedged backend as availability.

    Lives HERE, not beside the helper it exercises: EVERY test in
    ``search_integration_test.py`` carries ``integration``, which the default
    tier deselects via ``addopts`` in ``pyproject.toml``. Measured -- a bare
    passing ``test_probe`` written into that file reports ``1 deselected``, and
    the identical file without the ``_integration`` suffix reports ``1
    passed``. A unit test placed there would therefore never run in the tier
    that guards this logic.
    """

    @staticmethod
    def _run(
        fetch: Callable[[float], list[str]], *, backend: str, timeout_sec: float
    ) -> list[str]:
        """Call the live helper, imported at use so collection stays cheap."""
        from wesearch.search.search_integration_test import (  # noqa: PLC0415 -- see class docstring
            _query_once,
        )

        return _query_once(fetch, backend=backend, timeout_sec=timeout_sec)

    def test_skips_a_backend_that_never_answered(self) -> None:
        """``search`` classifies this; a direct backend call bypassed it.

        ``TimeoutError`` is in the tuple ``search`` converts to ``SearchError``,
        but the live cases call each backend DIRECTLY and so never reach that
        facade. Observed in CI: Google served its enablejs shell to curl, the
        browser fallback never returned, and the run failed on a bare
        ``TimeoutError`` while the sibling query passed in 6.91s.
        """

        def never_answers(timeout_sec: float) -> list[str]:
            del timeout_sec
            raise TimeoutError

        # A synthetic backend name: the helper keys its unreachable marker by
        # it, so borrowing "google" would let a marker left by a live run decide
        # this verdict.
        with pytest.raises(pytest.skip.Exception, match="did not answer"):
            self._run(never_answers, backend="wedged-probe", timeout_sec=0.1)

    def test_still_fails_a_malformed_response(self) -> None:
        """The skip clauses must not swallow a genuine parser regression.

        Guards the widening above: every clause added there trades a failure
        for a skip, and one too broad retires the live tests silently.
        """

        def raises_parser_fault(timeout_sec: float) -> list[str]:
            del timeout_sec
            raise ValueError("markup changed")

        with pytest.raises(ValueError, match="markup changed"):
            self._run(raises_parser_fault, backend="wedged-probe", timeout_sec=0.1)


class TestHeadersArg:
    def test_searxng_custom_headers(self) -> None:
        with _patch_searxng_fetch({"results": []}) as mock:
            searxng("q", headers={"User-Agent": "custom/1.0"})
        assert mock.call_args.kwargs["request"].content.headers == {
            "User-Agent": "custom/1.0",
        }

    def test_ddg_custom_headers_merge_with_defaults(self) -> None:
        with _patch_fetch(
            module="duckduckgo",
            return_value=_NO_RESULTS_DDG.encode(),
        ) as mock:
            duckduckgo("q", headers={"User-Agent": "x"})
        assert mock.call_args.kwargs["request"].content.headers == {
            "User-Agent": "x",
            "Accept": "*/*",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Accept-Language": "all,all-ALL;q=0.7",
            "Referer": "https://html.duckduckgo.com/html/",
        }


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
