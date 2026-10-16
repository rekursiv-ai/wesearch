"""DuckDuckGo search backend: scrapes the HTML endpoint, no key required."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse

import logging

from wesearch.fetch import (
    ContentParams,
    ObserveParams,
    PolicyParams,
    RequestParams,
    RetryParams,
    Transport,
    fetch,
)
from wesearch.search.types import (
    SearchError,
    SearchResult,
    clean_text,
    strip_scripts,
)
from wesearch.types.errors import PuzzleChallengeError


if TYPE_CHECKING:
    import bs4

    from wesearch.fetch import FetchSession
else:
    from wrapt import lazy_import

    bs4 = lazy_import("bs4")  # 39ms.


logger = logging.getLogger(__name__)


def duckduckgo(
    query: str,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
    *,
    max_query_chars: int = 499,
    timeout_sec: float = 10.0,
    connect_timeout_sec: float = 3.0,
    retries: int = 1,
    transport: Transport = "auto",
    session: FetchSession | None = None,
) -> list[SearchResult]:
    """Scrape DuckDuckGo's HTML-only endpoint.

    Args:
      query: Search query string.
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to fetch.
      max_query_chars: Reject a longer query; DuckDuckGo silently drops one.
      timeout_sec: Ceiling per attempt, browser leg included.
      connect_timeout_sec: Ceiling on the handshake alone, so an unreachable
        host fails fast rather than after ``timeout_sec``.
      retries: Retry attempts for a transient failure. An unreachable
        endpoint costs ``(retries + 1) * timeout_sec`` before raising.
      transport: Retrieval transport; ``"auto"`` applies domain routing.
      session: Connections to reuse; ``None`` uses the process's shared session.

    Returns:
      results: Up to ``num_results`` hits, in page order.

    Raises:
      ValueError: If ``num_results`` is negative.
      SearchError: If ``query`` exceeds ``max_query_chars``.
      PuzzleChallengeError: If DuckDuckGo serves its bot challenge.

    """
    if num_results < 0:
        raise ValueError(f"'num_results' must be >= 0, got {num_results}.")
    if num_results == 0:
        return []
    if len(query) > max_query_chars:
        raise SearchError(
            f"DuckDuckGo query exceeds {max_query_chars} characters (got {len(query)}).",
        )
    # GET, not POST: the HTML endpoint drops a POSTed query and serves its empty
    # homepage. ``kl=wt-wt`` requests region-neutral results.
    params = urlencode({"q": _duckduckgo_quote_bangs(query), "kl": "wt-wt"})
    # No identity headers here: fetch's identity layer supplies a coherent Chrome
    # set and per-egress cookies. A hand-built set needs ``raw_headers``, which
    # bypasses that layer and pins the request to bare curl with no browser
    # fallback.
    body = fetch(
        f"https://html.duckduckgo.com/html/?{params}",
        request=RequestParams(
            content=ContentParams(
                headers={
                    "Referer": "https://html.duckduckgo.com/html/",
                    **(headers or {}),
                },
            ),
            retry=RetryParams(
                retries=retries,
                timeout_sec=timeout_sec,
                connect_timeout_sec=connect_timeout_sec,
            ),
            # DDG serves its challenge with HTTP 200. Checking inside fetch raises
            # ``PuzzleChallengeError``, which ``auto`` answers by retrying in the
            # browser.
            observe=ObserveParams(body_validator=_duckduckgo_validate_body),
            policy=PolicyParams(transport=transport),
        ),
        session=session,
    )
    return _duckduckgo_parse(body.decode(), num_results)


def _duckduckgo_quote_bangs(query: str) -> str:
    """Quote DDG bang tokens to keep them in ordinary web search."""
    return " ".join(
        f"'{token}'" if token.startswith("!") else token for token in query.split()
    )


def _duckduckgo_validate_body(body: bytes) -> None:
    """Reject a DuckDuckGo success body that is really its challenge page."""
    _duckduckgo_check_captcha(body.decode(errors="replace"))


def _duckduckgo_check_captcha(page_html: str) -> None:
    """Raise when DDG returns its challenge page."""
    if _duckduckgo_is_challenge(bs4.BeautifulSoup(page_html, "html.parser")):
        raise PuzzleChallengeError("DuckDuckGo returned a challenge form.")


# Shared by the body validator and the empty-parse diagnostic so they cannot disagree.
def _duckduckgo_is_challenge(soup: bs4.BeautifulSoup) -> bool:
    """Whether this page is DDG's bot challenge rather than a results page."""
    return soup.select_one("form#challenge-form") is not None


def _duckduckgo_parse(
    page_html: str,
    max_results: int,
) -> list[SearchResult]:
    """Extract search results from DDG's HTML."""
    if max_results <= 0:
        return []  # The append-before-cap loop would otherwise return one.
    soup = bs4.BeautifulSoup(page_html, "html.parser")
    strip_scripts(soup)
    results: list[SearchResult] = []
    for container in soup.select("div#links > div.web-result"):
        link = container.select_one(
            "h2 a[href]",
        )
        if link is None:
            continue
        href = link.get("href")
        if not isinstance(href, str):
            continue
        url = _duckduckgo_extract_url(href)
        if url is None:
            continue
        title = clean_text(link.get_text(separator=" ", strip=True))
        if not title:
            continue

        snippet_el = container.select_one("a.result__snippet")
        snippet = (
            clean_text(snippet_el.get_text(separator=" ", strip=True))
            if snippet_el is not None
            else ""
        )
        results.append(SearchResult(url=url, title=title, snippet=snippet))
        if len(results) >= max_results:
            break

    if not results:
        if _duckduckgo_is_challenge(soup):
            logger.warning("No results parsed -- DDG served a bot challenge.")
        elif soup.select_one("div#links") is not None:
            logger.warning("No results parsed -- DDG returned an empty result list.")
        else:
            logger.warning("No results parsed -- DDG may have changed markup.")
    return results


def _duckduckgo_extract_url(href: str) -> str | None:
    """Extract a usable URL from DDG result links."""
    if not href:
        return None
    url = f"https:{href}" if href.startswith("//") else href
    parsed = urlparse(url)
    hostname = parsed.hostname
    if (
        hostname is not None
        and (hostname == "duckduckgo.com" or hostname.endswith(".duckduckgo.com"))
        and parsed.path == "/l/"
    ):
        wrapped = parse_qs(parsed.query).get("uddg", [])
        if wrapped:
            return wrapped[0]
    if parsed.scheme in {"http", "https"}:
        return url
    return None
