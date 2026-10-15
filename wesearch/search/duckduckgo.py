"""The DuckDuckGo backend: scrapes the HTML endpoint, no key required.

The always-available backend, which is why it is the default when no SearXNG
instance is configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
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
from wesearch.search.custom_types import (
    SearchError,
    SearchResult,
    clean_text,
    strip_scripts,
)
from wesearch.types.errors import PuzzleChallengeError


if TYPE_CHECKING:
    import bs4
else:
    from wrapt import lazy_import

    bs4 = lazy_import("bs4")


logger = logging.getLogger(__name__)


_DUCKDUCKGO_URL: Final = "https://html.duckduckgo.com/html/"


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
) -> list[SearchResult]:
    """Scrape DuckDuckGo's HTML-only endpoint.

    More reliable than Google scraping -- DDG doesn't block as
    aggressively.

    Args:
      query: Search query string.
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to fetch.
      max_query_chars: Reject a query longer than this. DuckDuckGo's HTML
        endpoint silently drops overlong queries, so fail loudly instead.
      timeout_sec: Ceiling per attempt, browser leg included. The page is a
        few tens of KB, so a slower answer means a dropped egress, not a slow one.
      connect_timeout_sec: Ceiling on the handshake alone, so an unroutable
        egress is detected in a handshake rather than a full request. A dropped
        SYN draws no RST, so only the clock reports it. Sized against the slow
        case, not the local one: an intercontinental origin (Zurich) measures
        165ms RTT and a 0.39s handshake from here, so 3s absorbs that plus a
        lost SYN (+1s initial RTO) and still fails a black-holed host in 3s
        rather than the full ``timeout_sec``.
      retries: Retry attempts for a transient failure. Multiplies with
        ``timeout_sec``: an egress that cannot reach the endpoint at all burns
        ``(retries + 1) * timeout_sec`` before raising, so a caller on a
        deadline lowers both rather than either alone.
      transport: Retrieval transport; ``"auto"`` applies domain routing.

    Returns:
      results: Parsed search results.

    """
    if num_results < 0:
        raise ValueError(f"'num_results' must be >= 0, got {num_results}.")
    if num_results == 0:
        return []  # Nothing to return, so do not pay for a round-trip.
    if len(query) > max_query_chars:
        raise SearchError(
            f"DuckDuckGo query exceeds {max_query_chars} characters (got {len(query)}).",
        )
    # The query goes in the URL, not a POST body: DuckDuckGo's HTML endpoint now
    # drops a POSTed query and serves its empty homepage (``body--home``),
    # yielding zero results. A GET with ``q`` in the query string returns the
    # real results page. ``kl=wt-wt`` keeps the region-neutral ("no region")
    # results the POST form sent.
    params = urlencode({"q": _duckduckgo_quote_bangs(query), "kl": "wt-wt"})
    # No identity headers of our own: fetch's identity layer sends the coherent
    # Chrome set curl impersonates and keeps DDG's cookies per egress. A
    # hand-built set needs ``raw_headers``, which skipped that layer and pinned
    # the request to bare curl -- DDG then blocked it in 0.16s with no browser
    # fallback to escalate to.
    body, _ = fetch(
        f"{_DUCKDUCKGO_URL}?{params}",
        request=RequestParams(
            content=ContentParams(
                headers={"Referer": _DUCKDUCKGO_URL, **(headers or {})},
            ),
            retry=RetryParams(
                retries=retries,
                timeout_sec=timeout_sec,
                connect_timeout_sec=connect_timeout_sec,
            ),
            # The challenge check runs INSIDE fetch. DDG serves its puzzle with
            # HTTP 200, so a check here turns that into a typed
            # ``PuzzleChallengeError``, which ``auto`` answers by retrying in the
            # browser and learning the domain.
            observe=ObserveParams(body_validator=_duckduckgo_validate_body),
            policy=PolicyParams(transport=transport),
        ),
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


# One predicate, shared by the validator that rejects the page and the diagnostic that
# explains an empty parse. Written twice, a selector change at DDG would make the
# diagnostic report "markup changed" for exactly the challenge page it exists to tell
# apart.
def _duckduckgo_is_challenge(soup: bs4.BeautifulSoup) -> bool:
    """Whether this page is DDG's bot challenge rather than a results page."""
    return soup.select_one("form#challenge-form") is not None


def _duckduckgo_check_captcha(page_html: str) -> None:
    """Raise when DDG returns its challenge page."""
    if _duckduckgo_is_challenge(bs4.BeautifulSoup(page_html, "html.parser")):
        raise PuzzleChallengeError("DuckDuckGo returned a challenge form.")


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


def _duckduckgo_parse(
    page_html: str,
    max_results: int,
) -> list[SearchResult]:
    """Extract search results from DDG's HTML."""
    if max_results <= 0:
        return []  # append-before-cap would otherwise return one at max=0.
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
        # Name which of three different causes this was. The single "markup
        # changed" message used to be emitted for a bot-challenge page too,
        # which sent an investigation after a parser that was working fine.
        if _duckduckgo_is_challenge(soup):
            logger.warning("No results parsed -- DDG served a bot challenge.")
        elif soup.select_one("div#links") is not None:
            logger.warning("No results parsed -- DDG returned an empty result list.")
        else:
            logger.warning("No results parsed -- DDG may have changed markup.")
    return results
