"""The DuckDuckGo backend: scrapes the HTML endpoint, no key required.

The always-available backend, which is why it is the default when no SearXNG
instance is configured.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Final
from urllib.parse import parse_qs, urlencode, urlparse

import logging

from wesearch.chrome.useragents import user_agent_pool
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


@cache
def _duckduckgo_user_agent() -> str:
    """A PROCESS-STABLE User-Agent for DuckDuckGo (drawn once, reused).

    DuckDuckGo derives its ``vqd`` anti-bot token from ``(query, User-Agent)`` and
    treats a UA that shifts between the results page and its follow-ups as a bot
    (which lowers the IP's reputation and triggers CAPTCHAs). A stable UA keeps
    the token valid across requests -- unlike the per-query UA the Google path
    uses. Cached, so the whole process presents one consistent DDG client.
    """
    pool = user_agent_pool("chrome_android")
    return f"{pool[0]} NSTNWV"


def duckduckgo(
    query: str,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
    *,
    max_query_chars: int = 499,
    timeout_sec: float = 30.0,
    connect_timeout_sec: float = 3.0,
    retries: int = 2,
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
      timeout_sec: HTTP ceiling per attempt.
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
            f"DuckDuckGo query exceeds {max_query_chars} characters (got {len(query)})."
        )
    request_headers = {
        "User-Agent": _duckduckgo_user_agent(),
        "Accept": "*/*",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Accept-Language": "all,all-ALL;q=0.7",
        "Referer": _DUCKDUCKGO_URL,
    }
    if headers:
        request_headers.update(headers)
    # The query goes in the URL, not a POST body: DuckDuckGo's HTML endpoint now
    # drops a POSTed query and serves its empty homepage (``body--home``),
    # yielding zero results. A GET with ``q`` in the query string returns the
    # real results page. ``kl=wt-wt`` keeps the region-neutral ("no region")
    # results the POST form sent.
    params = urlencode({"q": _duckduckgo_quote_bangs(query), "kl": "wt-wt"})
    # Send exactly these headers: the GSA mobile User-Agent must not be paired
    # with fetch's default desktop Chrome ``sec-ch-ua``/``sec-ch-ua-platform``,
    # whose drift from the UA can trip DuckDuckGo's bot check.
    body, _ = fetch(
        f"{_DUCKDUCKGO_URL}?{params}",
        request=RequestParams(
            content=ContentParams(headers=request_headers, raw_headers=True),
            retry=RetryParams(
                retries=retries,
                timeout_sec=timeout_sec,
                connect_timeout_sec=connect_timeout_sec,
            ),
            # The challenge check runs INSIDE fetch, as Google's does. DDG
            # serves its puzzle with HTTP 200, so checking the returned body
            # here would raise past fetch's own `except BotDetectionError` --
            # the hook that learns the domain and retries through Zendriver.
            # Detected outside, an automatic-transport caller just fails.
            observe=ObserveParams(body_validator=_duckduckgo_validate_body),
            policy=PolicyParams(transport=transport),
        ),
    )
    return _duckduckgo_parse(body.decode("utf-8"), num_results)


def _duckduckgo_quote_bangs(query: str) -> str:
    """Quote DDG bang tokens to keep them in ordinary web search."""
    return " ".join(
        f"'{token}'" if token.startswith("!") else token for token in query.split()
    )


def _duckduckgo_validate_body(body: bytes) -> None:
    """Reject a DuckDuckGo success body that is really its challenge page."""
    _duckduckgo_check_captcha(body.decode("utf-8", "replace"))


def _duckduckgo_check_captcha(page_html: str) -> None:
    """Raise when DDG returns its challenge page."""
    soup = bs4.BeautifulSoup(page_html, "html.parser")
    if soup.select_one("form#challenge-form") is not None:
        raise PuzzleChallengeError("DuckDuckGo returned a challenge form.")


def _duckduckgo_extract_url(href: str) -> str | None:
    """Extract a usable URL from DDG result links."""
    if not href:
        return None
    url = f"https:{href}" if href.startswith("//") else href
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if (
        hostname == "duckduckgo.com" or hostname.endswith(".duckduckgo.com")
    ) and parsed.path == "/l/":
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
        return []  # append-before-cap would otherwise return one at max=0
    soup = bs4.BeautifulSoup(page_html, "html.parser")
    strip_scripts(soup)
    results: list[SearchResult] = []
    for container in soup.select("div#links > div.web-result"):
        link = container.select_one("h2 a[href]")
        if link is None:
            continue
        href = link.get("href", "")
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
        logger.warning(
            "No results parsed -- DDG may have changed markup.",
        )
    return results
