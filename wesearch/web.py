"""Fetch a URL to clean text, and search the web -- sagent-independent.

The two public entry points return plain data (no tool-framework types):

  - :func:`fetch_web` fetches a URL (GET or POST) and renders the response to
    clean text, dispatching GET to the site-specific providers (Reddit, Google
    News, X) and falling back to a reader-proxy-aware ladder, then extracting
    the body by kind (RSS/Atom feed, reader-proxy markdown, or HTML via
    trafilatura).
  - :func:`search_web` wraps :func:`wesearch.search.search` and flattens
    its typed results to ``{"url", "title", "snippet"}`` dicts.

Both are synchronous; an async caller lifts them with ``asyncio.to_thread``.
SSRF pinning is an app-level concern the caller opts into: :func:`fetch_web`
accepts an optional ``validated_hosts`` resolver and threads it into every
underlying fetch (sagent's ``WebFetch`` passes its own). Default ``None`` leaves
the wesearch core fetch unpinned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal
from xml.etree.ElementTree import Element, ParseError

import html
import re

from wesearch.fetch import (
    Content,
    Policy,
    RequestParams,
    Retry,
    classify_challenge,
    fetch,
)
from wesearch.lib.custom_json import JSONValue
from wesearch.providers import google_news, reddit, x
from wesearch.providers.fallback import fetch_with_reader_fallback
from wesearch.search import search


if TYPE_CHECKING:
    import defusedxml.common as _defused_common
    import defusedxml.ElementTree as _defused_etree  # noqa: N813 -- match lazy_import name
    import trafilatura
else:
    from wrapt import lazy_import

    trafilatura = lazy_import("trafilatura")  # ~150ms
    # Bind the submodules directly so ``.fromstring`` / ``.DefusedXmlException``
    # resolve without an eager top-level ``import defusedxml.ElementTree``.
    _defused_etree = lazy_import("defusedxml.ElementTree")
    _defused_common = lazy_import("defusedxml.common")


# The only HTTP methods this module supports.
HttpMethod = Literal["GET", "POST"]

# Response kinds classified by the fetch path; select the extraction branch.
_KIND_HTML: Final = "html"  # raw HTML, needs trafilatura
_KIND_MARKDOWN: Final = "markdown"  # already-extracted markdown (reader proxy)
_KIND_RSS: Final = "rss"  # RSS 2.0 / Atom feed XML, needs feed formatter
# Maps a reddit.RedditPayload to its extraction kind. The public export serves
# only the RSS path; the JSON payloads ride the fenced OAuth path.
_REDDIT_PAYLOAD_KINDS: Final[dict[str, str]] = {
    "rss": _KIND_RSS,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class WebFetchResult:
    """Extracted result of a :func:`fetch_web` call.

    Attributes:
      text: The extracted, possibly-truncated text body.
      url: The URL that was fetched.
      kind: The response kind that drove extraction (one of the ``_KIND_*``
        constants).
      truncated: Whether ``text`` was cut to fit ``max_chars``.

    """

    text: str
    url: str
    kind: str
    truncated: bool


def fetch_web(
    url: str,
    *,
    method: HttpMethod = "GET",
    json_body: JSONValue = None,
    form_body: dict[str, str] | None = None,
    max_chars: int | None = None,
    policy: Policy | None = None,
) -> WebFetchResult:
    """Fetch a URL and render its response to clean text.

    GET requests are first offered to the site-specific wesearch providers
    (Reddit, Google News, X); the first whose ``matches`` returns True owns the
    fetch. URLs with no provider match go through ``fetch_with_reader_fallback``
    (a bot-wall-aware ladder that retries 403/429/503 via the reader proxy);
    POST bodies go through a direct :func:`wesearch.fetch.fetch`. The bytes
    are then rendered by kind: an RSS/Atom feed to markdown, reader-proxy
    markdown as-is, Reddit JSON via the comment/listing formatters, and HTML via
    trafilatura main-content extraction.

    Args:
      url: Target URL to fetch.
      method: HTTP method (``GET`` or ``POST``).
      json_body: JSON-serializable body for POST requests.
      form_body: Form-encoded body for POST requests.
      max_chars: Optional cap on the returned text. ``None`` (default) returns
        the full extracted text uncapped -- capping is a caller's presentation
        policy, and a hardcoded ceiling here would silently defeat a caller that
        appends its own truncation notice. When set, the text is cut to this
        length and ``truncated`` reports whether the cut occurred.
      policy: Transport and trust, forwarded into every underlying fetch
        (providers, reader-proxy ladder, and the direct POST path). Defaults to
        the safe ``untrusted`` level, which validates each host to a public
        address before connecting.

    Returns:
      result: A :class:`WebFetchResult` with the extracted text, the fetched
        URL, the response kind, and whether the text was truncated.

    Raises:
      BotDetectionError: When the fetch (or a success-body challenge check)
        classifies the response as a bot wall.
      FetchError: On an HTTP failure.

    """
    body, kind = _fetch_body(
        url,
        method=method,
        json_body=json_body,
        form_body=form_body,
        policy=policy,
    )
    text = _extract_text(body, kind=kind, method=method)
    if max_chars is None:
        return WebFetchResult(text=text, url=url, kind=kind, truncated=False)
    truncated = len(text) > max_chars
    return WebFetchResult(
        text=text[:max_chars], url=url, kind=kind, truncated=truncated
    )


def search_web(
    query: str,
    *,
    backend: Literal["duckduckgo", "searxng"] | None = None,
    num_results: int = 10,
) -> list[dict[str, str]]:
    """Search the web and return flat ``{"url", "title", "snippet"}`` dicts.

    Args:
      query: Search query string.
      backend: Search backend; defaults to the configured backend (DuckDuckGo,
        or SearXNG when ``SEARXNG_URL`` is set).
      num_results: Maximum number of results to return.

    Returns:
      results: One dict per hit with ``url``, ``title``, and ``snippet`` keys.

    """
    results = search(query, backend=backend, num_results=num_results)
    return [{"url": r.url, "title": r.title, "snippet": r.snippet} for r in results]


def _fetch_body(
    url: str,
    *,
    method: HttpMethod,
    json_body: JSONValue,
    form_body: dict[str, str] | None,
    policy: Policy | None = None,
) -> tuple[bytes, str]:
    """Fetch a URL and classify the response for downstream extraction.

    Args:
      url: Target URL to fetch.
      method: HTTP method (``GET`` or ``POST``).
      json_body: JSON-serializable body for POST requests.
      form_body: Form-encoded body for POST requests.
      policy: Transport and trust forwarded into every fetch.

    Returns:
      body: Raw response bytes.
      kind: One of the ``_KIND_*`` constants; selects the post-processing branch
        in :func:`_extract_text`.

    """
    policy = Policy() if policy is None else policy
    if method == "GET":
        if reddit.matches(url):
            body, payload = reddit.fetch_reddit(url, policy=policy)
            kind = _REDDIT_PAYLOAD_KINDS[payload]
            _raise_success_challenge(url, body)
            return body, kind
        if google_news.matches(url):
            body, news_payload = google_news.fetch_google_news(url, policy=policy)
            news_kind = _KIND_RSS if news_payload == "rss" else _KIND_HTML
            _raise_success_challenge(url, body)
            return body, news_kind
        if x.matches(url):
            return x.fetch_x(url, policy=policy), _KIND_MARKDOWN
        body, via_proxy = fetch_with_reader_fallback(url, policy=policy)
        if via_proxy:
            # The proxy already validated its own 200; a challenge would ride
            # the origin path only, which the fallback ladder never reaches.
            return body, _KIND_MARKDOWN
        _raise_success_challenge(url, body)
        return body, _KIND_HTML
    body, _session = fetch(
        url,
        request=RequestParams(
            content=Content(method=method, json=json_body, data=form_body),
            retry=Retry(timeout_sec=15),
            policy=policy,
        ),
    )
    _raise_success_challenge(url, body)
    return body, _KIND_HTML


def _extract_text(body: bytes, *, kind: str, method: HttpMethod) -> str:
    """Extract result text from a response body (unbounded; caller caps).

    ``kind`` selects the post-processing path:
      - ``_KIND_RSS``: parse as RSS 2.0 / Atom XML and format as markdown.
      - ``_KIND_MARKDOWN``: return as-is (the reader-proxy rung already rendered
        to markdown; running trafilatura on it would strip structure).
      - ``_KIND_HTML``: trafilatura main-content extraction, with a raw-content
        fallback when extraction returns nothing.
    """
    content = body.decode("utf-8", errors="replace")

    if kind == _KIND_RSS:
        return _format_rss(body)
    if kind == _KIND_MARKDOWN:
        return content
    if method == "POST" or content.lstrip().startswith(("{", "[")):
        return content
    extracted = trafilatura.extract(
        content,
        include_links=True,
        include_tables=True,
        # Emits a YAML front-matter block (title, url, description, date,
        # license) ahead of the body. Two reasons, neither cosmetic:
        #
        # 1. It recovers substance the body extraction drops. trafilatura scores
        #    article-shaped prose, so a page whose content is a short fragment
        #    loses it -- every Merriam-Webster entry returned the subscription
        #    advert and discarded the definition, which the page states verbatim
        #    in its meta description. Since that advert is non-empty, the
        #    ``or content`` fallback below never fired: the tool reported success
        #    on the wrong text. Eight other option combinations were measured
        #    (bare defaults, favor_recall, no_fallback, prune_xpath=None, ...);
        #    this is the only one that recovers it.
        # 2. It supplies the page URL, so relative links resolve absolute
        #    (``](#comment37161)`` -> ``](https://host#comment37161)``) instead
        #    of emitting fragments no reader can follow.
        #
        # Cost is ~200-580 chars of front-matter per page against a 400k cap.
        with_metadata=True,
    )
    return extracted or content


def _raise_success_challenge(url: str, body: bytes) -> None:
    """Raise when a successful retrieval is a cross-site interstitial."""
    error_type = classify_challenge(body, on_success_body=True)
    if error_type is not None:
        raise error_type(url=url, status=200, headers={}, body=body)


# Matches one ``<li>`` entry in a Google News RSS cluster description.
# The description body is a small fragment of HTML with the same shape
# every time: ``<ol><li><a href="..">title</a> &nbsp;&nbsp;<font ..>source
# </font></li>...</ol>``. We parse with a regex rather than an HTML
# parser because the fragment is well-formed-by-construction and the
# regex stays under ten lines.
_RSS_CLUSTER_LINK_RE = re.compile(
    r'<li>\s*<a\s+[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
    r"(?:[^<]*<font[^>]*>([^<]+)</font>)?",
    re.IGNORECASE | re.DOTALL,
)


def _format_rss(body: bytes) -> str:
    """Format an RSS or Atom feed as readable markdown.

    Args:
      body: Raw feed XML.

    Returns:
      formatted: Markdown text suitable for direct output.

    """
    try:
        root = _defused_etree.fromstring(body)
    except (ParseError, _defused_common.DefusedXmlException):
        return body.decode("utf-8", errors="replace")
    if _local_name(root.tag) == "feed":
        return _format_atom(root).rstrip()
    channel = root.find("channel") if _local_name(root.tag) == "rss" else root
    if channel is None:
        return body.decode("utf-8", errors="replace")
    lines: list[str] = []
    feed_title = (_child_text(channel, "title") or "").strip()
    if feed_title:
        lines.append(f"# {feed_title}\n")
    for item in _children(channel, "item"):
        _append_rss_item(item, lines)
    return "\n".join(lines).rstrip()


def _format_atom(feed: Element[str]) -> str:
    """Format an Atom feed as readable markdown."""
    lines: list[str] = []
    feed_title = (_child_text(feed, "title") or "").strip()
    if feed_title:
        lines.append(f"# {feed_title}\n")
    for entry in _children(feed, "entry"):
        _append_atom_entry(entry, lines)
    return "\n".join(lines)


def _append_atom_entry(entry: Element[str], lines: list[str]) -> None:
    """Append one Atom entry to ``lines``."""
    title = (_child_text(entry, "title") or "").strip()
    author = _atom_author(entry)
    updated = (
        _child_text(entry, "updated") or _child_text(entry, "published") or ""
    ).strip()
    link = _atom_link(entry)
    content = _atom_content(entry)
    if title:
        lines.append(f"## {title}")
    meta_parts = [p for p in (author, updated) if p]
    if meta_parts:
        lines.append(" -- ".join(meta_parts))
    if link:
        lines.append(link)
    if content:
        lines.append(content[:500])
    lines.append("")


def _atom_author(entry: Element[str]) -> str:
    """Return the Atom author name, if present."""
    author = _child(entry, "author")
    if author is None:
        return ""
    return (_child_text(author, "name") or "").strip()


def _atom_link(entry: Element[str]) -> str:
    """Return the first Atom link href, if present."""
    for link in _children(entry, "link"):
        href = link.attrib.get("href")
        if href:
            return href.strip()
    return ""


def _atom_content(entry: Element[str]) -> str:
    """Return cleaned Atom content or summary text."""
    raw = _child_text(entry, "content") or _child_text(entry, "summary") or ""
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(raw)).split())


def _append_rss_item(item: Element[str], lines: list[str]) -> None:
    """Append one feed item (heading + meta + cluster bullets) to ``lines``."""
    title = (_child_text(item, "title") or "").strip()
    link = (_child_text(item, "link") or "").strip()
    source_elem = _child(item, "source")
    source = (source_elem.text or "").strip() if source_elem is not None else ""
    pub_date = (_child_text(item, "pubDate") or "").strip()
    if title:
        lines.append(f"## {title}")
    meta_parts = [p for p in (source, pub_date) if p]
    if meta_parts:
        lines.append(" -- ".join(meta_parts))
    if link:
        lines.append(link)
    # The first cluster entry duplicates the item title; siblings follow.
    cluster = _parse_rss_cluster(_child_text(item, "description") or "")
    for sibling_title, sibling_link, sibling_source in cluster[1:]:
        suffix = f" -- {sibling_source}" if sibling_source else ""
        lines.append(f"- [{sibling_title}]({sibling_link}){suffix}")
    lines.append("")


def _local_name(tag: str) -> str:
    """Return an XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def _children(parent: Element[str], name: str) -> list[Element[str]]:
    """Return direct children with local tag name ``name``."""
    return [child for child in list(parent) if _local_name(child.tag) == name]


def _child(parent: Element[str], name: str) -> Element[str] | None:
    """Return the first direct child with local tag name ``name``."""
    for child in list(parent):
        if _local_name(child.tag) == name:
            return child
    return None


def _child_text(parent: Element[str], name: str) -> str | None:
    """Return text for the first direct child with local tag name ``name``."""
    child = _child(parent, name)
    return child.text if child is not None else None


def _parse_rss_cluster(description_html: str) -> list[tuple[str, str, str]]:
    """Parse the ``<ol>`` of sibling stories embedded in a feed item description.

    Args:
      description_html: HTML fragment from an ``<item><description>``.

    Returns:
      entries: ``(title, link, source)`` tuples, in document order. Empty list
        when the fragment lacks Google-News-style cluster markup.

    """
    return [
        (
            html.unescape(match.group(2)).strip(),
            match.group(1),
            html.unescape(match.group(3) or "").strip(),
        )
        for match in _RSS_CLUSTER_LINK_RE.finditer(description_html)
    ]
