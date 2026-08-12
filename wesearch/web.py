"""Fetch a URL to clean text -- sagent-independent.

:func:`fetch_web` returns plain data (no tool-framework types): it fetches a URL
(GET or POST) and renders the response to clean text, dispatching GET to the
site-specific providers (Reddit, Google News, X) and falling back to a
reader-proxy-aware ladder, then extracting the body by kind (RSS/Atom feed,
reader-proxy markdown, or HTML via the extractor named by
:class:`wesearch.types.params.PolicyParams`).

Synchronous; an async caller lifts it with ``asyncio.to_thread``. Transport,
extractor, and SSRF trust all travel in the ``policy`` argument; omitting it
takes the safe default, which validates each host to a public address before
connecting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from xml.etree.ElementTree import Element, ParseError

import html
import re

from wesearch.fetch import (
    ContentParams,
    Extractor,
    PolicyParams,
    RequestParams,
    RetryParams,
    classify_challenge,
    fetch,
)
from wesearch.fetch.custom_types import HttpMethod
from wesearch.fetch.extractor.html2text import extract_html2text
from wesearch.fetch.extractor.markdownify import extract_markdownify
from wesearch.fetch.extractor.raw import extract_raw
from wesearch.fetch.extractor.trafilatura import extract_trafilatura
from wesearch.fetch.providers import google_news, reddit, x
from wesearch.fetch.providers.fallback import fetch_with_reader_fallback
from wesearch.lib.custom_json import JSONValue
from wesearch.types.extractor import Extract


if TYPE_CHECKING:
    import defusedxml.common as _defused_common
    import defusedxml.ElementTree as _defused_etree  # noqa: N813 -- match lazy_import name
else:
    from wrapt import lazy_import

    # Bind the submodules directly so ``.fromstring`` / ``.DefusedXmlException``
    # resolve without an eager top-level ``import defusedxml.ElementTree``.
    _defused_etree = lazy_import("defusedxml.ElementTree")
    _defused_common = lazy_import("defusedxml.common")


# The extractor each ``Extractor`` name selects. A dict rather than a chain of
# ifs so an unknown name is a KeyError here, not a silent fall-through to the
# default -- and so the set of extractors is one readable list.
_EXTRACTORS: Final[dict[Extractor, Extract]] = {
    "html2text": extract_html2text,
    "markdownify": extract_markdownify,
    "trafilatura": extract_trafilatura,
    "raw": extract_raw,
}

# Response kinds classified by the fetch path; select the extraction branch.
_KIND_HTML: Final = "html"  # raw HTML, needs an extractor
_KIND_MARKDOWN: Final = "markdown"  # already-extracted markdown (reader proxy)
_KIND_RSS: Final = "rss"  # RSS 2.0 / Atom feed XML, needs feed formatter
# Maps a reddit.RedditPayload to its extraction kind. The public export serves
# only the RSS path; the JSON payloads ride the fenced OAuth path.
_REDDIT_PAYLOAD_KINDS: Final[dict[str, str]] = {
    "rss": _KIND_RSS,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class FetchResult:
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
    policy: PolicyParams | None = None,
) -> FetchResult:
    """Fetch a URL and render its response to clean text.

    GET requests are first offered to the site-specific wesearch providers
    (Reddit, Google News, X); the first whose ``matches`` returns True owns the
    fetch. URLs with no provider match go through ``fetch_with_reader_fallback``
    (a bot-wall-aware ladder that retries 403/429/503 via the reader proxy);
    POST bodies go through a direct :func:`wesearch.fetch.fetch`. The bytes
    are then rendered by kind: an RSS/Atom feed to markdown, reader-proxy
    markdown as-is, Reddit JSON via the comment/listing formatters, and HTML via
    ``policy.extractor`` (``html2text`` by default).

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
      policy: Transport, extractor, and trust, forwarded into every underlying
        fetch (providers, reader-proxy ladder, and the direct POST path).
        Defaults to the safe ``untrusted`` level, which validates each host to a
        public address before connecting, and to the ``html2text`` extractor.

    Returns:
      result: A :class:`FetchResult` with the extracted text, the fetched
        URL, the response kind, and whether the text was truncated. ``text``
        may be EMPTY on a successful fetch: an extractor that finds nothing
        returns nothing, and the raw markup is deliberately not substituted
        (see :func:`_extract_text`). A JS-only page is the common case.

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
    text = _extract_text(
        body,
        kind=kind,
        url=url,
        extractor=(policy or PolicyParams()).extractor,
    )
    if max_chars is None:
        return FetchResult(text=text, url=url, kind=kind, truncated=False)
    truncated = len(text) > max_chars
    return FetchResult(text=text[:max_chars], url=url, kind=kind, truncated=truncated)


def _fetch_body(
    url: str,
    *,
    method: HttpMethod,
    json_body: JSONValue,
    form_body: dict[str, str] | None,
    policy: PolicyParams | None = None,
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
    policy = PolicyParams() if policy is None else policy
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
            content=ContentParams(method=method, json=json_body, data=form_body),
            retry=RetryParams(timeout_sec=15),
            policy=policy,
        ),
    )
    _raise_success_challenge(url, body)
    return body, _KIND_HTML


def _extract_text(
    body: bytes,
    *,
    kind: str,
    url: str = "",
    extractor: Extractor = "html2text",
) -> str:
    """Extract result text from a response body (unbounded; caller caps).

    ``kind`` selects the post-processing path:
      - ``_KIND_RSS``: parse as RSS 2.0 / Atom XML and format as markdown.
      - ``_KIND_MARKDOWN``: return as-is (the reader-proxy rung already rendered
        to markdown; re-extracting it would strip structure).
      - ``_KIND_HTML``: the ``extractor`` named by the policy. Its output is
        returned even when empty: ``Extract`` permits that, and substituting the
        raw markup would hand a model a page of HTML in place of the text it
        asked for.
    """
    content = body.decode("utf-8", errors="replace")

    if kind == _KIND_RSS:
        return _format_rss(body)
    if kind == _KIND_MARKDOWN:
        return content
    # A JSON body is returned verbatim -- it is already structured text, and an
    # HTML extractor would mangle it. The METHOD does not decide this: a POST
    # that answers with HTML gets the same extraction a GET would, since the
    # caller asked for a page rendered as text either way.
    if content.lstrip().startswith(("{", "[")):
        return content
    # Verifying a change to this call: sagent's WebFetch caches GET results for
    # 15 minutes per (transport, url, extractor), so it replays the pre-change
    # text and a working fix reads as a failed one. Prove it via fetch_web in a
    # FRESH process, not by re-running the tool.
    return _EXTRACTORS[extractor](content, url=url)


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
    """Return cleaned Atom content or summary text.

    Reads the element's whole subtree, not just its direct ``.text``: Atom's
    ``type="xhtml"`` form nests the body in a ``<div>``, so the entry has NO
    direct text and a ``.text``-only read returned an empty body for a
    perfectly ordinary feed.
    """
    raw = _element_text(entry, "content") or _element_text(entry, "summary") or ""
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(raw)).split())


def _element_text(parent: Element[str], name: str) -> str:
    """Return the full text of the first ``name`` child, descendants included."""
    child = _child(parent, name)
    return "".join(child.itertext()) if child is not None else ""


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
