"""Fetch a URL and render its response to clean text.

:func:`fetch_web` is synchronous; an async caller lifts it with
``asyncio.to_thread``. Transport, extractor, and SSRF trust travel in its
``policy`` argument; the default validates each host to a public address
before connecting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, ParseError

import html
import re

from wesearch.fetch import (
    ContentParams,
    Extractor,
    FetchSession,
    PolicyParams,
    RequestParams,
    RetryParams,
    classify_challenge,
    fetch,
)
from wesearch.fetch.html_to_text import html_to_text
from wesearch.fetch.providers import google_news, reddit, x
from wesearch.fetch.providers.fallback import fetch_with_reader_fallback
from wesearch.fetch.providers.reader_proxy import fetch_reader_proxy
from wesearch.types.params import NO_BODY, NoBody


if TYPE_CHECKING:
    from treekle import PlainTree

    import defusedxml.common as _defused_common
    import defusedxml.ElementTree as _defused_etree  # noqa: N813 -- The public web adapter preserves the external provider's API name.

    from wesearch.mcp.types import FetchMethod
else:
    from wrapt import lazy_import

    _defused_etree = lazy_import("defusedxml.ElementTree")  # 7ms.
    _defused_common = lazy_import("defusedxml.common")  # 1ms.


@dataclass(frozen=True, slots=True, kw_only=True)
class FetchResult:
    """Extracted result of a :func:`fetch_web` call.

    Attributes:
      text: Extracted text, cut to ``max_chars`` when set. May be empty.
      url: The URL as requested.
      kind: How the body became text: ``"html"``, ``"markdown"``, ``"rss"``,
        or a Reddit JSON kind.
      truncated: Whether ``text`` was cut to fit ``max_chars``.

    """

    text: str
    url: str
    kind: str
    truncated: bool


def fetch_web(
    url: str,
    *,
    method: FetchMethod = "GET",
    json_body: PlainTree | NoBody = NO_BODY,
    form_body: dict[str, str] | None = None,
    max_chars: int | None = None,
    policy: PolicyParams | None = None,
    session: FetchSession | None = None,
) -> FetchResult:
    """Fetch a URL and render its response to clean text.

    A GET goes to the matching site provider (Reddit, Google News, X), else
    through a ladder that retries 403/429/503 via the reader proxy. A POST is
    fetched directly. The body is rendered by kind: an RSS/Atom feed to
    markdown, reader-proxy markdown and JSON as-is, Reddit JSON via its
    formatters, and HTML via ``policy.extractor``.

    Args:
      url: Target URL to fetch.
      method: HTTP method (``GET`` or ``POST``).
      json_body: JSON-serializable body for POST requests. Defaults to
        :data:`NO_BODY`, so ``json_body=None`` sends the JSON literal ``null``.
      form_body: Form-encoded body for POST requests.
      max_chars: Cap on the returned text; ``None`` returns it uncapped.
      policy: How to fetch and extract; ``None`` refuses private hosts and
        extracts with ``html2text``.
      session: Connections to reuse; ``None`` uses the process's shared session.

    Returns:
      result: The extracted text, requested URL, response kind, and whether
        the text was truncated. ``text`` may be empty on a successful fetch
        (e.g. a JS-only page); the raw markup is not substituted.

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
        session=session,
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
    method: FetchMethod,
    json_body: PlainTree | NoBody,
    form_body: dict[str, str] | None,
    policy: PolicyParams | None = None,
    session: FetchSession | None = None,
) -> tuple[bytes, str]:
    """Fetch a URL and classify the response for downstream extraction."""
    policy = PolicyParams() if policy is None else policy
    if method == "GET":
        if reddit.matches(url):
            body, payload = reddit.fetch_reddit(url, policy=policy, session=session)
            # The public export serves only the RSS payload; the JSON ones need OAuth.
            kind = {
                "rss": "rss",
            }[payload]
            _raise_success_challenge(url, body)
            return body, kind
        if google_news.matches(url):
            body, news_payload = google_news.fetch_google_news(
                url,
                policy=policy,
                session=session,
            )
            news_kind = "rss" if news_payload == "rss" else "html"
            _raise_success_challenge(url, body)
            return body, news_kind
        if x.matches(url):
            return fetch_reader_proxy(
                url,
                policy=policy,
                session=session,
            ), "markdown"
        body, via_proxy = fetch_with_reader_fallback(
            url,
            policy=policy,
            session=session,
        )
        if via_proxy:
            # The proxy validates its own 200; only an origin body can be a challenge.
            return body, "markdown"
        _raise_success_challenge(url, body)
        return body, "html"
    body = fetch(
        url,
        request=RequestParams(
            content=ContentParams(method=method, json=json_body, data=form_body),
            retry=RetryParams(timeout_sec=15),
            policy=policy,
        ),
        session=session,
    )
    _raise_success_challenge(url, body)
    return body, "html"


def _raise_success_challenge(url: str, body: bytes) -> None:
    """Raise when a successful retrieval is a cross-site interstitial."""
    error_type = classify_challenge(body, on_success_body=True)
    if error_type is not None:
        raise error_type(url=url, status=200, headers={}, body=body)


# An empty extraction is returned as is: substituting the raw markup would hand a
# model a page of HTML in place of the text it asked for.
def _extract_text(
    body: bytes,
    *,
    kind: str,
    url: str = "",
    extractor: Extractor = "html2text",
) -> str:
    """Extract result text from a response body (unbounded; caller caps)."""
    content = body.decode(errors="replace")
    if kind == "rss":
        return _format_rss(body)
    if kind == "markdown":  # Reader-proxy output, already extracted.
        return content
    # The body shape, not the method, decides: an HTML extractor would mangle
    # JSON, and a POST answered with HTML still wants extraction.
    if content.lstrip().startswith(("{", "[")):
        return content
    return html_to_text(content, extractor=extractor, url=url)


def _format_rss(body: bytes) -> str:
    """Format an RSS or Atom feed as readable markdown."""
    try:
        root = _defused_etree.fromstring(body)
    except (ParseError, _defused_common.DefusedXmlException):
        return body.decode(errors="replace")
    if _local_name(root.tag) == "feed":
        return _format_atom(root).rstrip()
    channel = root.find("channel") if _local_name(root.tag) == "rss" else root
    if channel is None:
        return body.decode(errors="replace")
    lines: list[str] = []
    feed_title = _child_text(channel, "title").strip()
    if feed_title:
        lines.append(f"# {feed_title}\n")
    for item in _children(channel, "item"):
        _append_rss_item(item, lines)
    return "\n".join(lines).rstrip()


def _local_name(tag: str) -> str:
    """Return an XML tag without its namespace."""
    return tag.rpartition("}")[2]


def _format_atom(feed: Element[str]) -> str:
    """Format an Atom feed as readable markdown."""
    lines: list[str] = []
    feed_title = _child_text(feed, "title").strip()
    if feed_title:
        lines.append(f"# {feed_title}\n")
    for entry in _children(feed, "entry"):
        _append_atom_entry(entry, lines)
    return "\n".join(lines)


def _child_text(parent: Element[str], name: str) -> str:
    """Return text for the first direct child with local tag name ``name``."""
    child = _child(parent, name)
    return (child.text or "") if child is not None else ""


def _child(parent: Element[str], name: str) -> Element[str] | None:
    """Return the first direct child with local tag name ``name``."""
    for child in list(parent):
        if _local_name(child.tag) == name:
            return child
    return None


def _children(parent: Element[str], name: str) -> list[Element[str]]:
    """Return direct children with local tag name ``name``."""
    return [child for child in list(parent) if _local_name(child.tag) == name]


def _append_atom_entry(entry: Element[str], lines: list[str]) -> None:
    """Append one Atom entry to ``lines``."""
    title = _child_text(entry, "title").strip()
    author = _atom_author(entry)
    updated = (_child_text(entry, "updated") or _child_text(entry, "published")).strip()
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
    return _child_text(author, "name").strip()


def _atom_link(entry: Element[str]) -> str:
    """Return the first Atom link href, if present."""
    for link in _children(entry, "link"):
        href = link.attrib.get("href", "")
        if href:
            return href.strip()
    return ""


# Reads the whole subtree, not ``.text``: Atom ``type="xhtml"`` nests the body in a
# ``<div>``, leaving no direct text.
def _atom_content(entry: Element[str]) -> str:
    """Return cleaned Atom content or summary text."""
    raw = _element_text(entry, "content") or _element_text(entry, "summary") or ""
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(raw)).split())


def _element_text(parent: Element[str], name: str) -> str:
    """Return the full text of the first ``name`` child, descendants included."""
    child = _child(parent, name)
    return "".join(child.itertext()) if child is not None else ""


def _append_rss_item(item: Element[str], lines: list[str]) -> None:
    """Append one feed item (heading + meta + cluster bullets) to ``lines``."""
    title = _child_text(item, "title").strip()
    link = _child_text(item, "link").strip()
    source_elem = _child(item, "source")
    source = (source_elem.text or "").strip() if source_elem is not None else ""
    pub_date = _child_text(item, "pubDate").strip()
    if title:
        lines.append(f"## {title}")
    meta_parts = [p for p in (source, pub_date) if p]
    if meta_parts:
        lines.append(" -- ".join(meta_parts))
    if link:
        lines.append(link)
    # The first cluster entry duplicates the item title; siblings follow.
    description = _child_text(item, "description")
    cluster = _parse_rss_cluster(description) if description else []
    for sibling_title, sibling_link, sibling_source in cluster[1:]:
        suffix = f" -- {sibling_source}" if sibling_source else ""
        lines.append(f"- [{sibling_title}]({sibling_link}){suffix}")
    lines.append("")


def _parse_rss_cluster(description_html: str) -> list[tuple[str, str, str]]:
    """Parse the ``<ol>`` of sibling stories embedded in a feed item description."""
    # A Google News cluster ``<li>`` is always
    # ``<li><a href="..">title</a> &nbsp;&nbsp;<font ..>source</font></li>``.
    return [
        (
            html.unescape(match.group(2)).strip(),
            match.group(1),
            html.unescape(match.group(3) or "").strip(),
        )
        for match in re.finditer(
            r'<li>\s*<a\s+[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
            r"(?:[^<]*<font[^>]*>([^<]+)</font>)?",
            description_html,
            re.IGNORECASE | re.DOTALL,
        )
    ]
