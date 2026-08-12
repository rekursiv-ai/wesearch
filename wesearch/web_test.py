"""Tests for ``wesearch.web``: fetch-to-text pipeline and web search."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from wesearch.fetch import PolicyParams, RequestParams
from wesearch.types.errors import CloudflareChallengeError, FetchError
from wesearch.web import (
    _KIND_HTML,
    _KIND_MARKDOWN,
    _KIND_RSS,
    FetchResult,
    _extract_text,
    _fetch_body,
    _format_rss,
    _parse_rss_cluster,
    fetch_web,
)


def _extract_nothing(html: str, *, url: str = "") -> str:
    """An extractor that finds no text, to exercise the raw-content fallback."""
    del html, url
    return ""


def _extract_wrong(html: str, *, url: str = "") -> str:
    """An extractor whose output must never appear, to prove it was not run."""
    del html, url
    return "WRONG"


# ---------------------------------------------------------------------------
# fetch_web / _fetch_body dispatch
# ---------------------------------------------------------------------------


def test_fetch_web_html_path_extracts() -> None:
    """A non-provider GET routes through the reader-fallback ladder as HTML."""
    with patch(
        "wesearch.web.fetch_with_reader_fallback",
        return_value=(b"<html><body><p>Hi</p></body></html>", False),
    ):
        result = fetch_web("https://example.com")
    assert isinstance(result, FetchResult)
    # Equality, and the markup check below: a containment-only assertion here
    # passes when extraction is BYPASSED and the source returned verbatim,
    # since the fixture's own markup contains "Hi".
    assert result.text.strip() == "Hi"
    assert "<p>" not in result.text
    assert result.kind == _KIND_HTML
    assert result.url == "https://example.com"
    assert not result.truncated


def test_empty_extraction_does_not_fall_back_to_raw_markup() -> None:
    """An extractor that finds no text yields no text, not the page source.

    ``Extract`` permits an empty return, and a page with nothing to extract is
    a real answer. Substituting ``content`` handed a model a page of raw HTML
    in place of the text it asked for -- and on a JS-only page that is the
    common case, not the corner one.
    """
    with (
        patch(
            "wesearch.web.fetch_with_reader_fallback",
            return_value=(b"<html><body>raw fallback</body></html>", False),
        ),
        patch.dict("wesearch.web._EXTRACTORS", {"html2text": _extract_nothing}),
    ):
        result = fetch_web("https://example.com")
    assert result.text == ""


def test_post_html_is_extracted_like_a_get() -> None:
    """A POST that answers with HTML gets the same extraction a GET would.

    The method decided this for a while, so a POST returning a page handed back
    raw markup while the docstring promised ``policy.extractor``.
    """
    text = _extract_text(b"<html><body><p>Hello</p></body></html>", kind=_KIND_HTML)
    assert "Hello" in text
    assert "<p>" not in text


def test_json_body_is_returned_verbatim() -> None:
    """A JSON body skips extraction: it is already text, and HTML tools mangle it."""
    body = b'{"hello": "world"}'
    assert _extract_text(body, kind=_KIND_HTML) == body.decode()


def test_html_extraction_keeps_content_an_extractor_would_score_away() -> None:
    """The default extractor returns every text node, not just article prose.

    Article extractors score blocks and drop what does not look like prose, so
    a page whose substance is a short fragment loses it: a dictionary entry's
    pronunciation sits inside an audio-playback anchor, which reads as
    navigation. That output is non-empty, so the ``or content`` fallback below
    never fires -- the tool reports success on text with the answer missing,
    the worst failure shape available.

    Asserted on a fragment of that exact shape rather than on a live page, and
    on the anchor text rather than a whole-page comparison, because the anchor
    is what the scoring drops.
    """
    text = _extract_text(
        b'<html><body><div class="pron"><a href="/audio">'
        b"\xcb\x88\xc4\x81-j\xc9\x99nt</a></div></body></html>",
        kind=_KIND_HTML,
    )
    assert "\u02c8\u0101-j\u0259nt" in text


def test_extractor_policy_selects_the_named_extractor() -> None:
    """``PolicyParams.extractor`` picks the implementation; ``raw`` proves dispatch."""
    html = b"<html><body><p>Hi</p></body></html>"
    assert _extract_text(html, kind=_KIND_HTML, extractor="raw") == html.decode()


def test_fetch_web_reader_proxy_body_is_markdown() -> None:
    """When the fallback came from the reader proxy, kind is markdown (as-is)."""
    with patch(
        "wesearch.web.fetch_with_reader_fallback",
        return_value=(b"# md heading", True),
    ):
        result = fetch_web("https://example.com")
    assert result.kind == _KIND_MARKDOWN
    assert result.text == "# md heading"


def test_fetch_web_reddit_rss_renders_feed() -> None:
    """A Reddit ``rss`` payload is parsed as an Atom feed and rendered."""
    feed = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<feed xmlns="http://www.w3.org/2005/Atom">'
        b"<title>Post title : LocalLLaMA</title>"
        b"<entry><title>Post title</title>"
        b"<author><name>/u/op</name></author>"
        b"<content type='html'>&lt;p&gt;body&lt;/p&gt;</content></entry>"
        b"<entry><title>/u/commenter on Post title</title>"
        b"<content type='html'>&lt;p&gt;hi&lt;/p&gt;</content></entry>"
        b"</feed>"
    )
    with patch(
        "wesearch.web.reddit.fetch_reddit", return_value=(feed, "rss")
    ) as mock_reddit:
        result = fetch_web("https://reddit.com/r/foo/comments/abc")
    mock_reddit.assert_called_once()
    assert result.kind == _KIND_RSS
    assert "Post title" in result.text
    assert "/u/commenter" in result.text


def test_fetch_web_reddit_error_surfaces() -> None:
    """A FetchError from the Reddit provider propagates to the caller."""
    err = FetchError(url="https://x", status=403, headers={}, body=b"blocked")
    with (
        patch("wesearch.web.reddit.fetch_reddit", side_effect=err),
        pytest.raises(FetchError) as exc_info,
    ):
        fetch_web("https://reddit.com/r/foo/comments/abc")
    assert exc_info.value.status == 403


def test_fetch_web_google_news_routes_to_rss() -> None:
    """A Google News RSS payload is rendered as markdown."""
    feed = (
        b"<?xml version='1.0'?>"
        b"<rss><channel>"
        b"<title>Top stories - Google News</title>"
        b"<item><title>Lead</title><link>https://x</link></item>"
        b"</channel></rss>"
    )
    with patch(
        "wesearch.web.google_news.fetch_google_news",
        return_value=(feed, "rss"),
    ):
        result = fetch_web("https://news.google.com/")
    assert result.kind == _KIND_RSS
    assert "# Top stories - Google News" in result.text
    assert "## Lead" in result.text


def test_fetch_web_google_news_html_payload() -> None:
    """A Google News ``html`` payload goes through the HTML extractor."""
    with patch(
        "wesearch.web.google_news.fetch_google_news",
        return_value=(b"<html><body>article</body></html>", "html"),
    ):
        result = fetch_web("https://news.google.com/articles/xyz")
    assert result.kind == _KIND_HTML
    assert result.text.strip() == "article"
    assert "<body>" not in result.text


def test_fetch_web_x_routes_to_markdown() -> None:
    """An X URL is rendered through the reader proxy as markdown."""
    with patch("wesearch.web.x.fetch_x", return_value=b"# tweet text") as mock_x:
        result = fetch_web("https://x.com/user/status/1")
    mock_x.assert_called_once()
    assert result.kind == _KIND_MARKDOWN
    assert result.text == "# tweet text"


def test_fetch_web_post_uses_direct_fetch() -> None:
    """POST bypasses providers and the ladder, going straight to fetch."""
    with patch("wesearch.web.fetch", return_value=(b'{"ok": 1}', None)) as mock_fetch:
        result = fetch_web("https://api.example/x", method="POST", json_body={"a": 1})
    mock_fetch.assert_called_once()
    assert result.kind == _KIND_HTML
    # JSON-looking content is returned as-is (no trafilatura).
    assert '"ok"' in result.text


def test_fetch_web_json_body_skips_extraction() -> None:
    """JSON-looking GET content is returned verbatim (no trafilatura)."""
    with patch(
        "wesearch.web.fetch_with_reader_fallback",
        return_value=(b'{"hello": "world"}', False),
    ):
        result = fetch_web("https://api.example/json")
    assert '"hello"' in result.text


def test_fetch_web_truncates_to_max_chars() -> None:
    """``max_chars`` caps the returned text and sets the truncated flag."""
    with patch(
        "wesearch.web.fetch_with_reader_fallback",
        return_value=(b'{"x": "' + b"y" * 100 + b'"}', False),
    ):
        result = fetch_web("https://api.example/json", max_chars=20)
    assert len(result.text) == 20
    assert result.truncated


def test_fetch_web_returns_full_text_when_max_chars_unset() -> None:
    """Without ``max_chars``, fetch_web returns the whole body, uncapped.

    Capping is a caller's presentation policy: a library that silently truncates
    to a hardcoded ceiling breaks a caller (sagent) that appends its own
    truncation notice -- the pre-capped text is already at the caller's limit, so
    the caller's notice never fires.
    """
    big = "y" * 500_000
    with (
        patch("wesearch.web.fetch_with_reader_fallback", return_value=(b"h", False)),
        patch("wesearch.web._extract_text", return_value=big),
    ):
        result = fetch_web("https://e.example/")
    assert len(result.text) == 500_000
    assert result.truncated is False


def test_fetch_web_flags_challenge_returned_as_success() -> None:
    """A challenge page arriving as HTTP 200 is raised as a bot error."""
    challenge = (
        b"<html><head><title>Just a moment...</title></head>"
        b"<body>Security check required. Ray ID: abc123</body></html>"
    )
    with (
        patch(
            "wesearch.web.fetch_with_reader_fallback",
            return_value=(challenge, False),
        ),
        pytest.raises(CloudflareChallengeError),
    ):
        fetch_web("https://blocked.example")


def test_fetch_web_defaults_to_untrusted_on_the_direct_path() -> None:
    """The direct POST path treats a caller URL as untrusted unless told otherwise.

    Formerly the caller had to pass a resolver to get SSRF validation, so the
    default was unvalidated and the one caller that opted in lost the browser
    transports for it. Safety is the default now; ``internal`` is the opt-out.
    """
    captured: dict[str, object] = {}

    def fake_fetch(url: str, *, request: RequestParams) -> tuple[bytes, None]:
        del url
        captured["trust"] = request.policy.trust
        return b"{}", None

    with patch("wesearch.web.fetch", side_effect=fake_fetch):
        fetch_web("https://api.example/x", method="POST", json_body={"a": 1})
    assert captured["trust"] == "untrusted"


def test_fetch_web_forwards_policy_to_reader_fallback() -> None:
    """The whole policy -- transport AND trust -- reaches the fallback ladder.

    Each rung fetches a URL of its own (the proxy's), so a rung that dropped the
    policy would silently fetch it unvalidated.
    """
    policy = PolicyParams(transport="stdlib", trust="internal")
    with patch("wesearch.web.fetch_with_reader_fallback") as fallback:
        fallback.return_value = (b"<html></html>", False)
        fetch_web("https://example.com/x", policy=policy)
    assert fallback.call_args.kwargs["policy"] is policy


def test_fetch_body_post_maps_to_html_kind() -> None:
    """``_fetch_body`` POST returns raw bytes tagged ``_KIND_HTML``."""
    with patch("wesearch.web.fetch", return_value=(b"posted", None)):
        body, kind = _fetch_body(
            "https://api.example/x",
            method="POST",
            json_body={"a": 1},
            form_body=None,
        )
    assert body == b"posted"
    assert kind == _KIND_HTML


# ---------------------------------------------------------------------------
# _extract_text branches
# ---------------------------------------------------------------------------


def test_extract_text_markdown_kind_returns_as_is() -> None:
    """Markdown kind (reader-proxy output) skips extraction entirely."""
    md = b"# Title\n\nReader proxy returned this verbatim.\n"
    with patch.dict("wesearch.web._EXTRACTORS", {"trafilatura": _extract_wrong}):
        out = _extract_text(md, kind=_KIND_MARKDOWN)
    assert "# Title" in out
    assert "WRONG" not in out


# ---------------------------------------------------------------------------
# RSS / Atom formatters
# ---------------------------------------------------------------------------


def test_format_rss_renders_title_and_items() -> None:
    """Each ``<item>`` becomes a heading with link and meta line."""
    feed = (
        b"<?xml version='1.0'?>"
        b"<rss><channel>"
        b"<title>Top stories</title>"
        b"<item>"
        b"<title>Headline one</title>"
        b"<link>https://example.com/a</link>"
        b"<pubDate>Fri, 22 May 2026 00:00:00 GMT</pubDate>"
        b"<source url='https://nyt.example'>NYT</source>"
        b"</item>"
        b"</channel></rss>"
    )
    out = _format_rss(feed)
    assert "# Top stories" in out
    assert "## Headline one" in out
    assert out.count("https://example.com/a") >= 1
    assert "NYT" in out
    assert "Fri, 22 May 2026" in out


def test_format_rss_expands_google_news_cluster() -> None:
    """Sibling stories embedded in a description's ``<ol>`` become bullets."""
    cluster = (
        "<ol>"
        '<li><a href="https://lead.example">Lead headline</a>'
        '&nbsp;&nbsp;<font color="#6f6f6f">NYT</font></li>'
        '<li><a href="https://sib.example/1">Sibling one</a>'
        '&nbsp;&nbsp;<font color="#6f6f6f">CNN</font></li>'
        '<li><a href="https://sib.example/2">Sibling two</a>'
        '&nbsp;&nbsp;<font color="#6f6f6f">BBC</font></li>'
        "</ol>"
    )
    feed = (
        "<?xml version='1.0'?>"
        "<rss><channel>"
        "<title>Top stories</title>"
        "<item>"
        "<title>Lead headline</title>"
        "<link>https://lead.example</link>"
        f"<description><![CDATA[{cluster}]]></description>"
        "</item>"
        "</channel></rss>"
    ).encode()
    out = _format_rss(feed)
    # Lead is shown once at the top; siblings as bullets.
    assert out.count("Lead headline") == 1
    assert "- [Sibling one](https://sib.example/1) -- CNN" in out
    assert "- [Sibling two](https://sib.example/2) -- BBC" in out


def test_format_rss_renders_atom_entries() -> None:
    """Atom ``<entry>`` feeds render the same listing-friendly shape."""
    feed = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<feed xmlns="http://www.w3.org/2005/Atom">'
        b"<title>newest submissions : LocalLLaMA</title>"
        b"<entry>"
        b"<title>Granite 4.1 Architecture Changes?</title>"
        b"<author><name>/u/the-salami</name></author>"
        b"<updated>2026-05-28T17:44:55+00:00</updated>"
        b"<link href='https://www.reddit.com/r/LocalLLaMA/comments/abc/post/'/>"
        b"<content type='html'>&lt;p&gt;Why pure transformer?&lt;/p&gt;</content>"
        b"</entry>"
        b"</feed>"
    )
    out = _format_rss(feed)
    assert "# newest submissions : LocalLLaMA" in out
    assert "## Granite 4.1 Architecture Changes?" in out
    assert "/u/the-salami -- 2026-05-28T17:44:55+00:00" in out
    assert out.count("https://www.reddit.com/r/LocalLLaMA/comments/abc/post/") >= 1
    assert "Why pure transformer?" in out


def test_format_rss_invalid_xml_returns_raw_decoded() -> None:
    """Malformed XML degrades gracefully to a decoded byte slice."""
    out = _format_rss(b"not xml at all")
    assert "not xml" in out


def test_format_rss_rejects_entity_expansion_payload() -> None:
    """Defusedxml blocks billion-laughs / XXE payloads at parse time."""
    payload = (
        b"<?xml version='1.0'?>"
        b"<!DOCTYPE lolz ["
        b"<!ENTITY lol 'lol'>"
        b"<!ENTITY lol2 '&lol;&lol;&lol;&lol;&lol;'>"
        b"]>"
        b"<rss><channel><title>&lol2;</title></channel></rss>"
    )
    out = _format_rss(payload)
    # The parse fails (defusedxml refuses entity definitions); we fall through
    # to the decoded-bytes branch. The "lol" entity text must not appear
    # expanded.
    assert "lollollol" not in out


def test_format_rss_empty_channel_no_items() -> None:
    """A feed with no items still emits the title heading."""
    out = _format_rss(
        b"<?xml version='1.0'?><rss><channel><title>Empty feed</title></channel></rss>",
    )
    assert "# Empty feed" in out


def test_parse_rss_cluster_unescapes_entities() -> None:
    """HTML entities in titles and sources are decoded."""
    fragment = (
        '<li><a href="https://example.com/x">It&#8217;s here</a>'
        "&nbsp;&nbsp;<font>The &amp; Co.</font></li>"
    )
    entries = _parse_rss_cluster(fragment)
    assert len(entries) == 1
    title, link, source = entries[0]
    assert title == "It\u2019s here"
    assert link == "https://example.com/x"
    assert source == "The & Co."


def test_parse_rss_cluster_optional_source() -> None:
    """Missing ``<font>`` source yields an empty string, not a crash."""
    fragment = '<li><a href="https://example.com/y">Bare title</a></li>'
    entries = _parse_rss_cluster(fragment)
    assert entries == [("Bare title", "https://example.com/y", "")]


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
