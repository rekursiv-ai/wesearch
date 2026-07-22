"""HTML element extraction.

Usage::

    from wesearch.fetch import RequestParams, fetch
    from wesearch.scrape import get_element_content

    html = fetch(
        "https://example.com", request=RequestParams(timeout_sec=10)
    )[0].decode("utf-8")

    # With cookies:
    body, _ = fetch(
        "https://example.com",
        request=RequestParams(cookies={"session": "abc", "lang": "en"}),
    )
    html = body.decode("utf-8")

    title = get_element_content(html, "h1")
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import bs4
else:
    from wrapt import lazy_import

    bs4 = lazy_import("bs4")  # 140ms


__all__ = [
    "get_element_content",
]


def get_element_content(html: str, selector: str) -> str | None:
    """Extract text from the first element matching a CSS selector.

    Args:
      html: HTML source string.
      selector: CSS selector.

    Returns:
      text: Stripped text content, or ``None`` if no match.

    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    element = soup.select_one(selector)
    return element.get_text(strip=True) if element else None
