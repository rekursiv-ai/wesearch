"""Article-body extraction via the ``trafilatura`` library.

Scores blocks and returns only what it judges to be the article, which is the
right answer for a news page or a blog post and the wrong one for any layout
whose substance is not article-shaped. See
:mod:`wesearch.fetch.extractor` for when that matters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import trafilatura
else:
    from wrapt import lazy_import

    trafilatura = lazy_import("trafilatura")


__all__ = ["extract_trafilatura"]


def extract_trafilatura(html: str, *, url: str = "") -> str:
    """Extract an HTML document's main article body as Markdown.

    Args:
      html: The page source.
      url: The page's own URL. Supplied to trafilatura so relative links
        resolve absolute and so its metadata header can name the source.

    Returns:
      text: The article text; empty when trafilatura finds no article.

    """
    return (
        trafilatura.extract(
            html,
            url=url or None,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            # Emits a YAML header (title, url, description, date) ahead of the
            # body. It recovers substance the body extraction drops: a page
            # whose content is a short fragment loses it to block scoring, but
            # the page states it verbatim in its meta description.
            with_metadata=True,
        )
        or ""
    )
