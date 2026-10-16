"""PDF download cascade for :mod:`wesearch.paper`.

:func:`download` tries each source in order and returns the first response
whose bytes start with the PDF magic:

1. arXiv direct (``https://arxiv.org/pdf/<id>``).
2. Open-access URL from S2 metadata (rate-gated through the shared S2 gate).
3. Source-only providers available in this build.

Nothing is written to disk; storing the PDF is the caller's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import logging

from treekle import MutablePlainTree

from wesearch.fetch import (
    FetchSession,
    RequestParams,
    RetryParams,
    fetch,
)
from wesearch.paper.errors import NotFoundError
from wesearch.paper.ids import s2_wire_id
from wesearch.paper.providers import s2
from wesearch.types.errors import FetchError


if TYPE_CHECKING:
    from wesearch.paper.types import IdType

__all__ = [
    "batch_oa_urls",
    "download",
    "oa_url_of",
]

logger = logging.getLogger(__name__)


def looks_like_pdf(
    content: bytes,
    *,
    min_pdf_bytes: int = 128,
    pdf_magic: bytes = b"%PDF-",
) -> bool:
    """Return whether ``content`` is a PDF, not an HTML page or captcha.

    Args:
      content: The downloaded bytes to test.
      min_pdf_bytes: Minimum length below which the content cannot be a real PDF.
      pdf_magic: The leading signature a PDF must start with.

    Returns:
      is_pdf: True when ``content`` is long enough and starts with ``pdf_magic``.

    """
    return len(content) >= min_pdf_bytes and content[: len(pdf_magic)] == pdf_magic


def oa_url_of(paper: dict[str, MutablePlainTree]) -> str | None:
    """Extract a non-empty ``openAccessPdf.url`` from an S2 paper record.

    Args:
      paper: S2 paper record; only ``openAccessPdf`` is read.

    Returns:
      url: The open-access PDF URL, or ``None`` when absent or empty.

    """
    oa = cast(dict[str, MutablePlainTree], paper.get("openAccessPdf") or {})
    url = oa.get("url")
    return url if isinstance(url, str) and url else None


def batch_oa_urls(
    wire_ids: list[str],
    *,
    session: FetchSession | None = None,
) -> list[str | None] | None:
    """Resolve open-access URLs for many ids in one batched S2 request.

    Args:
      wire_ids: S2 wire-format paper ids to resolve in a single batch.
      session: Connections to reuse; ``None`` uses the process's shared session.

    Returns:
      urls: Per-id OA URL in input order; a ``None`` element means S2 has no OA
        copy for that id. ``None`` overall when the batch call failed, so the
        caller can fall back to per-id lookups.

    """
    try:
        papers = s2.batch(wire_ids, "openAccessPdf", session=session)
    except Exception:  # noqa: BLE001 -- any backend failure permits the per-id fallback.
        return None
    return [oa_url_of(p) if p is not None else None for p in papers]


def download(
    kind: IdType,
    canonical: str,
    *,
    oa_url: str | None = None,
    oa_looked_up: bool = False,
    session: FetchSession | None = None,
) -> tuple[bytes, str]:
    """Download a paper's PDF from the first source that yields one.

    Args:
      kind: ``"doi"`` or ``"arxiv"``, as from ``normalize_id``.
      canonical: Bare identifier, as from ``normalize_id``.
      oa_url: Pre-resolved open-access URL from a batched lookup, if any.
      oa_looked_up: True when ``oa_url`` came from a completed lookup, so a
        ``None`` means "no open-access copy" and skips the per-id S2 query.
      session: Connections to reuse; ``None`` uses the process's shared session.

    Returns:
      pdf_bytes: The downloaded PDF content.
      source_label: Which source served it (e.g. ``"arxiv"``, ``"open_access"``).

    Raises:
      NotFoundError: When no enabled source returned a PDF.

    """
    if kind == "arxiv":
        body = _fetch_arxiv(canonical, session=session)
        if body is not None:
            return body, "arxiv"
    body = _fetch_open_access(
        kind,
        canonical,
        oa_url=oa_url,
        looked_up=oa_looked_up,
        session=session,
    )
    if body is not None:
        return body, "open_access"
    raise NotFoundError(f"No source returned a PDF for {kind}:{canonical}.")


def _fetch_arxiv(
    canonical: str,
    *,
    arxiv_pdf_base: str = "https://arxiv.org/pdf",
    session: FetchSession | None = None,
) -> bytes | None:
    """Fetch ``https://arxiv.org/pdf/<id>`` - no intermediary."""
    url = f"{arxiv_pdf_base}/{canonical}"
    try:
        return _download_pdf(url, session=session)
    except (FetchError, ValueError, OSError) as e:
        logger.debug("arXiv download failed: %s", e)
        return None


def _download_pdf(
    url: str,
    *,
    retries: int = 2,
    download_timeout_sec: float = 180.0,
    min_pdf_bytes: int = 128,
    session: FetchSession | None = None,
) -> bytes:
    """Download a URL, validate it looks like a PDF, return bytes."""
    body = fetch(
        url,
        request=RequestParams(
            retry=RetryParams(retries=retries, timeout_sec=download_timeout_sec),
        ),
        session=session,
    )
    return _validate_pdf(url, body, min_pdf_bytes=min_pdf_bytes)


def _validate_pdf(url: str, body: bytes, *, min_pdf_bytes: int = 128) -> bytes:
    """Return ``body`` if it looks like a PDF, else raise ``ValueError``."""
    if not looks_like_pdf(body, min_pdf_bytes=min_pdf_bytes):
        raise ValueError(
            f"GET {url} → non-PDF ({len(body)} bytes, prefix={body[:16]!r})",
        )
    return body


def _fetch_open_access(
    kind: IdType,
    canonical: str,
    *,
    oa_url: str | None,
    looked_up: bool,
    session: FetchSession | None = None,
) -> bytes | None:
    """Try an open-access PDF URL, looking it up via S2 when not yet resolved."""
    if oa_url is None and not looked_up:
        oa_url = _s2_oa_lookup(kind, canonical, session=session)
    if oa_url is None:
        return None
    try:
        return _download_pdf(oa_url, session=session)
    except (FetchError, ValueError, OSError) as e:
        logger.debug("OA download failed: %s", e)
        return None


def _s2_oa_lookup(
    kind: IdType,
    canonical: str,
    *,
    session: FetchSession | None = None,
) -> str | None:
    """Ask S2 for an ``openAccessPdf.url`` via the shared, rate-gated client."""
    try:
        data = s2.get(
            f"/paper/{s2_wire_id(kind, canonical)}",
            {"fields": "openAccessPdf"},
            session=session,
        )
    except Exception:  # noqa: BLE001 -- any lookup failure means no discoverable OA copy.
        return None
    return oa_url_of(data)
