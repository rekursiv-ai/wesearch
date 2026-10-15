"""Persistent transport routing learned by automatic web fetches."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import fcntl
import logging
import math
import os
import time

from wesearch.lib.userdirs import state_dir


_CWD: Final = Path(__file__).resolve().parent


logger = logging.getLogger(__name__)


__all__ = [
    "remember_zendriver_domain",
    "zendriver_domains",
]


def zendriver_domains(
    *,
    path: Path | None = None,
    ttl_sec: float = 7 * 24 * 3600.0,
) -> frozenset[str]:
    """Return domains observed to bot-wall the header transports.

    The recorded fact is that CURL was walled, not that the browser then
    succeeded: :func:`wesearch.fetch.fetch` persists the domain when curl
    raises, before the browser runs. That is deliberate -- a domain that walled
    curl will wall it again, so skipping straight to the browser is right even
    when the browser also fails. Reading this as "the browser works here" is
    what makes the entry look like poison rather than a routing fact.

    A learned entry expires after ``ttl_sec``: one block from a burned egress
    pinned the domain to the single contended Chrome forever, long after the
    wall had lifted. The bundled list never expires; a learned entry with no
    timestamp predates expiry and reads as already expired.

    Args:
      path: Domain-list path. Defaults to the bundled list plus the per-user
        ``zendriver-domains.txt`` under the wesearch state directory.
      ttl_sec: Seconds a learned entry routes to Zendriver after it was written.

    Returns:
      domains: Normalized domains currently routed directly to Zendriver.

    """
    cutoff = time.time() - ttl_sec
    if path is not None:
        return _read_domains(path, cutoff=cutoff)
    return _read_domains(_CWD / "zendriver-domains.txt", cutoff=-math.inf) | (
        _read_domains(
            state_dir() / "rekursiv-ai" / "wesearch" / "zendriver-domains.txt",
            cutoff=cutoff,
        )
    )


def remember_zendriver_domain(
    domain: str,
    *,
    path: Path | None = None,
    never_route: frozenset[str] = frozenset({"r.jina.ai"}),
) -> None:
    """Atomically add ``domain`` to the cross-process Zendriver domain list.

    Each line is ``domain<TAB>unix-time``; a repeat refreshes the time, so a
    domain that keeps walling curl keeps its route.

    Args:
      domain: DNS hostname observed to bot-wall the header transports.
      path: Domain-list path. Defaults to the per-user
        ``zendriver-domains.txt`` under the wesearch state directory.
      never_route: Hosts never learned. The reader proxy is itself the fallback
        for a walled page; routing it through the browser makes the fallback
        depend on the resource it exists to avoid.

    Raises:
      ValueError: If ``domain`` is empty or not one line.

    """
    normalized = domain.strip().casefold()
    if not normalized or "\n" in normalized or "\r" in normalized:
        raise ValueError(f"Invalid Zendriver domain: {domain!r}.")
    if normalized in never_route:
        return
    target = (
        state_dir() / "rekursiv-ai" / "wesearch" / "zendriver-domains.txt"
        if path is None
        else path
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor = os.open(
        target,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
        0o600,
    )
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_EX)
        # A corrupt cache is DISCARDED and rewritten rather than raising: the
        # read path degrades to empty on non-UTF-8, so raising here would leave
        # one bad byte permanently blocking every future write with no way back
        # except deleting the file by hand.
        #
        # Discarded whole, not per-line: decoding with errors="replace" turned
        # bad bytes into U+FFFD and then PERSISTED those as domains, so the file
        # accumulated junk the read path would have rejected outright.
        try:
            existing = _read_all(file_descriptor).decode()
        except UnicodeDecodeError:
            logger.warning(
                "Discarding undecodable Zendriver domain list at %s.",
                target,
            )
            existing = ""
        entries = dict(
            entry for line in existing.splitlines() if (entry := _parse(line))
        )
        entries[normalized] = time.time()
        payload = "".join(
            f"{name}\t{stamp:.0f}\n" for name, stamp in sorted(entries.items())
        ).encode()
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        os.ftruncate(file_descriptor, 0)
        _write_all(file_descriptor, payload)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _parse(line: str) -> tuple[str, float] | None:
    """Return ``(domain, written_at)`` for one list line, or ``None`` if blank."""
    name, _, stamp = line.partition("\t")
    normalized = name.strip().casefold()
    if not normalized:
        return None
    try:
        return normalized, float(stamp)
    except ValueError:
        return normalized, 0.0


def _read_all(file_descriptor: int) -> bytes:
    """Read one open file descriptor to EOF."""
    chunks: list[bytes] = []
    while chunk := os.read(file_descriptor, 1 << 20):
        chunks.append(chunk)
    return b"".join(chunks)


def _write_all(file_descriptor: int, data: bytes) -> None:
    """Write ``data`` fully, honoring short writes."""
    view = memoryview(data)
    while view:
        view = view[os.write(file_descriptor, view) :]


# Optional: ``_read_domains`` returns an empty set when the file is absent, so a
# checkout without it simply starts with no bundled defaults.
# The list is an optional, rebuildable cache on the per-fetch hot path, so a missing,
# permission-denied, or corrupt (non-UTF-8) file degrades to no learned routing rather
# than aborting every automatic fetch.
def _read_domains(path: Path, *, cutoff: float) -> frozenset[str]:
    """Read one locked domain list, returning empty when absent or unreadable."""
    try:
        file_descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except FileNotFoundError:
        return frozenset[str]()
    except OSError:
        logger.warning("Ignoring unreadable Zendriver domain list at %s.", path)
        return frozenset[str]()
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_SH)
        raw = _read_all(file_descriptor)
    except OSError:
        # The docstring's promise covers the whole read, not just the open: a
        # lock or read failure on an optional rebuildable cache must not abort
        # every automatic fetch, and this runs on the per-fetch hot path.
        logger.warning("Ignoring unreadable Zendriver domain list at %s.", path)
        return frozenset[str]()
    finally:
        os.close(file_descriptor)
    try:
        text = raw.decode()
    except UnicodeDecodeError:
        logger.warning("Ignoring undecodable Zendriver domain list at %s.", path)
        return frozenset[str]()
    return frozenset(
        name
        for line in text.splitlines()
        if (entry := _parse(line)) is not None
        for name, stamp in (entry,)
        if stamp >= cutoff
    )
