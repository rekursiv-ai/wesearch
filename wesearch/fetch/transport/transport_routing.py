"""Persistent transport routing learned by automatic web fetches."""

from __future__ import annotations

from pathlib import Path

import fcntl
import logging
import os

from wesearch.lib.userdirs import state_dir


logger = logging.getLogger(__name__)


__all__ = [
    "remember_zendriver_domain",
    "zendriver_domains",
]


def _normalize(line: str) -> str:
    """Return the canonical (stripped, casefolded) form of one domain line."""
    return line.strip().casefold()


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


def _bundled_domains_path() -> Path:
    """Return the read-only domain defaults shipped alongside this module.

    Optional: ``_read_domains`` returns an empty set when the file is absent, so
    a checkout without it simply starts with no bundled defaults.
    """
    return Path(__file__).parent / "zendriver-domains.txt"


def _read_domains(path: Path) -> frozenset[str]:
    """Read one locked domain list, returning empty when absent or unreadable.

    The list is an optional, rebuildable cache on the per-fetch hot path, so a
    missing, permission-denied, or corrupt (non-UTF-8) file degrades to no
    learned routing rather than aborting every automatic fetch.
    """
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
        normalized for line in text.splitlines() if (normalized := _normalize(line))
    )


def zendriver_domains(*, path: Path | None = None) -> frozenset[str]:
    """Return domains observed to bot-wall the header transports.

    The recorded fact is that CURL was walled, not that the browser then
    succeeded: :func:`wesearch.fetch.fetch` persists the domain when curl
    raises, before the browser runs. That is deliberate -- a domain that walled
    curl will wall it again, so skipping straight to the browser is right even
    when the browser also fails. Reading this as "the browser works here" is
    what makes the entry look like poison rather than a routing fact.

    Args:
      path: Domain-list path. Defaults to the bundled list plus the per-user
        ``zendriver-domains.txt`` under the wesearch state directory.

    Returns:
      domains: Normalized domains currently routed directly to Zendriver.

    """
    if path is not None:
        return _read_domains(path)
    return _read_domains(_bundled_domains_path()) | _read_domains(
        state_dir() / "rekursiv-ai" / "wesearch" / "zendriver-domains.txt"
    )


def remember_zendriver_domain(domain: str, *, path: Path | None = None) -> None:
    """Atomically add ``domain`` to the cross-process Zendriver domain list.

    Args:
      domain: DNS hostname observed to bot-wall the header transports.
      path: Domain-list path. Defaults to the per-user
        ``zendriver-domains.txt`` under the wesearch state directory.

    Raises:
      ValueError: If ``domain`` is empty or not one line.

    """
    normalized = domain.strip().casefold()
    if not normalized or "\n" in normalized or "\r" in normalized:
        raise ValueError(f"Invalid Zendriver domain: {domain!r}.")
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
                "Discarding undecodable Zendriver domain list at %s.", target
            )
            existing = ""
        domains = {
            value for line in existing.splitlines() if (value := _normalize(line))
        }
        if normalized in domains:
            return
        domains.add(normalized)
        payload = "".join(f"{value}\n" for value in sorted(domains)).encode()
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        os.ftruncate(file_descriptor, 0)
        _write_all(file_descriptor, payload)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
