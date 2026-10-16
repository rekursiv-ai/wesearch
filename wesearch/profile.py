"""Cookies kept per public IP and host, shared across processes.

A site that sees the same cookies on every request sees one returning visitor;
one that sees none sees a new client each time, which reads as a bot. A new
public IP (a VPN roll) starts with none.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import quote

import fcntl
import json
import os
import tempfile
import threading
import time

from treekle import from_plain, json_loads

from wesearch.lib.userdirs import data_dir


if TYPE_CHECKING:
    from collections.abc import Generator


__all__ = [
    "Profile",
    "ProfileStore",
    "parse_set_cookie",
]


def parse_set_cookie(header_value: str) -> dict[str, str]:
    """Return the cookies a ``Set-Cookie`` value sets, without expired ones.

    Attributes (``Path``, ``Domain``, ...) are dropped.

    Args:
      header_value: One ``Set-Cookie`` value, or several joined by newlines.

    Returns:
      cookies: Cookie names and values.

    """
    cookies: dict[str, str] = {}
    for chunk in header_value.split("\n"):
        parts = chunk.split(";")
        head = parts[0].strip()
        if "=" not in head:
            continue
        name, value = head.split("=", 1)
        name = name.strip()
        if not name:
            continue
        if _is_deletion(parts[1:]):
            continue
        cookies[name] = value.strip()
    return cookies


@dataclass(frozen=True, slots=True, kw_only=True)
class Profile:
    """The cookies one host set for one public IP.

    Attributes:
      cookies: Cookie names and values.
      created: Unix time it was first saved; it expires ``ttl_sec`` later.

    """

    cookies: dict[str, str] = field(default_factory=dict)

    created: float = field(default_factory=time.time)


class ProfileStore:
    """Saves :class:`Profile` per public IP and host, one locked file each.

    Safe across threads and processes.

    Args:
      base_dir: Where the files go; ``None`` is wesearch's data directory.
      ttl_sec: Seconds a profile lives; an older one reads as absent.

    """

    def __init__(
        self,
        *,
        base_dir: Path | None = None,
        ttl_sec: float = 12 * 3600.0,
    ) -> None:
        self._base = (
            data_dir() / "rekursiv-ai" / "wesearch" if base_dir is None else base_dir
        )
        self._ttl_sec = ttl_sec
        self._lock = threading.Lock()

    def load(self, egress_ip: str, domain: str) -> Profile | None:
        """Return the profile for ``egress_ip`` and ``domain``, if live.

        Args:
          egress_ip: Public IP.
          domain: Host.

        Returns:
          profile: The profile; ``None`` when absent, expired, or corrupt.

        """
        path = self._path(egress_ip, domain)
        with self._locked(path):
            raw = self._read(path)
            if raw is None:
                return None
            profile = _try_decode(raw)
            if profile is None:
                return None
            if time.time() - profile.created > self._ttl_sec:
                path.unlink(missing_ok=True)
                return None
            return profile

    def save(self, egress_ip: str, domain: str, profile: Profile) -> None:
        """Save ``profile`` for ``egress_ip`` and ``domain``, replacing any.

        Args:
          egress_ip: Public IP.
          domain: Host.
          profile: The cookies to keep.

        """
        path = self._path(egress_ip, domain)
        with self._locked(path):
            self._write(path, _encode(profile))

    def discard(self, egress_ip: str, domain: str) -> None:
        """Delete the profile for ``egress_ip`` and ``domain``, if any.

        Args:
          egress_ip: Public IP.
          domain: Host.

        """
        path = self._path(egress_ip, domain)
        with self._locked(path):
            path.unlink(missing_ok=True)

    def update_cookies(
        self,
        egress_ip: str,
        domain: str,
        cookies: dict[str, str],
    ) -> None:
        """Set ``cookies`` in the saved profile, keeping its others; no-op if none.

        Args:
          egress_ip: Public IP.
          domain: Host.
          cookies: Cookie names and values that win.

        """
        if not cookies:
            return
        path = self._path(egress_ip, domain)
        with self._locked(path):
            raw = self._read(path)
            if raw is None:
                return
            profile = _try_decode(raw)
            if profile is None:
                return
            merged = {**profile.cookies, **cookies}
            self._write(
                path,
                _encode(
                    Profile(cookies=merged, created=profile.created),
                ),
            )

    def _path(self, egress_ip: str, domain: str) -> Path:
        # Quoted: an IPv6 address's colons are illegal in some filenames.
        key = (
            f"{quote(egress_ip).replace('/', '%2F')}|"
            f"{quote(domain).replace('/', '%2F')}"
        )
        return self._base / f"{key}.json"

    # A sidecar, not the key file: ``_write`` renames over the key file, so a
    # lock on it would bind to the replaced inode and serialize nothing.
    @contextmanager
    def _locked(self, path: Path) -> Generator[None]:
        """Hold the key's lock against every thread, store, and process."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            fd = os.open(
                path.with_name(f".{path.name}.lock"),
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
                0o600,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    def _read(self, path: Path, *, max_bytes: int = 1 << 20) -> bytes | None:
        """Read a key's bytes, or ``None`` when absent."""
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        except FileNotFoundError:
            return None
        try:
            data = os.read(fd, max_bytes)
        finally:
            os.close(fd)
        return data or None

    def _write(self, path: Path, data: bytes) -> None:
        """Atomically replace a key's file: write a unique temp sibling, then rename."""
        fd, tmp = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            _ = os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        Path(tmp).replace(path)


def parsedate_to_datetime_or_none(value: str) -> datetime | None:
    """Return an HTTP date as an aware datetime, or ``None`` if malformed.

    Args:
      value: An HTTP date, e.g. ``Wed, 21 Oct 2015 07:28:00 GMT``.

    Returns:
      when: The time, UTC when the date names no zone.

    """
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


# ``Max-Age`` wins over ``Expires`` (RFC 6265); a malformed value keeps the cookie.
def _is_deletion(attributes: list[str]) -> bool:
    """Whether cookie attributes mark it expired (past ``Max-Age`` or ``Expires``)."""
    for attr in attributes:
        key, _, value = attr.strip().partition("=")
        if key.strip().lower() == "max-age":
            try:
                return int(value.strip()) <= 0
            except ValueError:
                return False
    for attr in attributes:
        key, _, value = attr.strip().partition("=")
        if key.strip().lower() == "expires":
            expires = parsedate_to_datetime_or_none(value.strip())
            if expires is not None:
                return expires <= datetime.now(UTC)
    return False


# A corrupt file reads as absent and is replaced on the next save.
def _try_decode(raw: bytes) -> Profile | None:
    """Deserialize JSON bytes to a profile, or ``None`` if corrupt/malformed."""
    try:
        obj = json_loads(raw, dict[str, object])
        created = obj.get("created")
        if not isinstance(created, (int, float)):
            return None
        raw_cookies = obj.get("cookies")
        cookies = (
            {
                key: value
                for key, value in cast(dict[object, object], raw_cookies).items()
                if isinstance(key, str) and isinstance(value, str)
            }
            if isinstance(raw_cookies, dict)
            else {}
        )
        return Profile(
            cookies=from_plain(cookies, dict[str, str]),
            created=float(created),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _encode(profile: Profile) -> bytes:
    """Serialize a profile to JSON bytes."""
    return json.dumps(
        {
            "cookies": profile.cookies,
            "created": profile.created,
        },
    ).encode()
