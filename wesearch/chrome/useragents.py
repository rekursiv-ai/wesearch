"""Real-browser Chrome User-Agent pools, filtered from ``intoli/user-agents``.

Two pools: ``chrome_desktop`` and ``chrome_android``. :func:`newest_chrome_major`
reads the newest Chrome version a pool records. :func:`refresh` re-downloads
and re-filters the dataset; run it as a maintenance step::

    python -m wesearch.chrome.useragents
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Final, Literal, Protocol, cast
from urllib import error, request

import gzip
import logging
import re
import stat
import tempfile

from treekle import ReadError, from_plain, json_loads

from wesearch.chrome.headers import (
    ChromePlatform,
    chrome_user_agent,
    user_agent_platform,
)


_CWD: Final = Path(__file__).resolve().parent


__all__ = [
    "UserAgentKind",
    "newest_chrome_major",
    "refresh",
    "refresh_all",
    "user_agent_pool",
]

type UserAgentKind = Literal["chrome_desktop", "chrome_android"]

logger = logging.getLogger(__name__)


@cache
def user_agent_pool(kind: UserAgentKind) -> tuple[str, ...]:
    """Return a cached User-Agent pool loaded from its package data file.

    Args:
        kind: User-Agent platform family.

    Returns:
        pool: User-Agent strings for the requested platform family.

    """
    return tuple(line for line in _pool_path(kind).read_text().splitlines() if line)


def newest_chrome_major(platform: ChromePlatform) -> int:
    """Return the newest Chrome major the pool records for ``platform``.

    Args:
      platform: The OS whose real-browser entries to consider.

    Returns:
      major: The highest ``Chrome/<major>`` among them.

    """
    kind: UserAgentKind = (
        "chrome_android" if platform == "Android" else "chrome_desktop"
    )
    return max(
        int(match.group(1))
        for ua in user_agent_pool(kind)
        if user_agent_platform(ua) == platform
        for match in re.finditer(r"Chrome/(\d+)", ua)
    )


def refresh(kind: UserAgentKind) -> None:
    """Rewrite one User-Agent pool from the intoli dataset.

    Args:
        kind: Pool to refresh.

    """
    selected = _select_user_agents(_download_records(), kind=kind)
    _replace_pool(kind, selected)
    user_agent_pool.cache_clear()
    logger.info("wrote %d user agents to %s", len(selected), _pool_path(kind))


def refresh_all() -> None:
    """Rewrite both User-Agent pools from one validated intoli snapshot."""
    records = _download_records()
    kinds: tuple[UserAgentKind, ...] = ("chrome_desktop", "chrome_android")
    selected_by_kind: dict[UserAgentKind, list[str]] = {
        kind: _select_user_agents(records, kind=kind) for kind in kinds
    }
    backups: dict[UserAgentKind, Path | None] = {}
    replaced: list[UserAgentKind] = []
    try:
        for kind in kinds:
            backups[kind] = _backup_pool(kind)
        try:
            for kind, selected in selected_by_kind.items():
                _replace_pool(kind, selected)
                replaced.append(kind)
                logger.info(
                    "wrote %d user agents to %s",
                    len(selected),
                    _pool_path(kind),
                )
        except BaseException:
            for kind in reversed(replaced):
                _restore_pool(kind, backups[kind])
            raise
    finally:
        for backup_path in backups.values():
            if backup_path is not None:
                backup_path.unlink(missing_ok=True)
        user_agent_pool.cache_clear()


def _pool_path(kind: UserAgentKind) -> Path:
    """Return the file holding ``kind``'s pool, alongside this module."""
    return _CWD / f"{kind}_useragents.txt"


def _select_user_agents(records: list[object], *, kind: UserAgentKind) -> list[str]:
    """Select safe, plain Chrome identities for one pool."""
    selected_set: set[str] = set()
    for raw_record in records:
        try:
            record = from_plain(raw_record, dict[str, object])
        except ReadError:
            continue
        if not record:
            continue
        ua = record.get("userAgent")
        device = record.get("deviceCategory")
        if not isinstance(ua, str):
            continue
        # A UA must occupy exactly one pool-file line.
        if not ua or ua != ua.strip() or not ua.isprintable():
            continue
        if kind == "chrome_android":
            keep = (
                device in ("mobile", "tablet")
                and "Android" in ua
                and "Mobile" in ua
                and _is_plain_chrome(ua)
                and "Android 10; K" not in ua
            )
        else:
            keep = (
                device == "desktop"
                and _is_plain_chrome(ua)
                and "Mobile" not in ua
                and "Android" not in ua
            )
        if keep:
            selected_set.add(ua)
    selected = sorted(selected_set)
    if len(selected) < 2:
        raise RuntimeError(
            f"refresh produced fewer than 2 distinct user agents for {kind}.",
        )
    return selected


def _is_plain_chrome(ua: str) -> bool:
    """Return whether a User-Agent is Chrome without a vendor wrapper."""
    return "Chrome/" in ua and not any(
        marker in ua
        for marker in (
            "CriOS/",
            "Edg/",
            "EdgA/",
            "EdgiOS/",
            "FBAN/",
            "FBAV/",
            "FB_IAB/",
            "HeadlessChrome/",
            "HuaweiBrowser/",
            "IABMV/",
            "OPR/",
            "OPT/",
            "SamsungBrowser/",
            "Vivaldi/",
            "YaBrowser/",
        )
    )


def _download_records() -> list[object]:
    """Download and parse the compressed intoli User-Agent dataset."""
    url = (
        "https://raw.githubusercontent.com/intoli/user-agents/"
        "main/src/user-agents.json.gz"
    )
    dataset_request = request.Request(
        url,
        headers={"User-Agent": _refresh_user_agent("chrome_desktop")},
    )
    attempt = 0
    while True:
        try:
            body = _read_response(dataset_request)
            break
        except error.HTTPError as http_error:
            retryable = http_error.code == 429 or 500 <= http_error.code < 600
            if not retryable or attempt == 2:
                raise
        except (TimeoutError, error.URLError):
            if attempt == 2:
                raise
        attempt += 1
    parsed = json_loads(gzip.decompress(body))
    if not isinstance(parsed, list):
        raise RuntimeError(  # noqa: TRY004 -- Dataset shape is a downloader contract error.
            "expected JSON array",
        )
    return from_plain(parsed, list[object])


def _refresh_user_agent(kind: UserAgentKind) -> str:
    """Return a coherent fixed identity for the maintenance download."""
    platform: ChromePlatform = "Android" if kind == "chrome_android" else "Linux"
    return chrome_user_agent(newest_chrome_major(platform), platform)


def _read_response(dataset_request: request.Request) -> bytes:
    """Perform one HTTPS request and return the body."""
    # The stub types urlopen's result as Any; the cast names the slice used here.
    response = cast(
        _ReadableBody,
        request.urlopen(dataset_request, timeout=30),  # noqa: S310 -- fixed HTTPS dataset URL.
    )
    with response:
        return response.read()


class _ReadableBody(Protocol):
    """The slice of ``HTTPResponse`` the downloader uses; a ``BytesIO`` also fits."""

    def read(self) -> bytes: ...

    def __enter__(self) -> object: ...

    def __exit__(self, *exc: object) -> object: ...


def _replace_pool(kind: UserAgentKind, selected: list[str]) -> None:
    """Atomically replace one pool file with validated User-Agents."""
    pool_path = _pool_path(kind)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=pool_path.parent,
        prefix=f".{pool_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_file.write("\n".join(selected) + "\n")
    temporary_path = Path(temporary_file.name)
    try:
        mode = stat.S_IMODE(pool_path.stat().st_mode) if pool_path.exists() else 0o644
        temporary_path.chmod(mode)
        temporary_path.replace(pool_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _backup_pool(kind: UserAgentKind) -> Path | None:
    """Copy one existing pool to a same-directory rollback file."""
    pool_path = _pool_path(kind)
    if not pool_path.exists():
        return None
    backup_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=pool_path.parent,
            prefix=f".{pool_path.name}.",
            suffix=".bak",
            delete=False,
        ) as backup_file:
            backup_path = Path(backup_file.name)
            backup_file.write(pool_path.read_bytes())
        backup_path.chmod(stat.S_IMODE(pool_path.stat().st_mode))
    except BaseException:
        if backup_path is not None:
            backup_path.unlink(missing_ok=True)
        raise
    return backup_path


def _restore_pool(kind: UserAgentKind, backup_path: Path | None) -> None:
    """Restore one pool from its rollback file or prior absence."""
    pool_path = _pool_path(kind)
    if backup_path is None:
        pool_path.unlink(missing_ok=True)
    else:
        backup_path.replace(pool_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    refresh_all()
