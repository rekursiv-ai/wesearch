"""Real-browser User-Agent pools, one per impersonated browser kind.

Two maintained pools of real User-Agents, filtered from the ``intoli/user-agents``
dataset: ``chrome_desktop`` (desktop Chrome) and ``chrome_android`` (Android
Chrome). Each kind identifies a platform family and maps to its corresponding
curl_cffi impersonation target (:func:`impersonate_target`). Pool entries retain
real-world browser versions and device models; the mapping does not promise an
exact version match with curl_cffi's TLS identity.

The hot path (:func:`user_agent_pool`, :func:`draw_user_agent`,
:func:`impersonate_target`) only reads local pool files. :func:`refresh`
re-downloads and re-filters the dataset; run it as a maintenance step::

    python -m wesearch.chrome.useragents
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal, cast

import gzip
import json
import logging
import random
import stat
import tempfile
import urllib.error
import urllib.request

from wesearch.chrome.headers import (
    chrome_user_agent,
    impersonate_version_platform,
)


__all__ = [
    "UserAgentKind",
    "draw_user_agent",
    "impersonate_target",
    "kind_for_impersonate",
    "refresh",
    "refresh_all",
    "user_agent_pool",
]

type UserAgentKind = Literal["chrome_desktop", "chrome_android"]

logger = logging.getLogger(__name__)

_RNG = random.SystemRandom()


def impersonate_target(kind: UserAgentKind) -> str:
    """Return the curl_cffi target for a User-Agent platform family.

    Args:
        kind: User-Agent platform family.

    Returns:
        target: Corresponding curl_cffi impersonation target.

    """
    return "chrome_android" if kind == "chrome_android" else "chrome"


def kind_for_impersonate(impersonate: str) -> UserAgentKind:
    """Return the platform-family pool for a curl_cffi target.

    The inverse of :func:`impersonate_target`, kept beside it as the ONE source
    of truth for the impersonate<->kind mapping (an unknown target degrades to
    the desktop pool, matching :func:`impersonate_version_platform`).

    Args:
        impersonate: curl_cffi impersonation target.

    Returns:
        kind: Corresponding User-Agent platform family.

    """
    return "chrome_android" if impersonate == "chrome_android" else "chrome_desktop"


@cache
def user_agent_pool(kind: UserAgentKind) -> tuple[str, ...]:
    """Return a cached User-Agent pool loaded from its package data file.

    Args:
        kind: User-Agent platform family.

    Returns:
        pool: User-Agent strings for the requested platform family.

    """
    return tuple(line for line in _pool_path(kind).read_text().splitlines() if line)


def draw_user_agent(kind: UserAgentKind) -> str:
    """Draw a random real User-Agent from one platform-family pool.

    Args:
        kind: User-Agent platform family.

    Returns:
        user_agent: Random entry from the requested pool.

    """
    return _RNG.choice(user_agent_pool(kind))


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
                    "wrote %d user agents to %s", len(selected), _pool_path(kind)
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


def _download_records() -> list[object]:
    """Download and parse the compressed intoli User-Agent dataset."""
    url = (
        "https://raw.githubusercontent.com/intoli/user-agents/"
        "main/src/user-agents.json.gz"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _refresh_user_agent("chrome_desktop")},
    )
    body = b""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(  # noqa: S310 -- fixed HTTPS dataset URL.
                request, timeout=30
            ) as response:
                body = response.read()
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 and not 500 <= error.code < 600:
                raise
            if attempt == 2:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
    parsed: object = json.loads(gzip.decompress(body))
    if not isinstance(parsed, list):
        raise RuntimeError(f"expected JSON array from {url}; upstream shape changed?")  # noqa: TRY004
    return cast("list[object]", parsed)


def _select_user_agents(records: list[object], *, kind: UserAgentKind) -> list[str]:
    """Select safe, plain Chrome identities for one pool."""
    keep = _is_android_chrome if kind == "chrome_android" else _is_desktop_chrome
    selected_set: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        record = cast("dict[str, object]", record)
        ua = record.get("userAgent")
        device = record.get("deviceCategory")
        if (
            isinstance(ua, str)
            and _is_safe_user_agent(ua)
            and keep(ua, device if isinstance(device, str) else "")
        ):
            selected_set.add(ua)
    selected = sorted(selected_set)
    if len(selected) < 2:
        raise RuntimeError(
            f"refresh produced fewer than 2 distinct user agents for {kind}."
        )
    return selected


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


def _is_safe_user_agent(ua: str) -> bool:
    """Return whether a User-Agent can occupy exactly one pool-file line."""
    return bool(ua) and ua == ua.strip() and ua.isprintable()


def _is_plain_chrome(ua: str) -> bool:
    """Return whether a User-Agent is Chrome without a vendor wrapper."""
    return "Chrome/" in ua and not any(
        marker in ua
        for marker in (
            "CriOS/",
            "Edg/",
            "EdgA/",
            "EdgiOS/",
            "HeadlessChrome/",
            "HuaweiBrowser/",
            "OPR/",
            "OPT/",
            "SamsungBrowser/",
            "Vivaldi/",
            "YaBrowser/",
        )
    )


def _is_desktop_chrome(ua: str, device: str) -> bool:
    """A plain desktop Chrome UA (not mobile, not an Edge/Opera/Samsung variant)."""
    return (
        device == "desktop"
        and _is_plain_chrome(ua)
        and "Mobile" not in ua
        and "Android" not in ua
    )


def _is_android_chrome(ua: str, device: str) -> bool:
    """A plain mobile Android Chrome UA."""
    return (
        device in ("mobile", "tablet")
        and "Android" in ua
        and "Mobile" in ua
        and _is_plain_chrome(ua)
        and "Android 10; K" not in ua
    )


def _pool_path(kind: UserAgentKind) -> Path:
    """The file holding ``kind``'s pool, alongside this module."""
    return Path(__file__).with_name(f"{kind}_useragents.txt")


def _refresh_user_agent(kind: UserAgentKind) -> str:
    """Return a coherent fixed identity for the maintenance download."""
    major, platform = impersonate_version_platform(impersonate_target(kind))
    return chrome_user_agent(major, platform)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    refresh_all()
