"""Pace requests to each site, and back off a site that blocked this IP.

Shared by every process on the host, per site and public IP. A block (a bot
challenge, a 429, or for some sites a dropped connection) locks the site; when
the lock ends one caller probes it while the rest wait. Success unlocks all;
another block doubles the lock.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import urlparse

import fcntl
import os
import random
import struct
import time

from wesearch.lib.userdirs import data_dir
from wesearch.types.errors import BotDetectionError, CooldownError, FetchError


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path


__all__ = ["SITE_POLICIES", "SitePolicy", "challenged_urls", "clear_host", "gated"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SitePolicy:
    """How one site is paced, and how a block on it is recovered from.

    Attributes:
      host: Exact hostname the policy covers.
      path_prefix: Narrows the policy to URLs under this path; the longest
        matching prefix wins.
      interval_sec: Mean seconds between requests across every process; ``0``
        leaves the site unpaced.
      jitter_sec: Extra uniform delay drawn when pacing binds, for a site that
        flags a fixed cadence.
      burst: Requests allowed back to back before pacing binds.
      cooldown_sec: Lock after the first block; each failed probe doubles it.
      max_cooldown_sec: Ceiling on the doubled lock.
      probe_lease_sec: How long one caller holds the probe before another may
        take it -- frees a probe whose process died mid-request.
      unreachable_blocks: Count a timeout or refused connection as a block --
        for a site that drops a flagged IP's packets rather than answering.

    """

    host: str
    path_prefix: str = "/"
    interval_sec: float = 0.0
    jitter_sec: float = 0.0
    burst: int = 1
    cooldown_sec: float = 30.0
    max_cooldown_sec: float = 3600.0
    probe_lease_sec: float = 120.0
    unreachable_blocks: bool = False


SITE_POLICIES: Final[tuple[SitePolicy, ...]] = (
    # Google flags an IP that searches faster than a person.
    SitePolicy(
        host="www.google.com",
        path_prefix="/search",
        interval_sec=5.0,
        burst=5,
    ),
    # Scholar flags a fixed cadence: uniform 4-10s held ~23 requests per IP before a
    # CAPTCHA where a fixed interval at the same mean burned in ~21. Its bans last
    # roughly a day or two, so a probe sooner only renews one.
    SitePolicy(
        host="scholar.google.com",
        path_prefix="/scholar",
        interval_sec=4.0,
        jitter_sec=6.0,
        cooldown_sec=48 * 3600.0,
        max_cooldown_sec=48 * 3600.0,
    ),
    # DuckDuckGo drops a flagged IP's packets, so a timeout IS its block.
    SitePolicy(
        host="html.duckduckgo.com",
        interval_sec=2.0,
        burst=5,
        unreachable_blocks=True,
    ),
    # Brave's plan allows 50 requests/s shared by every key.
    SitePolicy(
        host="api.search.brave.com",
        interval_sec=0.025,
        burst=40,
        cooldown_sec=2.0,
        max_cooldown_sec=60.0,
    ),
    # S2's authenticated tier is 1 request/s across every holder of the key.
    SitePolicy(
        host="api.semanticscholar.org",
        interval_sec=1.0,
        cooldown_sec=2.0,
        max_cooldown_sec=60.0,
    ),
    # OpenAlex budgets credits per day; a light pace smooths bursts.
    SitePolicy(
        host="api.openalex.org",
        interval_sec=0.1,
        cooldown_sec=2.0,
        max_cooldown_sec=60.0,
    ),
)


@contextmanager
def gated(
    url: str,
    *,
    egress: str,
    policies: Sequence[SitePolicy] = SITE_POLICIES,
) -> Generator[None]:
    """Hold one paced slot for ``url``'s site; record how the request ended.

    Args:
      url: The URL about to be requested.
      egress: This host's public IP, or ``""`` when not yet known.
      policies: Per-site pace and recovery; an unlisted site is unpaced and
        recovers with the :class:`SitePolicy` defaults.

    Yields:
      slot: ``None``; holding it is the permission to send one request.

    Raises:
      CooldownError: When the site is locked, or another caller is probing it.

    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host:
        yield
        return
    policy = _policy_for(host, parsed.path or "/", policies)
    path = (
        data_dir()
        / "rekursiv-ai"
        / "wesearch"
        / "ratelimit"
        / f"{host}:{egress}:{policy.path_prefix.strip('/').replace('/', '-')}_gate.lock"
    )
    tracked = policy.interval_sec > 0 or path.exists()
    if tracked:
        time.sleep(_admit(path, url, policy))
    try:
        yield
    except BotDetectionError:
        _trip(path, policy, challenge_url=url)
        raise
    except FetchError as e:
        blocked = e.status == 429 or (e.status == 0 and policy.unreachable_blocks)
        if blocked:
            _trip(path, policy)
        elif tracked:
            _release_probe(path)
        raise
    except TimeoutError:
        if policy.unreachable_blocks:
            _trip(path, policy)
        elif tracked:
            _release_probe(path)
        raise
    except BaseException:
        if tracked:
            _release_probe(path)
        raise
    if tracked:
        _reset(path, forget=policy.interval_sec <= 0)


def clear_host(host: str) -> int:
    """Unlock every site on ``host``, for every egress IP; keep its pace.

    Args:
      host: Exact hostname whose locks to lift.

    Returns:
      count: Number of gate files cleared.

    """
    base = data_dir() / "rekursiv-ai" / "wesearch" / "ratelimit"
    if not base.is_dir():
        return 0
    paths = [
        p
        for p in base.iterdir()
        if p.name.startswith(f"{host}:") and p.name.endswith("_gate.lock")
    ]
    for path in paths:
        _reset(path)
    return len(paths)


def challenged_urls() -> list[str]:
    """Return the URL that drew each standing challenge lock, oldest first.

    Only a bot challenge records one: a person can clear it in a browser, while
    a rate limit or a dropped connection lifts only with time.

    Returns:
      urls: One URL per challenge-locked site, deduplicated.

    """
    base = data_dir() / "rekursiv-ai" / "wesearch" / "ratelimit"
    if not base.is_dir():
        return []
    recorded = sorted(base.glob("*_gate.url"), key=lambda p: p.stat().st_mtime)
    return list(dict.fromkeys(p.read_text() for p in recorded))


def _policy_for(host: str, path: str, policies: Sequence[SitePolicy]) -> SitePolicy:
    """Return the longest-prefix policy for ``host``/``path``, else the default."""
    matches = [p for p in policies if p.host == host and path.startswith(p.path_prefix)]
    return max(matches, key=lambda p: len(p.path_prefix), default=SitePolicy(host=host))


def _admit(path: Path, url: str, policy: SitePolicy) -> float:
    """Admit the caller, make it the probe, or refuse it; return seconds to wait."""
    now = time.time()
    label = f"{policy.host}{policy.path_prefix.rstrip('/')}"
    with _state(path) as state:
        tokens, updated, locked_until, failures, probe_until = state
        if now < locked_until:
            raise CooldownError(
                f"{label} is cooling down for {locked_until - now:.0f}s after a "
                "block on this IP.",
                url=url,
            )
        if failures:
            if now < probe_until:
                raise CooldownError(
                    f"{label} is probing whether its block on this IP has lifted.",
                    url=url,
                )
            state[4] = now + policy.probe_lease_sec
        if policy.interval_sec <= 0:
            return 0.0
        rate = 1.0 / policy.interval_sec
        if not updated:
            tokens, updated = float(policy.burst), now
        tokens = min(float(policy.burst), tokens + (now - updated) * rate)
        wait = max(0.0, (1.0 - tokens) / rate)
        if wait:
            wait += random.SystemRandom().uniform(0.0, policy.jitter_sec)
        state[0:2] = [tokens + wait * rate - 1.0, now + wait]
        return wait


@contextmanager
def _state(path: Path) -> Generator[list[float]]:
    """Read, then rewrite if changed, the gate state under an exclusive ``flock``."""
    # Five doubles: (tokens, updated) pace the site; (locked_until, failures,
    # probe_until) recover it from a block. All zero means never used.
    layout = struct.Struct("<5d")
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        # A recovered site's file is deleted under the lock; one opened before
        # that is a dead inode, so reopen the live one.
        held = os.fstat(fd)
        try:
            live = path.stat()
        except FileNotFoundError:
            live = None
        if live is not None and (live.st_dev, live.st_ino) == (
            held.st_dev,
            held.st_ino,
        ):
            break
        os.close(fd)
    try:
        raw = os.read(fd, layout.size).ljust(layout.size, b"\0")
        state = list(cast(tuple[float, ...], layout.unpack(raw)))
        before = list(state)
        yield state
        if state != before:
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, layout.pack(*state))
    finally:
        os.close(fd)


def _trip(path: Path, policy: SitePolicy, *, challenge_url: str = "") -> None:
    """Relock for twice the previous lock, capped; the first block uses the base."""
    now = time.time()
    with _state(path) as state:
        failures = state[3] + 1
        until = now + min(
            policy.cooldown_sec * 2 ** (failures - 1),
            policy.max_cooldown_sec,
        )
        # The next probe may start when the lock ends, not before.
        state[2:5] = [until, failures, until]
        if challenge_url:
            _challenge_path(path).write_text(challenge_url)


def _challenge_path(path: Path) -> Path:
    """Return the file beside a gate's state holding the URL a challenge locked."""
    return path.with_suffix(".url")


def _release_probe(path: Path) -> None:
    """Free the probe for the next caller; the failure said nothing about a block."""
    with _state(path) as state:
        state[4] = 0.0


def _reset(path: Path, *, forget: bool = False) -> None:
    """Unlock every caller at once; the site answered. ``forget`` drops the file."""
    with _state(path) as state:
        state[2:5] = [0.0, 0.0, 0.0]
        _challenge_path(path).unlink(missing_ok=True)
        if forget:
            path.unlink()
