"""Pace requests to each key, and back off a key that was blocked.

A ``Gate`` keeps, per key, a pace and a lock. A key names what is paced: a site
and public IP for a fetch, or an account for a model provider. A block (a bot
challenge, a 429, or for some sites a dropped connection) locks the key; when
the lock ends one caller probes it while the rest wait. Success unlocks all;
another block doubles the lock. A server that says when to come back holds the
key until then, with no probe.

``FileGate``, the default, shares its state with every process on the host
through one locked file per key. A caller may supply another ``Gate``, as one a
daemon serves, so long as every caller on the host reaches the same state.

Typical usage example:

  with gated(url, egress=ip, gate=session.gate):
      ...  # Send one request.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Protocol
from urllib.parse import quote, unquote, urlparse

import asyncio
import fcntl
import os
import random
import time

from treekle import json_dumps, json_loads

from wesearch.lib.userdirs import data_dir
from wesearch.types.errors import BotDetectionError, CooldownError, FetchError


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Sequence
    from pathlib import Path


__all__ = [
    "SITE_POLICIES",
    "Admission",
    "FileGate",
    "Gate",
    "GatePolicy",
    "GateState",
    "admitted",
    "challenged_urls",
    "clear_host",
    "gated",
    "waited",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class GatePolicy:
    """How one key is paced, and how a block on it is recovered from.

    Attributes:
      host: Exact hostname whose URLs ``gated`` paces by this policy; ``""`` for
        a key that is no site, as an account, which no URL matches.
      path_prefix: Narrows the policy to URLs under this path; the longest
        matching prefix wins.
      interval_sec: Mean seconds between requests across every process; ``0``
        leaves the key unpaced.
      jitter_sec: Extra uniform delay drawn when pacing binds, for a site that
        flags a fixed cadence.
      burst: Requests allowed back to back before pacing binds.
      cooldown_sec: Lock after the first block; each failed probe doubles it.
      max_cooldown_sec: Ceiling on the doubled lock.
      probe_lease_sec: How long one caller holds the probe before another may
        take it -- frees a probe whose process died mid-request.
      unreachable_blocks: Count a timeout or refused connection as a block --
        for a site that drops a flagged IP's packets rather than answering.
      max_wait_sec: Longest lock or probe a caller sits out before sending;
        one longer refuses it at once.

    """

    host: str = ""
    path_prefix: str = "/"
    interval_sec: float = 0.0
    jitter_sec: float = 0.0
    burst: int = 1
    cooldown_sec: float = 30.0
    max_cooldown_sec: float = 3600.0
    probe_lease_sec: float = 120.0
    unreachable_blocks: bool = False
    max_wait_sec: float = 0.0


SITE_POLICIES: Final[tuple[GatePolicy, ...]] = (
    # Google flags an IP that searches faster than a person.
    GatePolicy(
        host="www.google.com",
        path_prefix="/search",
        interval_sec=5.0,
        burst=5,
    ),
    # A result link answers a client that follows more than ~3 quickly with a
    # 3-second refresh page, and a 400 if followed sooner (measured 2026-10-30).
    GatePolicy(
        host="www.google.com",
        path_prefix="/goto",
        interval_sec=3.5,
        burst=3,
    ),
    # Scholar flags a fixed cadence: uniform 4-10s held ~23 requests per IP before a
    # CAPTCHA where a fixed interval at the same mean burned in ~21. Its bans last
    # roughly a day or two, so a probe sooner only renews one.
    GatePolicy(
        host="scholar.google.com",
        path_prefix="/scholar",
        interval_sec=4.0,
        jitter_sec=6.0,
        cooldown_sec=48 * 3600.0,
        max_cooldown_sec=48 * 3600.0,
    ),
    # DuckDuckGo drops a flagged IP's packets, so a timeout IS its block.
    GatePolicy(
        host="html.duckduckgo.com",
        interval_sec=2.0,
        burst=5,
        unreachable_blocks=True,
    ),
    # Brave's plan allows 50 requests/s shared by every key.
    GatePolicy(
        host="api.search.brave.com",
        interval_sec=0.025,
        burst=40,
        cooldown_sec=2.0,
        max_cooldown_sec=60.0,
    ),
    # S2's authenticated tier is 1 request/s across every holder of the key.
    GatePolicy(
        host="api.semanticscholar.org",
        interval_sec=1.0,
        cooldown_sec=2.0,
        max_cooldown_sec=60.0,
    ),
    # OpenAlex budgets credits per day; a light pace smooths bursts.
    GatePolicy(
        host="api.openalex.org",
        interval_sec=0.1,
        cooldown_sec=2.0,
        max_cooldown_sec=60.0,
    ),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class GateState:
    """What a gate keeps for one key; all zero means never used.

    Attributes:
      tokens: Requests the pace allows, as of ``updated``.
      updated: When ``tokens`` was counted, in seconds since the epoch.
      locked_until: When the lock a block drew ends; a success lifts it.
      held_until: When a hold a server named ends; only its time lifts it.
      failures: Blocks in a row; past zero, the first caller back probes.
      probe_until: When the current probe's lease ends.
      challenge_url: The URL whose bot challenge drew the lock; ``""`` for none.
      challenged_at: When ``challenge_url`` was recorded.

    """

    tokens: float = 0.0
    updated: float = 0.0
    locked_until: float = 0.0
    held_until: float = 0.0
    failures: int = 0
    probe_until: float = 0.0
    challenge_url: str = ""
    challenged_at: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class Admission:
    """A gate's answer to a caller about to send.

    Attributes:
      wait_sec: Seconds to wait first.
      admitted: Send after ``wait_sec``; else ask again after it.
      reason: Why the caller was not admitted, as ``cooling down for 30s``.

    """

    wait_sec: float = 0.0
    admitted: bool = True
    reason: str = ""


class Gate(Protocol):
    """Paces and locks keys for every caller that shares it."""

    def admit(self, key: str, policy: GatePolicy) -> Admission:
        """Admit the caller, make it the probe, or say how long to wait."""
        ...

    async def admitted(self, key: str, policy: GatePolicy) -> Admission:
        """Wait until the caller is admitted to ``key``, seeing a lock set meanwhile.

        Args:
          key: What is paced.
          policy: How it is paced, and the longest lock waited out.

        Returns:
          admission: ``Admission()`` once admitted; else, unwaited, the refusal
            of a lock or probe longer than ``policy.max_wait_sec``.

        """
        ...

    def trip(self, key: str, policy: GatePolicy, *, challenge_url: str = "") -> None:
        """Lock ``key`` for twice the previous lock, capped; the first uses the base."""
        ...

    def hold(self, key: str, *, until: float) -> None:
        """Lock ``key`` until ``until``, a time the server named; no probe follows."""
        ...

    def release_probe(self, key: str) -> None:
        """Free the probe; the failure said nothing about a block."""
        ...

    def reset(self, key: str, *, forget: bool = False) -> None:
        """Lift the lock a block drew on ``key``; a server's hold stays to its time."""
        ...

    def states(self) -> dict[str, GateState]:
        """Return every key the gate keeps, and its state."""
        ...


class FileGate:
    """A ``Gate`` shared by every process on the host, one locked file per key.

    Args:
      base_dir: Where the files go; ``None`` is wesearch's data directory, read
        at each call.

    """

    def __init__(self, *, base_dir: Path | None = None) -> None:
        self._base = base_dir

    def admit(self, key: str, policy: GatePolicy) -> Admission:
        """Admit the caller, make it the probe, or say how long to wait.

        An unpaced key with no state is admitted without a file: else every
        request to an untroubled site would pay for a locked read and write.

        Args:
          key: What is paced.
          policy: How it is paced and recovered.

        Returns:
          admission: The wait, and whether to send after it.

        """
        path = self._path(key)
        if policy.interval_sec <= 0 and not path.exists():
            return Admission()
        with _state(path) as box:
            admission, box[0] = _admit(box[0], policy, now=time.time())
        return admission

    async def admitted(self, key: str, policy: GatePolicy) -> Admission:
        """Wait until the caller is admitted to ``key``.

        Files wake no one: a lock another caller sets during a wait is seen when
        the wait ends, and the gate is asked again.

        Args:
          key: What is paced.
          policy: How it is paced, and the longest lock waited out.

        Returns:
          admission: ``Admission()`` once admitted; else, unwaited, the refusal
            of a lock or probe longer than ``policy.max_wait_sec``.

        """

        async def admit() -> Admission:
            return await asyncio.to_thread(self.admit, key, policy)

        return await waited(admit, policy, wait=_slept)

    def trip(self, key: str, policy: GatePolicy, *, challenge_url: str = "") -> None:
        """Lock ``key`` for twice the previous lock, capped; the first uses the base.

        Args:
          key: What is paced.
          policy: The base lock and its cap.
          challenge_url: The URL whose bot challenge drew the lock; ``""`` for none.

        """
        now = time.time()
        with _state(self._path(key)) as box:
            failures = box[0].failures + 1
            lock_sec = min(
                policy.cooldown_sec * 2.0 ** (failures - 1),
                policy.max_cooldown_sec,
            )
            until = now + lock_sec
            # The next probe may start when the lock ends, not before.
            box[0] = replace(
                box[0],
                locked_until=until,
                failures=failures,
                probe_until=until,
            )
            if challenge_url:
                box[0] = replace(box[0], challenge_url=challenge_url, challenged_at=now)

    def hold(self, key: str, *, until: float) -> None:
        """Lock ``key`` until ``until``, a time the server named; no probe follows."""
        with _state(self._path(key)) as box:
            box[0] = replace(box[0], held_until=max(box[0].held_until, until))

    def release_probe(self, key: str) -> None:
        """Free the probe; the failure said nothing about a block."""
        path = self._path(key)
        if not path.exists():
            return
        with _state(path) as box:
            box[0] = replace(box[0], probe_until=0.0)

    def reset(self, key: str, *, forget: bool = False) -> None:
        """Lift the lock a block drew on ``key``, and its backoff; a hold stays.

        A success resets: a reply that also said its window is spent holds the
        key past the success, until the time it named.

        Args:
          key: What is paced.
          forget: Also drop its pace, and its file once no hold is left.

        """
        path = self._path(key)
        if not path.exists():
            return
        with _state(path) as box:
            held = box[0].held_until if box[0].held_until > time.time() else 0.0
            box[0] = GateState(
                tokens=box[0].tokens,
                updated=box[0].updated,
                held_until=held,
            )
            if forget and not held:
                path.unlink()

    def states(self) -> dict[str, GateState]:
        """Return every key the gate keeps, and its state.

        Returns:
          states: Each key's state, as last written.

        """
        base = self._dir()
        if not base.is_dir():
            return {}
        found: dict[str, GateState] = {}
        for path in base.glob("*.gate"):
            try:
                found[unquote(path.stem)] = _read_shared(path)
            except FileNotFoundError:
                continue
        return found

    def _dir(self) -> Path:
        if self._base is not None:
            return self._base
        return data_dir() / "rekursiv-ai" / "wesearch" / "ratelimit"

    def _path(self, key: str) -> Path:
        return self._dir() / f"{quote(key, safe='')}.gate"


@contextmanager
def gated(
    url: str,
    *,
    egress: str,
    gate: Gate,
    policies: Sequence[GatePolicy] = SITE_POLICIES,
) -> Generator[None]:
    """Hold one paced slot for ``url``'s site; record how the request ended.

    Args:
      url: The URL about to be requested.
      egress: This host's public IP, or ``""`` when not yet known.
      gate: The state every caller on the host shares.
      policies: Per-site pace and recovery; an unlisted site is unpaced and
        recovers with the :class:`GatePolicy` defaults.

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
    policy = _site_policy(host, parsed.path or "/", policies)
    key = f"{host}:{egress}:{policy.path_prefix.strip('/').replace('/', '-')}"
    label = f"{policy.host}{policy.path_prefix.rstrip('/')}"
    admitted(
        gate,
        key,
        policy,
        refuse=lambda reason: CooldownError(f"{label} is {reason}.", url=url),
    )
    try:
        yield
    except BotDetectionError as block:
        if block.blocks_ip:
            gate.trip(key, policy, challenge_url=url)
        else:
            gate.release_probe(key)
        raise
    except FetchError as e:
        if e.status == 429 or (e.status == 0 and policy.unreachable_blocks):
            gate.trip(key, policy)
        else:
            gate.release_probe(key)
        raise
    except TimeoutError:
        if policy.unreachable_blocks:
            gate.trip(key, policy)
        else:
            gate.release_probe(key)
        raise
    except BaseException:
        gate.release_probe(key)
        raise
    gate.reset(key, forget=policy.interval_sec <= 0)


def admitted(
    gate: Gate,
    key: str,
    policy: GatePolicy,
    *,
    refuse: Callable[[str], Exception],
) -> None:
    """Wait until ``gate`` admits the caller to ``key``.

    A lock or probe no longer than ``policy.max_wait_sec`` is sat out, then the
    gate asked again, so a lock set meanwhile is seen.

    Args:
      gate: The shared state.
      key: What is paced.
      policy: How it is paced and recovered.
      refuse: Makes the error raised for a longer wait, from its reason.

    Raises:
      Exception: What ``refuse`` makes, when the wait is too long.

    """
    while True:
        admission = gate.admit(key, policy)
        if not admission.admitted and admission.wait_sec > policy.max_wait_sec:
            raise refuse(admission.reason)
        time.sleep(admission.wait_sec)
        if admission.admitted:
            return


async def waited(
    admit: Callable[[], Awaitable[Admission]],
    policy: GatePolicy,
    *,
    wait: Callable[[float], Awaitable[bool]],
) -> Admission:
    """Ask ``admit`` until it admits the caller, waiting as each answer says.

    The async callers' one loop; ``admitted`` is the sync callers'.

    Args:
      admit: Asks the gate once.
      policy: The longest lock or probe sat out, as ``max_wait_sec``.
      wait: Waits the seconds given; ``True`` when cut short by a change to the
        key, so the slot a pace reserved goes unused and the gate is asked again.

    Returns:
      admission: ``Admission()`` once admitted; else, unwaited, the refusal of a
        lock or probe longer than ``policy.max_wait_sec``.

    """
    while True:
        admission = await admit()
        if not admission.admitted and admission.wait_sec > policy.max_wait_sec:
            return admission
        if admission.wait_sec and await wait(admission.wait_sec):
            continue
        if admission.admitted:
            return Admission()


def clear_host(host: str, *, gate: Gate) -> int:
    """Unlock every site on ``host``, for every egress IP; keep its pace.

    Args:
      host: Exact hostname whose locks to lift.
      gate: The shared state.

    Returns:
      count: Number of keys cleared.

    """
    keys = [key for key in gate.states() if key.startswith(f"{host}:")]
    for key in keys:
        gate.reset(key)
    return len(keys)


def challenged_urls(*, gate: Gate) -> list[str]:
    """Return the URL that drew each standing challenge lock, oldest first.

    Only a bot challenge records one: a person can clear it in a browser, while
    a rate limit or a dropped connection lifts only with time.

    Args:
      gate: The shared state.

    Returns:
      urls: One URL per challenge-locked key, deduplicated.

    """
    held = sorted(
        (s for s in gate.states().values() if s.challenge_url),
        key=lambda s: s.challenged_at,
    )
    return list(dict.fromkeys(s.challenge_url for s in held))


def _site_policy(host: str, path: str, policies: Sequence[GatePolicy]) -> GatePolicy:
    """Return the longest-prefix policy for ``host``/``path``, else the default."""
    matches = [p for p in policies if p.host == host and path.startswith(p.path_prefix)]
    return max(matches, key=lambda p: len(p.path_prefix), default=GatePolicy(host=host))


def _admit(
    state: GateState,
    policy: GatePolicy,
    *,
    now: float,
) -> tuple[Admission, GateState]:
    """Return the admission ``state`` gives at ``now``, and the state after it."""
    lock_ends = max(state.locked_until, state.held_until)
    if now < lock_ends:
        wait = lock_ends - now
        reason = f"cooling down for {wait:.0f}s"
        # Callers one lock released together would meet the next block together.
        jitter = random.SystemRandom().uniform(0.0, policy.jitter_sec)
        return Admission(wait_sec=wait + jitter, admitted=False, reason=reason), state
    if state.failures:
        if now < state.probe_until:
            return Admission(
                wait_sec=min(state.probe_until - now, policy.cooldown_sec),
                admitted=False,
                reason="probing whether its block has lifted",
            ), state
        state = replace(state, probe_until=now + policy.probe_lease_sec)
    if policy.interval_sec <= 0:
        return Admission(), state
    rate = 1.0 / policy.interval_sec
    tokens, updated = (
        (state.tokens, state.updated) if state.updated else (float(policy.burst), now)
    )
    tokens = min(float(policy.burst), tokens + (now - updated) * rate)
    wait = max(0.0, (1.0 - tokens) / rate)
    if wait:
        wait += random.SystemRandom().uniform(0.0, policy.jitter_sec)
    paced = replace(state, tokens=tokens + wait * rate - 1.0, updated=now + wait)
    return Admission(wait_sec=wait), paced


@contextmanager
def _state(path: Path) -> Generator[list[GateState]]:
    """Read, then rewrite if changed, the key's state under an exclusive ``flock``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        # A recovered key's file is deleted under the lock; one opened before
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
        before = _read(os.read(fd, held.st_size))
        box = [before]
        yield box
        if box[0] != before and path.exists():
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, json_dumps(box[0], target=GateState).encode())
    finally:
        os.close(fd)


# Opened without ``O_CREAT``: a file deleted since it was listed is not made again, and
# raises ``FileNotFoundError``.
def _read_shared(path: Path) -> GateState:
    """Read the key's state between writes, under a shared ``flock``."""
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        # A writer truncates, then writes: unlocked, a read between saw no state.
        fcntl.flock(fd, fcntl.LOCK_SH)
        return _read(os.read(fd, os.fstat(fd).st_size))
    finally:
        os.close(fd)


async def _slept(seconds: float) -> bool:
    """Sleep ``seconds``; return ``False``, for a file wakes no one early."""
    await asyncio.sleep(seconds)
    return False


def _read(raw: bytes) -> GateState:
    """Return the state ``raw`` holds; a new or unreadable file is a fresh key."""
    if not raw:
        return GateState()
    try:
        return json_loads(raw, GateState)
    except ValueError:
        return GateState()
