"""Tests for the host-wide fetch gate."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import asyncio
import fcntl
import os
import threading

import pytest

from wesearch.fetch.gate import (
    Admission,
    FileGate,
    GatePolicy,
    GateState,
    admitted,
    challenged_urls,
    clear_host,
    gated,
    waited,
)
from wesearch.lib.userdirs import data_dir
from wesearch.types.errors import (
    CooldownError,
    FetchError,
    GoogleJavascriptRequiredError,
    PuzzleChallengeError,
)


_URL = "https://probe.example/search?q=x"
_FAST = (GatePolicy(host="probe.example", cooldown_sec=0.05, max_cooldown_sec=1.0),)


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def time(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.now += delay


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    made = _Clock()
    monkeypatch.setattr("wesearch.fetch.gate.time", made)
    return made


@pytest.fixture
def gate() -> FileGate:
    return FileGate()


def _block(url: str = _URL, *, egress: str = "203.0.113.1") -> None:
    """Send one request through the gate and have the site block it."""
    with (
        pytest.raises(PuzzleChallengeError),
        gated(url, egress=egress, gate=FileGate(), policies=_FAST),
    ):
        raise PuzzleChallengeError("challenge")


def _send(url: str = _URL, *, egress: str = "203.0.113.1") -> None:
    """Send one request through the gate and have the site answer it."""
    with gated(url, egress=egress, gate=FileGate(), policies=_FAST):
        pass


def test_a_block_locks_the_site_for_every_caller() -> None:
    _block()
    with pytest.raises(CooldownError, match=r"probe\.example is cooling down"):
        _send()


def test_a_rate_limit_locks_the_site(gate: FileGate) -> None:
    with pytest.raises(FetchError), gated(_URL, egress="", gate=gate, policies=_FAST):
        raise FetchError(_URL, 429, {}, b"")
    with pytest.raises(CooldownError):
        _send(egress="")


def test_a_page_that_only_needs_javascript_does_not_lock(gate: FileGate) -> None:
    # Google's enablejs page refused nobody: locking Google for 15 minutes on it
    # failed every search after one render that read the page too early.
    with (
        pytest.raises(GoogleJavascriptRequiredError),
        gated(_URL, egress="", gate=gate, policies=_FAST),
    ):
        raise GoogleJavascriptRequiredError("enablejs")
    _send(egress="")
    assert challenged_urls(gate=gate) == []


def test_a_page_that_only_needs_javascript_frees_the_probe(
    clock: _Clock,
    gate: FileGate,
) -> None:
    # After a block, the first caller back is the probe; a JavaScript page says
    # nothing about the block, so the next caller must not wait out its lease.
    _block()
    clock.now += 1.0
    with (
        pytest.raises(GoogleJavascriptRequiredError),
        gated(_URL, egress="203.0.113.1", gate=gate, policies=_FAST),
    ):
        raise GoogleJavascriptRequiredError("enablejs")
    _send()


def test_an_ordinary_failure_does_not_lock(gate: FileGate) -> None:
    with pytest.raises(FetchError), gated(_URL, egress="", gate=gate, policies=_FAST):
        raise FetchError(_URL, 503, {}, b"")
    _send(egress="")


def test_an_unreachable_site_locks_only_when_its_policy_says_so(
    gate: FileGate,
) -> None:
    dropping = (GatePolicy(host="probe.example", unreachable_blocks=True),)
    with pytest.raises(TimeoutError), gated(_URL, egress="", gate=gate, policies=_FAST):
        raise TimeoutError
    _send(egress="")
    with (
        pytest.raises(TimeoutError),
        gated(_URL, egress="", gate=gate, policies=dropping),
    ):
        raise TimeoutError
    with pytest.raises(CooldownError):
        _send(egress="")


def test_a_refused_connection_counts_as_unreachable(gate: FileGate) -> None:
    dropping = (GatePolicy(host="probe.example", unreachable_blocks=True),)
    with (
        pytest.raises(FetchError),
        gated(_URL, egress="", gate=gate, policies=dropping),
    ):
        raise FetchError(_URL, 0, {}, b"timed out")
    with pytest.raises(CooldownError):
        _send(egress="")


def test_an_expired_lock_admits_one_probe_while_others_wait(
    clock: _Clock,
    gate: FileGate,
) -> None:
    _block()
    clock.sleep(0.06)
    with (
        gated(_URL, egress="203.0.113.1", gate=gate, policies=_FAST),
        pytest.raises(CooldownError, match="probing"),
    ):
        _send()


def test_a_successful_probe_unlocks_every_caller(clock: _Clock) -> None:
    _block()
    clock.sleep(0.06)
    _send()
    for _ in range(3):
        _send()


def test_a_failed_probe_doubles_the_lock(clock: _Clock) -> None:
    _block()
    clock.sleep(0.06)
    _block()
    clock.sleep(0.06)
    with pytest.raises(CooldownError, match="cooling down"):
        _send()


def test_a_success_resets_the_backoff(clock: _Clock) -> None:
    _block()
    clock.sleep(0.06)
    _block()
    clock.sleep(0.11)
    _send()
    _block()
    clock.sleep(0.06)
    _send()


def test_a_block_belongs_to_the_egress_that_drew_it() -> None:
    _block(egress="203.0.113.1")
    _send(egress="198.51.100.7")


def test_a_block_belongs_to_the_site_that_drew_it(gate: FileGate) -> None:
    policies = (
        GatePolicy(host="probe.example", path_prefix="/"),
        GatePolicy(host="probe.example", path_prefix="/search"),
        GatePolicy(host="probe.example", path_prefix="/scholar"),
    )
    with (
        pytest.raises(PuzzleChallengeError),
        gated(_URL, egress="", gate=gate, policies=policies),
    ):
        raise PuzzleChallengeError("challenge")
    scholar = "https://probe.example/scholar?q=x"
    with gated(scholar, egress="", gate=gate, policies=policies):
        pass


def test_clearing_the_host_lifts_its_lock(gate: FileGate) -> None:
    _block()
    assert clear_host("probe.example", gate=gate) == 1
    _send()


def test_a_challenge_records_the_url_that_drew_it(gate: FileGate) -> None:
    _block()
    assert challenged_urls(gate=gate) == [_URL]


def test_a_rate_limit_records_no_challenge(gate: FileGate) -> None:
    with pytest.raises(FetchError), gated(_URL, egress="", gate=gate, policies=_FAST):
        raise FetchError(_URL, 429, {}, b"")
    assert challenged_urls(gate=gate) == []


def test_one_url_per_site_however_many_blocks(gate: FileGate) -> None:
    _block(egress="203.0.113.1")
    _block(egress="198.51.100.7")
    assert challenged_urls(gate=gate) == [_URL]


def test_challenges_list_oldest_first(clock: _Clock, gate: FileGate) -> None:
    with (
        pytest.raises(PuzzleChallengeError),
        gated("https://other.example/b", egress="", gate=gate),
    ):
        raise PuzzleChallengeError("challenge")
    clock.now += 1.0
    _block("https://probe.example/a")
    assert challenged_urls(gate=gate) == [
        "https://other.example/b",
        "https://probe.example/a",
    ]


def test_an_answered_site_drops_its_challenge(clock: _Clock, gate: FileGate) -> None:
    _block()
    clock.sleep(0.06)
    _send()
    assert challenged_urls(gate=gate) == []


def test_clearing_the_host_drops_its_challenge(gate: FileGate) -> None:
    _block()
    clear_host("probe.example", gate=gate)
    assert challenged_urls(gate=gate) == []


def test_no_state_means_no_challenges(gate: FileGate) -> None:
    assert challenged_urls(gate=gate) == []


def test_clearing_one_host_leaves_another_locked(gate: FileGate) -> None:
    _block()
    with (
        pytest.raises(PuzzleChallengeError),
        gated("https://other.example/", egress="", gate=gate),
    ):
        raise PuzzleChallengeError("challenge")
    assert clear_host("probe.example", gate=gate) == 1
    with (
        pytest.raises(CooldownError),
        gated("https://other.example/", egress="", gate=gate),
    ):
        pass


def test_clearing_a_host_with_no_state_clears_nothing(gate: FileGate) -> None:
    assert clear_host("never.example", gate=gate) == 0


def test_the_burst_is_free_and_then_paced(clock: _Clock, gate: FileGate) -> None:
    policies = (GatePolicy(host="probe.example", interval_sec=0.05, burst=2),)
    start = clock.time()
    elapsed: list[float] = []
    for _ in range(4):
        with gated(_URL, egress="", gate=gate, policies=policies):
            elapsed.append(clock.time() - start)
    assert elapsed == pytest.approx([0.0, 0.0, 0.05, 0.10])
    clock.sleep(0.10)
    for _ in range(3):
        with gated(_URL, egress="", gate=gate, policies=policies):
            elapsed.append(clock.time() - start)
    assert elapsed[4:] == pytest.approx([0.20, 0.20, 0.25])


@pytest.mark.asyncio
async def test_a_caller_sits_out_a_lock_it_may_wait_for_then_probes(
    clock: _Clock,
    gate: FileGate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def sleep(delay: float) -> None:
        slept.append(delay)
        clock.sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    patient = GatePolicy(host="probe.example", cooldown_sec=0.5, max_wait_sec=0.5)
    gate.trip("probe.example", patient)

    assert await gate.admitted("probe.example", patient) == Admission()
    assert slept == [pytest.approx(0.5)]


@pytest.mark.asyncio
async def test_a_lock_longer_than_the_wait_allowed_is_refused_unwaited(
    gate: FileGate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asyncio, "sleep", _never_sleep)
    hurried = GatePolicy(host="probe.example", cooldown_sec=0.05)
    gate.trip("probe.example", hurried)

    refusal = await gate.admitted("probe.example", hurried)

    assert not refusal.admitted
    assert refusal.wait_sec == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_a_wait_cut_short_asks_the_gate_again_before_sending() -> None:
    """The pace's slot went unused, so the slot it was waiting for is not taken."""
    answers = [Admission(wait_sec=0.5), Admission(wait_sec=0.0)]
    asked: list[Admission] = []
    waits: list[float] = []

    async def admit() -> Admission:
        asked.append(answers[len(asked)])
        return asked[-1]

    async def wait(seconds: float) -> bool:
        waits.append(seconds)
        return True

    patient = GatePolicy(host="probe.example", max_wait_sec=1.0)

    assert await waited(admit, patient, wait=wait) == Admission()
    assert (len(asked), waits) == (2, [0.5])


async def _never_sleep(delay: float) -> None:
    raise AssertionError(f"slept {delay}s on a lock it should refuse")


def test_jitter_lengthens_only_a_wait_the_pace_imposed(
    clock: _Clock,
    gate: FileGate,
) -> None:
    policies = (GatePolicy(host="probe.example", interval_sec=0.02, jitter_sec=0.05),)
    start = clock.time()
    with gated(_URL, egress="", gate=gate, policies=policies):
        pass
    assert clock.time() - start < 0.02
    with gated(_URL, egress="", gate=gate, policies=policies):
        pass
    assert clock.time() - start >= 0.02


def test_an_unpaced_site_that_never_failed_leaves_no_state(gate: FileGate) -> None:
    _send()
    assert gate.states() == {}


def test_an_unpaced_site_that_recovered_leaves_no_state(
    clock: _Clock,
    gate: FileGate,
) -> None:
    # Else every later request to it pays for a locked file read and write.
    _block()
    clock.sleep(0.06)
    _send()
    assert gate.states() == {}


def test_a_paced_site_that_recovered_keeps_its_pace(
    clock: _Clock,
    gate: FileGate,
) -> None:
    policies = (GatePolicy(host="probe.example", interval_sec=0.05, cooldown_sec=0.01),)
    with (
        pytest.raises(PuzzleChallengeError),
        gated(_URL, egress="", gate=gate, policies=policies),
    ):
        raise PuzzleChallengeError("challenge")
    clock.sleep(0.02)
    with gated(_URL, egress="", gate=gate, policies=policies):
        pass
    assert clear_host("probe.example", gate=gate) == 1
    start = clock.time()
    with gated(_URL, egress="", gate=gate, policies=policies):
        pass
    assert clock.time() - start == pytest.approx(0.05)


def test_a_block_that_waited_on_a_deleted_gate_still_locks() -> None:
    # A writer that opened the file before another process deleted it must not
    # record the block on the unlinked inode, where no one would read it.
    real = fcntl.flock
    unlinked: list[Path] = []

    def flock(fd: int, op: int) -> None:
        if not unlinked:
            path = Path(f"/proc/self/fd/{fd}").readlink()
            path.unlink()
            unlinked.append(path)
        real(fd, op)

    with patch("wesearch.fetch.gate.fcntl.flock", flock):
        _block()
    assert unlinked
    with pytest.raises(CooldownError):
        _send()


def test_a_hostless_url_passes_ungated(gate: FileGate) -> None:
    for _ in range(2):
        with (
            pytest.raises(PuzzleChallengeError),
            gated("data:,x", egress="", gate=gate),
        ):
            raise PuzzleChallengeError("challenge")


def test_a_hostless_url_that_succeeds_passes_ungated(gate: FileGate) -> None:
    ran: list[str] = []
    with gated("data:,x", egress="", gate=gate):
        ran.append("body")
    assert ran == ["body"]
    assert challenged_urls(gate=gate) == []


def test_an_ordinary_failure_frees_the_probe(clock: _Clock, gate: FileGate) -> None:
    _block()
    clock.sleep(0.06)
    with (
        pytest.raises(FetchError),
        gated(_URL, egress="203.0.113.1", gate=gate, policies=_FAST),
    ):
        raise FetchError(_URL, 503, {}, b"")
    _block()
    clock.sleep(0.06)
    with pytest.raises(CooldownError):
        _send()


def test_a_timeout_on_a_site_that_answers_frees_the_probe(
    clock: _Clock,
    gate: FileGate,
) -> None:
    _block()
    clock.sleep(0.06)
    with (
        pytest.raises(TimeoutError),
        gated(_URL, egress="203.0.113.1", gate=gate, policies=_FAST),
    ):
        raise TimeoutError
    _send()


def test_an_interrupted_request_frees_the_probe(clock: _Clock, gate: FileGate) -> None:
    _block()
    clock.sleep(0.06)
    with (
        pytest.raises(KeyboardInterrupt),
        gated(_URL, egress="203.0.113.1", gate=gate, policies=_FAST),
    ):
        raise KeyboardInterrupt
    _send()


# -- hold and admitted: a lock a server names, and a wait sat out ----------------


@pytest.mark.parametrize("duration", [39.0, 60.0])
def test_a_hold_short_enough_is_sat_out(
    clock: _Clock,
    gate: FileGate,
    duration: float,
) -> None:
    policy = GatePolicy(max_wait_sec=60.0)
    start = clock.now
    gate.hold("anthropic@work", until=start + duration)
    admitted(gate, "anthropic@work", policy, refuse=RuntimeError)
    assert clock.now == pytest.approx(start + duration)


def test_a_hold_longer_than_the_wait_refuses(clock: _Clock, gate: FileGate) -> None:
    policy = GatePolicy(max_wait_sec=60.0)
    gate.hold("anthropic@work", until=clock.now + 600.0)
    with pytest.raises(RuntimeError, match="cooling down for 600s"):
        admitted(gate, "anthropic@work", policy, refuse=RuntimeError)
    assert clock.now == 1_000.0


def test_a_hold_set_during_the_wait_is_sat_out_too(
    clock: _Clock,
    gate: FileGate,
) -> None:
    # The first wait ends where a second caller's 429 began another: the gate
    # is asked again rather than sent through.
    policy = GatePolicy(max_wait_sec=60.0)
    gate.hold("k", until=clock.now + 10.0)
    sleeps: list[float] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        clock.now += delay
        if len(sleeps) == 1:
            gate.hold("k", until=clock.now + 20.0)

    with patch.object(clock, "sleep", sleep):
        admitted(gate, "k", policy, refuse=RuntimeError)
    assert sleeps[:2] == [pytest.approx(10.0), pytest.approx(20.0)]


def test_a_hold_draws_no_probe(clock: _Clock, gate: FileGate) -> None:
    gate.hold("k", until=clock.now + 1.0)
    clock.now += 2.0
    policy = GatePolicy()
    assert gate.admit("k", policy) == Admission()
    assert gate.admit("k", policy) == Admission()


def test_a_hold_never_shortens_a_lock(clock: _Clock, gate: FileGate) -> None:
    gate.hold("k", until=clock.now + 50.0)
    gate.hold("k", until=clock.now + 5.0)
    assert gate.states()["k"].held_until == clock.now + 50.0


def test_a_success_lifts_a_blocks_lock_but_not_a_servers_hold(
    clock: _Clock,
    gate: FileGate,
) -> None:
    # A reply that succeeds and says its window is spent: the reset after it
    # must not lift the hold it set.
    policy = GatePolicy()
    gate.trip("k", policy)
    gate.hold("k", until=clock.now + 600.0)
    gate.reset("k", forget=True)
    assert gate.states() == {"k": GateState(held_until=clock.now + 600.0)}
    clock.now += 601.0
    gate.reset("k", forget=True)
    assert gate.states() == {}


def test_waiting_on_a_probe_polls_at_the_cooldown(
    clock: _Clock,
    gate: FileGate,
) -> None:
    policy = GatePolicy(cooldown_sec=2.0, max_wait_sec=600.0)
    gate.trip("k", policy)
    clock.now += 2.0
    assert gate.admit("k", policy).admitted
    waiting = gate.admit("k", policy)
    assert (waiting.admitted, waiting.wait_sec) == (False, 2.0)
    clock.now += policy.probe_lease_sec - policy.cooldown_sec / 2
    waiting = gate.admit("k", policy)
    assert (waiting.admitted, waiting.wait_sec) == (False, 1.0)
    clock.now += waiting.wait_sec
    assert gate.admit("k", policy).admitted


def test_keys_that_are_no_filenames_round_trip(gate: FileGate) -> None:
    gate.hold("openai/sub@a b", until=5_000.0)
    assert set(gate.states()) == {"openai/sub@a b"}
    assert all(
        p.parent == data_dir() / "rekursiv-ai" / "wesearch" / "ratelimit"
        for p in (data_dir() / "rekursiv-ai" / "wesearch" / "ratelimit").iterdir()
    )


def test_an_unreadable_state_reads_as_fresh(gate: FileGate) -> None:
    base = data_dir() / "rekursiv-ai" / "wesearch" / "ratelimit"
    base.mkdir(parents=True)
    (base / "k.gate").write_text("{not json")
    assert gate.states() == {"k": GateState()}


def test_a_key_cleared_while_listed_is_skipped(
    gate: FileGate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Another process's reset deletes a recovered key's file between the
    # listing and the read.
    gate.hold("gone", until=5_000.0)
    gate.hold("kept", until=5_000.0)
    real = os.open

    def opened(path: str | os.PathLike[str], flags: int, *args: int) -> int:
        if Path(path).name == "gone.gate":
            raise FileNotFoundError(path)
        return real(path, flags, *args)

    monkeypatch.setattr(os, "open", opened)
    assert set(gate.states()) == {"kept"}


def test_a_state_listed_mid_write_is_read_once_written(gate: FileGate) -> None:
    """A write truncates, then writes: read between, the key had no state at all."""
    gate.hold("k", until=5_000.0)
    path = data_dir() / "rekursiv-ai" / "wesearch" / "ratelimit" / "k.gate"
    written = path.read_bytes()
    listed: list[dict[str, GateState]] = []
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.ftruncate(fd, 0)
        reading = threading.Thread(target=lambda: listed.append(gate.states()))
        reading.start()
        reading.join(0.1)
        os.write(fd, written)
    finally:
        os.close(fd)
    reading.join(5)
    assert listed == [{"k": GateState(held_until=5_000.0)}]


def test_a_key_that_is_no_site_is_paced_with_no_host(gate: FileGate) -> None:
    """An account is paced and held as a site is, and matched by no URL."""
    account = GatePolicy(cooldown_sec=5.0, max_wait_sec=60.0)
    assert account.host == ""
    gate.trip("openai-sub@work/sol-6.1", account)
    assert not gate.admit("openai-sub@work/sol-6.1", account).admitted


def test_a_gate_kept_elsewhere_keeps_its_own_state(tmp_path: Path) -> None:
    elsewhere = FileGate(base_dir=tmp_path)
    elsewhere.hold("k", until=5_000.0)
    assert set(elsewhere.states()) == {"k"}
    assert FileGate().states() == {}


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
