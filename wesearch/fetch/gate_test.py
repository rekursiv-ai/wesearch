"""Tests for the host-wide fetch gate."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fcntl
import time

import pytest

from wesearch.fetch.gate import SitePolicy, challenged_urls, clear_host, gated
from wesearch.types.errors import CooldownError, FetchError, PuzzleChallengeError


_URL = "https://probe.example/search?q=x"
_FAST = (SitePolicy(host="probe.example", cooldown_sec=0.05, max_cooldown_sec=1.0),)


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


def _block(url: str = _URL, *, egress: str = "203.0.113.1") -> None:
    """Send one request through the gate and have the site block it."""
    with (
        pytest.raises(PuzzleChallengeError),
        gated(url, egress=egress, policies=_FAST),
    ):
        raise PuzzleChallengeError("challenge")


def _send(url: str = _URL, *, egress: str = "203.0.113.1") -> None:
    """Send one request through the gate and have the site answer it."""
    with gated(url, egress=egress, policies=_FAST):
        pass


def test_a_block_locks_the_site_for_every_caller() -> None:
    _block()
    with pytest.raises(CooldownError, match=r"probe\.example is cooling down"):
        _send()


def test_a_rate_limit_locks_the_site() -> None:
    with pytest.raises(FetchError), gated(_URL, egress="", policies=_FAST):
        raise FetchError(_URL, 429, {}, b"")
    with pytest.raises(CooldownError):
        _send(egress="")


def test_an_ordinary_failure_does_not_lock() -> None:
    with pytest.raises(FetchError), gated(_URL, egress="", policies=_FAST):
        raise FetchError(_URL, 503, {}, b"")
    _send(egress="")


def test_an_unreachable_site_locks_only_when_its_policy_says_so() -> None:
    dropping = (SitePolicy(host="probe.example", unreachable_blocks=True),)
    with pytest.raises(TimeoutError), gated(_URL, egress="", policies=_FAST):
        raise TimeoutError
    _send(egress="")
    with pytest.raises(TimeoutError), gated(_URL, egress="", policies=dropping):
        raise TimeoutError
    with pytest.raises(CooldownError):
        _send(egress="")


def test_a_refused_connection_counts_as_unreachable() -> None:
    dropping = (SitePolicy(host="probe.example", unreachable_blocks=True),)
    with pytest.raises(FetchError), gated(_URL, egress="", policies=dropping):
        raise FetchError(_URL, 0, {}, b"timed out")
    with pytest.raises(CooldownError):
        _send(egress="")


def test_an_expired_lock_admits_one_probe_while_others_wait(clock: _Clock) -> None:
    _block()
    clock.sleep(0.06)
    with (
        gated(_URL, egress="203.0.113.1", policies=_FAST),
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


def test_a_block_belongs_to_the_site_that_drew_it() -> None:
    policies = (
        SitePolicy(host="probe.example", path_prefix="/search"),
        SitePolicy(host="probe.example", path_prefix="/scholar"),
    )
    with pytest.raises(PuzzleChallengeError), gated(_URL, egress="", policies=policies):
        raise PuzzleChallengeError("challenge")
    with gated("https://probe.example/scholar?q=x", egress="", policies=policies):
        pass


def test_clearing_the_host_lifts_its_lock() -> None:
    _block()
    assert clear_host("probe.example") == 1
    _send()


def test_a_challenge_records_the_url_that_drew_it() -> None:
    _block()
    assert challenged_urls() == [_URL]


def test_a_rate_limit_records_no_challenge() -> None:
    with pytest.raises(FetchError), gated(_URL, egress="", policies=_FAST):
        raise FetchError(_URL, 429, {}, b"")
    assert challenged_urls() == []


def test_one_url_per_site_however_many_blocks() -> None:
    _block(egress="203.0.113.1")
    _block(egress="198.51.100.7")
    assert challenged_urls() == [_URL]


def test_challenges_list_oldest_first() -> None:
    _block("https://probe.example/a")
    time.sleep(0.01)
    with (
        pytest.raises(PuzzleChallengeError),
        gated("https://other.example/b", egress=""),
    ):
        raise PuzzleChallengeError("challenge")
    assert challenged_urls() == ["https://probe.example/a", "https://other.example/b"]


def test_an_answered_site_drops_its_challenge(clock: _Clock) -> None:
    _block()
    clock.sleep(0.06)
    _send()
    assert challenged_urls() == []


def test_clearing_the_host_drops_its_challenge() -> None:
    _block()
    clear_host("probe.example")
    assert challenged_urls() == []


def test_no_state_means_no_challenges() -> None:
    assert challenged_urls() == []


def test_clearing_one_host_leaves_another_locked() -> None:
    _block()
    with (
        pytest.raises(PuzzleChallengeError),
        gated("https://other.example/", egress=""),
    ):
        raise PuzzleChallengeError("challenge")
    assert clear_host("probe.example") == 1
    with pytest.raises(CooldownError), gated("https://other.example/", egress=""):
        pass


def test_clearing_a_host_with_no_state_clears_nothing() -> None:
    assert clear_host("never.example") == 0


def test_the_burst_is_free_and_then_paced(clock: _Clock) -> None:
    policies = (SitePolicy(host="probe.example", interval_sec=0.05, burst=2),)
    start = clock.time()
    for _ in range(2):
        with gated(_URL, egress="", policies=policies):
            pass
    assert clock.time() - start < 0.04
    with gated(_URL, egress="", policies=policies):
        pass
    assert clock.time() - start >= 0.04


def test_jitter_lengthens_only_a_wait_the_pace_imposed(clock: _Clock) -> None:
    policies = (SitePolicy(host="probe.example", interval_sec=0.02, jitter_sec=0.05),)
    start = clock.time()
    with gated(_URL, egress="", policies=policies):
        pass
    assert clock.time() - start < 0.02
    with gated(_URL, egress="", policies=policies):
        pass
    assert clock.time() - start >= 0.02


def test_an_unpaced_site_that_never_failed_leaves_no_state() -> None:
    _send()
    assert clear_host("probe.example") == 0


def test_an_unpaced_site_that_recovered_leaves_no_state(clock: _Clock) -> None:
    # Else every later request to it pays for a locked file read and write.
    _block()
    clock.sleep(0.06)
    _send()
    assert clear_host("probe.example") == 0


def test_a_paced_site_that_recovered_keeps_its_pace(clock: _Clock) -> None:
    policies = (SitePolicy(host="probe.example", interval_sec=0.05, cooldown_sec=0.01),)
    with pytest.raises(PuzzleChallengeError), gated(_URL, egress="", policies=policies):
        raise PuzzleChallengeError("challenge")
    clock.sleep(0.02)
    with gated(_URL, egress="", policies=policies):
        pass
    assert clear_host("probe.example") == 1


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


def test_a_hostless_url_passes_ungated() -> None:
    for _ in range(2):
        with pytest.raises(PuzzleChallengeError), gated("data:,x", egress=""):
            raise PuzzleChallengeError("challenge")


def test_a_hostless_url_that_succeeds_passes_ungated() -> None:
    ran: list[str] = []
    with gated("data:,x", egress=""):
        ran.append("body")
    assert ran == ["body"]
    assert challenged_urls() == []


def test_an_ordinary_failure_frees_the_probe(clock: _Clock) -> None:
    _block()
    clock.sleep(0.06)
    with pytest.raises(FetchError), gated(_URL, egress="203.0.113.1", policies=_FAST):
        raise FetchError(_URL, 503, {}, b"")
    _send()


def test_a_timeout_on_a_site_that_answers_frees_the_probe(clock: _Clock) -> None:
    _block()
    clock.sleep(0.06)
    with pytest.raises(TimeoutError), gated(_URL, egress="203.0.113.1", policies=_FAST):
        raise TimeoutError
    _send()


def test_an_interrupted_request_frees_the_probe(clock: _Clock) -> None:
    _block()
    clock.sleep(0.06)
    with (
        pytest.raises(KeyboardInterrupt),
        gated(_URL, egress="203.0.113.1", policies=_FAST),
    ):
        raise KeyboardInterrupt
    _send()


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
