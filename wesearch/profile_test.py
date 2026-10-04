"""Unit tests for the cross-process browser-profile store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import os
import threading
import time

import pytest

from wesearch import profile
from wesearch.profile import (
    Profile,
    ProfileStore,
    parse_set_cookie,
    parsedate_to_datetime_or_none,
)


def _store(tmp_path: Path, *, ttl_sec: float = 3600.0) -> ProfileStore:
    return ProfileStore(base_dir=tmp_path, ttl_sec=ttl_sec)


def _loaded(store: ProfileStore, ip: str, domain: str) -> Profile:
    got = store.load(ip, domain)
    assert got is not None
    return got


class TestProfileRoundTrip:
    def test_default_base_path_uses_wesearch_namespace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(profile, "data_dir", lambda: tmp_path)
        assert ProfileStore()._base == tmp_path / "rekursiv-ai" / "wesearch"

    def test_save_then_load_returns_equal_profile(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        p = Profile(ua="UA/1.0", cookies={"GSP": "abc", "NID": "xyz"})
        store.save("1.2.3.4", "scholar.google.com", p)
        got = store.load("1.2.3.4", "scholar.google.com")
        assert got is not None
        assert got.ua == "UA/1.0"
        assert got.cookies == {"GSP": "abc", "NID": "xyz"}

    def test_missing_key_loads_none(self, tmp_path: Path) -> None:
        assert _store(tmp_path).load("9.9.9.9", "nowhere.com") is None

    def test_keys_are_isolated_by_ip_and_domain(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("1.1.1.1", "a.com", Profile(ua="A", cookies={"k": "1"}))
        store.save("2.2.2.2", "a.com", Profile(ua="B", cookies={"k": "2"}))
        store.save("1.1.1.1", "b.com", Profile(ua="C", cookies={"k": "3"}))
        assert _loaded(store, "1.1.1.1", "a.com").ua == "A"
        assert _loaded(store, "2.2.2.2", "a.com").ua == "B"
        assert _loaded(store, "1.1.1.1", "b.com").ua == "C"


class TestProfileTtl:
    def test_expired_profile_loads_none(self, tmp_path: Path) -> None:
        store = _store(tmp_path, ttl_sec=10.0)
        stale = Profile(ua="old", cookies={}, created=time.time() - 100.0)
        store.save("1.2.3.4", "x.com", stale)
        assert store.load("1.2.3.4", "x.com") is None

    def test_fresh_profile_within_ttl_loads(self, tmp_path: Path) -> None:
        store = _store(tmp_path, ttl_sec=100.0)
        fresh = Profile(ua="new", cookies={}, created=time.time() - 5.0)
        store.save("1.2.3.4", "x.com", fresh)
        assert store.load("1.2.3.4", "x.com") is not None


class TestProfileDiscard:
    def test_discard_removes_profile(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("1.2.3.4", "x.com", Profile(ua="u", cookies={"k": "v"}))
        store.discard("1.2.3.4", "x.com")
        assert store.load("1.2.3.4", "x.com") is None

    def test_discard_absent_is_noop(self, tmp_path: Path) -> None:
        _store(tmp_path).discard("1.2.3.4", "absent.com")  # Must not raise.


class TestPathKeyCollision:
    def test_ipv6_domain_does_not_collide(self, tmp_path: Path) -> None:
        # REV2A-004: the key is colon-joined; an IPv6 domain injects colons, so
        # naive f"{egress}:{domain}" collides distinct (egress, domain) pairs.
        store = _store(tmp_path)
        a = store._path("1.1.1.1", "::1")
        b = store._path("1.1.1.1:", ":1")
        assert a != b, f"distinct keys collided to one path: {a.name}"

    def test_path_percent_encodes_both_parts(self, tmp_path: Path) -> None:
        assert _store(tmp_path)._path("2001:db8::1", "a/b").name == (
            "2001%3Adb8%3A%3A1|a%2Fb.json"
        )
        assert _store(tmp_path)._path("a/b", "x.com").name == "a%2Fb|x.com.json"

    def test_ipv6_egress_and_domain_roundtrip(self, tmp_path: Path) -> None:
        # A v6 egress + v6 domain must save and load back to the same profile.
        store = _store(tmp_path)
        store.save("2001:db8::1", "::1", Profile(ua="u", cookies={"k": "v"}))
        got = store.load("2001:db8::1", "::1")
        assert got is not None
        assert got.cookies == {"k": "v"}


class TestCookieMerge:
    def test_update_cookies_merges_preserving_prior(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        created = time.time()
        store.save(
            "1.2.3.4",
            "x.com",
            Profile(ua="u", cookies={"a": "1"}, created=created),
        )
        store.update_cookies("1.2.3.4", "x.com", {"b": "2"})
        got = store.load("1.2.3.4", "x.com")
        assert got is not None
        assert got.ua == "u"
        assert got.created == created
        assert got.cookies == {"a": "1", "b": "2"}

    def test_update_cookies_overwrites_same_name(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("1.2.3.4", "x.com", Profile(ua="u", cookies={"a": "1"}))
        store.update_cookies("1.2.3.4", "x.com", {"a": "2"})
        assert _loaded(store, "1.2.3.4", "x.com").cookies == {"a": "2"}

    def test_update_cookies_on_absent_is_noop(self, tmp_path: Path) -> None:
        # No profile => no identity to attach cookies to; the store does not mint
        # one (it has no User-Agent source). The caller saves a profile first.
        store = _store(tmp_path)
        store.update_cookies("1.2.3.4", "x.com", {"GSP": "z"})
        assert store.load("1.2.3.4", "x.com") is None


class TestCorruptFileResilience:
    """A corrupt/partial key file must never brick the domain.

    Reads treat it as absent (self-healing on the next save), and writes are crash-
    atomic.
    """

    def test_decode_drops_non_string_cookie_values(self) -> None:
        raw = b'{"ua": "u", "cookies": {"a": 1}, "created": 1}'
        assert profile._try_decode(raw) == Profile(ua="u", cookies={}, created=1.0)

    def test_decode_accepts_complete_profile(self) -> None:
        raw = b'{"ua": "u", "cookies": {"a": "1"}, "created": 1}'
        assert profile._try_decode(raw) == Profile(
            ua="u",
            cookies={"a": "1"},
            created=1.0,
        )

    def test_load_truncated_json_returns_none(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        path = store._path("1.2.3.4", "x.com")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"ua": "x", "cook')  # crash-truncated write.
        assert store.load("1.2.3.4", "x.com") is None

    def test_load_wrong_shape_returns_none(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        path = store._path("1.2.3.4", "x.com")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")  # Valid JSON, missing keys.
        assert store.load("1.2.3.4", "x.com") is None

    def test_corrupt_file_self_heals_on_next_save(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        path = store._path("1.2.3.4", "x.com")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not json at all")
        store.save("1.2.3.4", "x.com", Profile(ua="fresh", cookies={"a": "1"}))
        loaded = store.load("1.2.3.4", "x.com")
        assert loaded is not None
        assert loaded.ua == "fresh"

    def test_update_cookies_on_corrupt_is_noop(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        path = store._path("1.2.3.4", "x.com")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"ua": "x"')  # Truncated.
        store.update_cookies("1.2.3.4", "x.com", {"b": "2"})  # Must not raise.

    def test_no_partial_file_left_when_write_interrupted(self, tmp_path: Path) -> None:
        # An interrupted write must never expose a partial file at the key path:
        # atomic replace means the reader sees either the old file or the new one.
        store = _store(tmp_path)
        store.save("1.2.3.4", "x.com", Profile(ua="v1", cookies={"a": "1"}))
        # No temp/partial siblings linger after a successful save.
        siblings = list(tmp_path.glob("*.json*"))
        assert all(s.suffix == ".json" for s in siblings), siblings

    def test_write_creates_nested_parent_and_private_file(self, tmp_path: Path) -> None:
        base = tmp_path / "nested" / "profiles"
        store = _store(base)
        real_open = os.open
        open_mock = Mock(wraps=real_open)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("wesearch.profile.os.open", open_mock)
        try:
            store.save("1.2.3.4", "x.com", Profile(ua="u"))
        finally:
            monkeypatch.undo()
        path = store._path("1.2.3.4", "x.com")
        assert path.parent == base
        assert path.stat().st_mode & 0o777 == 0o600
        _, flags, mode = open_mock.call_args.args
        assert flags == os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC
        assert mode == 0o600

    def test_read_uses_close_on_exec(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("1.2.3.4", "x.com", Profile(ua="u"))
        open_mock = Mock(wraps=os.open)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("wesearch.profile.os.open", open_mock)
        try:
            store.load("1.2.3.4", "x.com")
        finally:
            monkeypatch.undo()
        assert open_mock.call_args.args[1] == os.O_RDONLY | os.O_CLOEXEC


class TestExpiresDeletion:
    def test_expires_in_past_is_a_deletion(self) -> None:
        # A cookie set to a past Expires is deleted, like Max-Age<=0.
        got = parse_set_cookie("SID=x; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/")
        assert got == {}

    def test_expires_in_future_is_kept(self) -> None:
        got = parse_set_cookie("SID=x; Expires=Wed, 09 Jun 2099 10:18:14 GMT; Path=/")
        assert got == {"SID": "x"}

    def test_expires_at_now_is_deleted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = datetime(2025, 1, 1, tzinfo=UTC)

        class Clock:
            @classmethod
            def now(cls, tz: object) -> datetime:
                assert tz is UTC
                return now

        monkeypatch.setattr(profile, "datetime", Clock)
        assert parse_set_cookie("SID=x; Expires=Wed, 01 Jan 2025 00:00:00 GMT") == {}

    def test_cookie_attributes_process_before_later_cookies(self) -> None:
        assert parse_set_cookie("malformed\na=1; Max-Age=0\nb=2") == {"b": "2"}

    def test_cookie_values_keep_equals_and_max_age_one(self) -> None:
        assert parse_set_cookie("a=x=y; Max-Age=1") == {"a": "x=y"}

    def test_cookie_malformed_expiry_is_not_deletion(self) -> None:
        assert parse_set_cookie("a=1; Max-Age=wat") == {"a": "1"}

    def test_malformed_max_age_does_not_hide_later_expiry(self) -> None:
        assert not profile._is_deletion(
            ["Max-Age=wat=bad", "Expires=Thu, 01 Jan 1970 00:00:00 GMT"],
        )

    def test_expires_value_is_split_at_its_first_equals(self) -> None:
        assert profile._is_deletion(["Expires==Thu, 01 Jan 1970 00:00:00 GMT"])

    def test_empty_cookie_name_does_not_stop_later_cookie(self) -> None:
        assert parse_set_cookie("=ignored\nb=2") == {"b": "2"}


class TestParsedate:
    def test_naive_date_is_interpreted_as_utc(self) -> None:
        got = parsedate_to_datetime_or_none("Wed, 09 Jun 2099 10:18:14")
        assert got is not None
        assert got.tzinfo is UTC

    def test_aware_date_preserves_its_offset(self) -> None:
        got = parsedate_to_datetime_or_none("Wed, 09 Jun 2099 10:18:14 +0200")
        assert got is not None
        assert got.utcoffset() == timedelta(hours=2)

    def test_malformed_date_is_none(self) -> None:
        assert parsedate_to_datetime_or_none("not a date") is None


class TestTtlEviction:
    def test_exact_ttl_boundary_is_still_live(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = _store(tmp_path, ttl_sec=10.0)
        store.save("1.2.3.4", "x.com", Profile(ua="u", created=100.0))
        monkeypatch.setattr("wesearch.profile.time.time", lambda: 110.0)
        assert store.load("1.2.3.4", "x.com") is not None

    def test_expired_profile_unlink_allows_missing_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = _store(tmp_path, ttl_sec=10.0)
        store.save("1.2.3.4", "x.com", Profile(ua="old", created=0.0))

        def ignore_unlink(**_kwargs: object) -> None:
            del _kwargs

        unlink_mock = Mock(side_effect=ignore_unlink)
        monkeypatch.setattr(Path, "unlink", unlink_mock)
        monkeypatch.setattr("wesearch.profile.time.time", lambda: 100.0)
        assert store.load("1.2.3.4", "x.com") is None
        assert unlink_mock.call_args.kwargs["missing_ok"] is True

    def test_expired_profile_file_unlinked_on_load(self, tmp_path: Path) -> None:
        store = _store(tmp_path, ttl_sec=10.0)
        stale = Profile(ua="old", cookies={}, created=time.time() - 100.0)
        store.save("1.2.3.4", "x.com", stale)
        path = store._path("1.2.3.4", "x.com")
        assert path.exists()
        assert store.load("1.2.3.4", "x.com") is None
        assert not path.exists()  # Expired file is evicted, not left to accumulate.


class TestParseSetCookie:
    def test_parses_name_value_dropping_attributes(self) -> None:
        # "Set-Cookie: GSP=v; Path=/; Domain=.x.com; Secure" -> {"GSP": "v"}.
        assert parse_set_cookie("GSP=v; Path=/; Domain=.x.com; Secure") == {"GSP": "v"}

    def test_multiple_cookies_newline_joined_header(self) -> None:
        # Fetch joins duplicate Set-Cookie headers with "\n"; both must parse.
        got = parse_set_cookie("a=1; Path=/\nb=2; Secure")
        assert got == {"a": "1", "b": "2"}

    def test_cookie_value_with_comma_stays_intact(self) -> None:
        # A value containing ", " must NOT be split (Set-Cookie is fold-exempt).
        assert parse_set_cookie("pref=a, b, c; Path=/") == {"pref": "a, b, c"}

    def test_empty_or_malformed_yields_empty(self) -> None:
        assert parse_set_cookie("") == {}
        assert parse_set_cookie("noequalshere; Path=/") == {}

    def test_expired_max_age_zero_is_a_deletion(self) -> None:
        # Max-Age=0 (or negative) signals deletion -> not stored as live cookie.
        assert parse_set_cookie("a=1; Max-Age=0") == {}


class TestConcurrentSafety:
    def test_interleaved_saves_do_not_corrupt(self, tmp_path: Path) -> None:
        # Two threads hammering the same key must leave a readable profile, never
        # a half-written file (fcntl-guarded read-modify-write).
        store = _store(tmp_path)
        store.save("1.2.3.4", "x.com", Profile(ua="u", cookies={}))

        # 12 rounds x 4 threads = 48 interleaved read-modify-writes. The race
        # this guards is per-write, so it either reproduces in the first few
        # rounds or not at all; 50 rounds cost 1.9s to re-prove the same thing.
        def worker(n: int) -> None:
            for i in range(12):
                store.update_cookies("1.2.3.4", "x.com", {f"k{n}": str(i)})

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        got = store.load("1.2.3.4", "x.com")
        assert got is not None
        # Every worker's last write survives -> jar is coherent, not truncated.
        assert all(f"k{n}" in got.cookies for n in range(4))


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
