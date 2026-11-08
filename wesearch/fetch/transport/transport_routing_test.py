"""Tests for persistent automatic web-fetch transport routing."""

from pathlib import Path
from typing import Final

import errno
import os
import time

import pytest

from wesearch.fetch.transport import transport_routing
from wesearch.fetch.transport.transport_routing import (
    remember_zendriver_domain,
    zendriver_domains,
)
from wesearch.lib.userdirs import state_dir


_CWD: Final = Path(__file__).resolve().parent


def test_default_path_uses_per_user_state() -> None:
    """A remembered domain lands in the per-user state file, not anywhere else.

    Asserted by writing through the real code path and reading the file back.
    Comparing two hand-built ``state_dir(...)`` expressions instead compares an
    expression to itself and passes against any implementation.
    """
    remember_zendriver_domain("walled.example")
    expected = state_dir() / "rekursiv-ai" / "wesearch" / "zendriver-domains.txt"
    assert expected.is_file()
    assert "walled.example" in expected.read_text().split()


def test_default_remember_does_not_modify_bundled_manifest() -> None:
    bundled = _CWD / "zendriver-domains.txt"
    # The bundled default list is optional; read defensively so this asserts
    # "the bundle is never mutated" whether it is present or absent.
    before = bundled.read_bytes() if bundled.exists() else None

    remember_zendriver_domain("learned.example")

    after = bundled.read_bytes() if bundled.exists() else None
    assert after == before
    learned = state_dir() / "rekursiv-ai" / "wesearch" / "zendriver-domains.txt"
    assert [line.split("\t")[0] for line in learned.read_text().splitlines()] == [
        "learned.example",
    ]
    assert "learned.example" in zendriver_domains()


def test_absent_domain_list_is_empty(tmp_path: Path) -> None:
    assert zendriver_domains(path=tmp_path / "domains.txt") == frozenset()


def test_remembered_domains_are_normalized_sorted_and_deduplicated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "domains.txt"
    remember_zendriver_domain("B.Example", path=path)
    remember_zendriver_domain("a.example", path=path)
    remember_zendriver_domain("b.example", path=path)

    assert _names(path) == ["a.example", "b.example"]
    assert zendriver_domains(path=path) == frozenset({"a.example", "b.example"})


def test_learned_entry_expires_after_its_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One block must not pin a domain to the browser forever."""
    path = tmp_path / "domains.txt"
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    remember_zendriver_domain("walled.example", path=path)

    monkeypatch.setattr(time, "time", lambda: 1_099.0)
    assert zendriver_domains(path=path, ttl_sec=100) == {"walled.example"}
    monkeypatch.setattr(time, "time", lambda: 1_100.0)
    assert zendriver_domains(path=path, ttl_sec=100) == {"walled.example"}
    monkeypatch.setattr(time, "time", lambda: 1_101.0)
    assert zendriver_domains(path=path, ttl_sec=100) == frozenset()


def test_relearning_refreshes_the_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "domains.txt"
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    remember_zendriver_domain("walled.example", path=path)
    monkeypatch.setattr(time, "time", lambda: 1_090.0)
    remember_zendriver_domain("walled.example", path=path)

    monkeypatch.setattr(time, "time", lambda: 1_150.0)
    assert zendriver_domains(path=path, ttl_sec=100) == {"walled.example"}
    assert path.read_text() == "walled.example\t1090\n"


def test_untimestamped_learned_entry_reads_as_expired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A learned list written before expiry existed must not route forever."""
    path = tmp_path / "domains.txt"
    path.write_text("stale.example\n")
    monkeypatch.setattr(time, "time", lambda: 1.0)

    assert zendriver_domains(path=path, ttl_sec=0) == frozenset()


def test_blank_and_nameless_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "domains.txt"
    path.write_text(f"\n  \t{time.time():.0f}\nkept.example\t{time.time():.0f}\n")

    assert zendriver_domains(path=path) == frozenset({"kept.example"})
    remember_zendriver_domain("new.example", path=path)
    assert _names(path) == ["kept.example", "new.example"]


def test_bundled_list_never_expires() -> None:
    """The bundled roster is curated, untimestamped, and always routes."""
    bundled = _CWD / "zendriver-domains.txt"
    if not bundled.exists():
        pytest.skip("no bundled list in this build")
    first = bundled.read_text().split()[0]

    assert first in zendriver_domains(ttl_sec=0)


def test_reader_proxy_is_never_learned(tmp_path: Path) -> None:
    """The reader proxy is the fallback for a wall; routing it to Chrome breaks that."""
    path = tmp_path / "domains.txt"
    remember_zendriver_domain("r.jina.ai", path=path)

    assert not path.exists()


def _names(path: Path) -> list[str]:
    """Return the domain column of a learned list, in file order."""
    return [line.split("\t")[0] for line in path.read_text().splitlines()]


def test_newline_domain_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid Zendriver domain"):
        remember_zendriver_domain("safe.example\nother.example", path=tmp_path / "x")


def test_read_casefolds_existing_entries(tmp_path: Path) -> None:
    # Runtime lookups key on urlparse().hostname (always lowercase), so a
    # bundled or hand-edited mixed-case entry must still match. The writer
    # casefolds; the reader must too.
    path = tmp_path / "domains.txt"
    path.write_text(f"Foo.Example\t{time.time():.0f}\nBar.EXAMPLE\t{time.time():.0f}\n")

    assert zendriver_domains(path=path) == frozenset({"foo.example", "bar.example"})


def test_unreadable_domain_list_degrades_to_empty(tmp_path: Path) -> None:
    # An optional learned-route cache with corrupt (non-UTF-8) contents must
    # not abort every automatic fetch; it degrades to no learned routing.
    path = tmp_path / "domains.txt"
    path.write_bytes(b"good.example\n\xff\xfe not utf-8\n")

    # One assertion, not an ``or``: the contract is whole-file degradation
    # (``_read_domains`` returns empty on UnicodeDecodeError), so admitting the
    # partial-read alternative would let the behavior drift silently.
    assert zendriver_domains(path=path) == frozenset()


def test_remember_preserves_existing_domains(tmp_path: Path) -> None:
    path = tmp_path / "domains.txt"
    path.write_text(f"kept.example\t{time.time():.0f}\n")

    remember_zendriver_domain("fresh.example", path=path)

    assert zendriver_domains(path=path) == {"fresh.example", "kept.example"}
    assert _names(path) == ["fresh.example", "kept.example"]


def test_read_failure_after_open_degrades_to_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An I/O error mid-read yields no routing, not an aborted fetch.

    ``zendriver_domains`` runs on the per-fetch hot path, so an unreadable
    optional cache must not propagate: guarding only ``os.open`` left every
    automatic fetch failing on a lock or read fault.
    """
    path = tmp_path / "domains.txt"
    path.write_text("good.example\n")

    def raise_eio(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise OSError(errno.EIO, "simulated read failure")

    monkeypatch.setattr(transport_routing, "_read_all", raise_eio)
    assert zendriver_domains(path=path) == frozenset()


def test_corrupt_cache_is_rewritten_rather_than_blocking_writes(
    tmp_path: Path,
) -> None:
    """A non-UTF-8 byte must not permanently prevent learning new domains."""
    path = tmp_path / "domains.txt"
    path.write_bytes(b"\xff\xfe not utf-8\n")

    remember_zendriver_domain("fresh.example", path=path)

    assert "fresh.example" in zendriver_domains(path=path)


def test_corrupt_cache_is_discarded_whole_not_mangled(tmp_path: Path) -> None:
    """Undecodable bytes must not be persisted as replacement-character domains.

    Decoding with ``errors="replace"`` kept the junk as U+FFFD "domains" and
    wrote them back, so the file accumulated entries the read path would have
    rejected outright.
    """
    path = tmp_path / "domains.txt"
    path.write_bytes(b"good.example\n\xff\xfe junk\n")

    remember_zendriver_domain("fresh.example", path=path)

    assert zendriver_domains(path=path) == frozenset({"fresh.example"})
    assert _names(path) == ["fresh.example"]
    assert "\ufffd" not in path.read_text()


def test_read_domains_logs_open_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "domains.txt"

    def fail_open(*args: object, **kwargs: object) -> int:
        del args, kwargs
        raise OSError("permission denied")

    monkeypatch.setattr(os, "open", fail_open)
    caplog.set_level("WARNING", logger=transport_routing.__name__)

    assert zendriver_domains(path=path) == frozenset()
    assert [record.getMessage() for record in caplog.records] == [
        f"Ignoring unreadable Zendriver domain list at {path}.",
    ]


def test_read_domains_logs_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "domains.txt"
    path.write_text("good.example\n")

    def fail_read(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise OSError("read failed")

    monkeypatch.setattr(transport_routing, "_read_all", fail_read)
    caplog.set_level("WARNING", logger=transport_routing.__name__)

    assert zendriver_domains(path=path) == frozenset()
    assert [record.getMessage() for record in caplog.records] == [
        f"Ignoring unreadable Zendriver domain list at {path}.",
    ]


def test_read_domains_logs_decode_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "domains.txt"
    path.write_bytes(b"good.example\n\xff")
    caplog.set_level("WARNING", logger=transport_routing.__name__)

    assert zendriver_domains(path=path) == frozenset()
    assert [record.getMessage() for record in caplog.records] == [
        f"Ignoring undecodable Zendriver domain list at {path}.",
    ]


def test_read_all_joins_short_reads_in_order_until_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    chunks = iter((b"first", b"second", b""))

    def read(descriptor: int, size: int) -> bytes:
        calls.append((descriptor, size))
        return next(chunks)

    monkeypatch.setattr(os, "read", read)

    assert transport_routing._read_all(3) == b"firstsecond"
    assert calls == [(3, 1 << 20)] * 3


def test_read_domains_uses_close_on_exec_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "domains.txt"
    path.write_text(f"good.example\t{time.time():.0f}\n")
    calls: list[tuple[object, ...]] = []
    real_open = os.open

    def record_open(
        file_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        calls.append((file_path, flags) if mode == 0o777 else (file_path, flags, mode))
        return real_open(file_path, flags, mode)

    monkeypatch.setattr(os, "open", record_open)
    assert transport_routing.zendriver_domains(path=path) == {"good.example"}
    assert calls == [
        (path, os.O_RDONLY | os.O_CLOEXEC),
    ]


def test_carriage_return_domain_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"^Invalid Zendriver domain:"):
        remember_zendriver_domain("safe.example\rother.example", path=tmp_path / "x")


def test_remember_uses_private_file_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "domains.txt"
    calls: list[tuple[object, ...]] = []
    real_open = os.open

    def record_open(
        file_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
    ) -> int:
        calls.append((file_path, flags) if mode == 0o777 else (file_path, flags, mode))
        return real_open(file_path, flags, mode)

    monkeypatch.setattr(os, "open", record_open)
    remember_zendriver_domain("private.example", path=path)

    assert calls == [
        (
            path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
            0o600,
        ),
    ]


def test_corrupt_remember_logs_discard(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "domains.txt"
    path.write_bytes(b"\xff")
    caplog.set_level("WARNING", logger=transport_routing.__name__)

    remember_zendriver_domain("fresh.example", path=path)

    assert [record.getMessage() for record in caplog.records] == [
        f"Discarding undecodable Zendriver domain list at {path}.",
    ]


def test_remember_truncates_before_rewriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "domains.txt"
    path.write_text("old.example\n")
    real_ftruncate = os.ftruncate
    sizes: list[int] = []

    def record_ftruncate(descriptor: int, size: int) -> None:
        sizes.append(size)
        real_ftruncate(descriptor, size)

    monkeypatch.setattr(os, "ftruncate", record_ftruncate)
    remember_zendriver_domain("new.example", path=path)

    assert sizes == [0]
    assert _names(path) == ["new.example", "old.example"]


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
