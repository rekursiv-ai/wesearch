"""Tests for persistent automatic web-fetch transport routing."""

from pathlib import Path
from typing import Final

import errno
import os

import pytest

from wesearch.fetch.transport import transport_routing
from wesearch.fetch.transport.transport_routing import (
    remember_zendriver_domain,
    zendriver_domains,
)
from wesearch.lib.userdirs import state_dir


_CWD: Final = Path(__file__).resolve().parent


def test_default_path_uses_per_user_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remembered domain lands in the per-user state file, not anywhere else.

    Asserted by writing through the real code path and reading the file back.
    Comparing two hand-built ``state_dir(...)`` expressions instead compares an
    expression to itself and passes against any implementation.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    remember_zendriver_domain("walled.example")
    expected = state_dir() / "rekursiv-ai" / "wesearch" / "zendriver-domains.txt"
    assert expected.is_file()
    assert "walled.example" in expected.read_text().split()


def test_default_remember_does_not_modify_bundled_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = _CWD / "zendriver-domains.txt"
    # The bundled default list is optional; read defensively so this asserts
    # "the bundle is never mutated" whether it is present or absent.
    before = bundled.read_bytes() if bundled.exists() else None
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    remember_zendriver_domain("learned.example")

    after = bundled.read_bytes() if bundled.exists() else None
    assert after == before
    assert (
        state_dir() / "rekursiv-ai" / "wesearch" / "zendriver-domains.txt"
    ).read_text() == "learned.example\n"
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

    assert path.read_text() == "a.example\nb.example\n"
    assert zendriver_domains(path=path) == frozenset({"a.example", "b.example"})


def test_newline_domain_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid Zendriver domain"):
        remember_zendriver_domain("safe.example\nother.example", path=tmp_path / "x")


def test_read_casefolds_existing_entries(tmp_path: Path) -> None:
    # Runtime lookups key on urlparse().hostname (always lowercase), so a
    # bundled or hand-edited mixed-case entry must still match. The writer
    # casefolds; the reader must too.
    path = tmp_path / "domains.txt"
    path.write_text("Foo.Example\nBar.EXAMPLE\n")

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


def test_remember_preserves_domains_past_the_read_chunk(tmp_path: Path) -> None:
    # The domain list must be read to EOF before rewrite; a fixed read cap would
    # silently drop every domain past it on the next remember.
    path = tmp_path / "domains.txt"
    # 50k * 25 bytes/line ~= 1.25 MiB, safely past the old 1 MiB read cap.
    many = [f"dom-{i:07d}.example.test" for i in range(50_000)]
    path.write_text("".join(f"{d}\n" for d in many))
    assert path.stat().st_size > (1 << 20)

    remember_zendriver_domain("fresh.example", path=path)

    result = zendriver_domains(path=path)
    assert "fresh.example" in result
    assert many[-1] in result
    assert len(result) == len(many) + 1


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


def test_read_all_reads_beyond_one_megabyte(tmp_path: Path) -> None:
    path = tmp_path / "domains.txt"
    payload = b"a" * ((1 << 21) + 1)
    path.write_bytes(payload)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        assert transport_routing._read_all(descriptor) == payload
    finally:
        os.close(descriptor)


def test_read_all_uses_one_megabyte_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    chunks = iter((b"first", b""))

    def read(descriptor: int, size: int) -> bytes:
        del descriptor
        calls.append(size)
        return next(chunks)

    monkeypatch.setattr(os, "read", read)

    assert transport_routing._read_all(3) == b"first"
    assert calls == [1 << 20, 1 << 20]


def test_read_domains_uses_close_on_exec_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "domains.txt"
    path.write_text("good.example\n")
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
    assert path.read_text() == "new.example\nold.example\n"


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
