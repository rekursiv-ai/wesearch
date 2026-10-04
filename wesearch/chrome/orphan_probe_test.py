"""Tests for the orphan-process probe argument handling."""

from __future__ import annotations

from unittest.mock import patch

import argparse
import sys

import pytest

from wesearch.chrome import orphan_probe
from wesearch.chrome.capture import die_with_parent


def test_main_sleep_mode_does_not_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["orphan_probe.py", "--sleep", "--hang-seconds", "1"],
    )
    with patch("wesearch.chrome.orphan_probe.time.sleep") as sleep:
        assert orphan_probe.main() == 0
    sleep.assert_called_once_with(1.0)


def test_main_help_uses_the_process_probe_description(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["orphan_probe.py", "--help"])
    with pytest.raises(SystemExit, match=r"^0$"):
        orphan_probe.main()
    output = capsys.readouterr().out
    assert "Stand-in for Chrome in the process-reaping tests" in output


def test_main_passes_the_exact_process_probe_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptions: list[str | None] = []
    real_parser = argparse.ArgumentParser()

    def recording_parser(*, description: str | None = None) -> argparse.ArgumentParser:
        descriptions.append(description)
        return real_parser

    monkeypatch.setattr(argparse, "ArgumentParser", recording_parser)
    monkeypatch.setattr(
        sys,
        "argv",
        ["orphan_probe.py", "--sleep", "--hang-seconds", "0"],
    )
    with patch("wesearch.chrome.orphan_probe.time.sleep"):
        assert orphan_probe.main() == 0
    description = orphan_probe.__doc__
    assert description is not None
    assert descriptions == [description.split("\n", 2)[2]]


def test_add_arguments_sets_exact_defaults_and_flags() -> None:
    parser = argparse.ArgumentParser()
    orphan_probe._add_arguments(parser)

    defaults = parser.parse_args([])
    assert vars(defaults) == {"proofed": False, "sleep": False, "hang_seconds": 90.0}
    flags = parser.parse_args(["--proofed", "--sleep", "--hang-seconds", "2.5"])
    assert vars(flags) == {"proofed": True, "sleep": True, "hang_seconds": 2.5}


def test_add_arguments_help_describes_each_flag() -> None:
    parser = argparse.ArgumentParser()
    orphan_probe._add_arguments(parser)
    help_text = parser.format_help()

    assert "--proofed" in help_text
    assert "start the child through subprocess with" in help_text
    assert "--sleep" in help_text
    assert "internal: be the child, and just hang" in help_text
    assert "--hang-seconds HANG_SECONDS" in help_text
    assert "how long each process sleeps before giving up" in help_text


def test_spawn_child_uses_fork_without_proofing() -> None:
    with (
        patch("wesearch.chrome.orphan_probe.os.fork", return_value=1234) as fork,
        patch("wesearch.chrome.orphan_probe.time.sleep") as sleep,
        patch("wesearch.chrome.orphan_probe.os._exit") as exit_process,
    ):
        assert orphan_probe._spawn_child(proofed=False, hang_seconds=2.0) == 1234
    fork.assert_called_once_with()
    sleep.assert_not_called()
    exit_process.assert_not_called()


def test_spawn_child_exits_the_forked_child() -> None:
    with (
        patch("wesearch.chrome.orphan_probe.os.fork", return_value=0),
        patch("wesearch.chrome.orphan_probe.time.sleep") as sleep,
        patch("wesearch.chrome.orphan_probe.os._exit") as exit_process,
    ):
        assert orphan_probe._spawn_child(proofed=False, hang_seconds=2.0) == 0
    sleep.assert_called_once_with(2.0)
    exit_process.assert_called_once_with(0)


def test_spawn_child_proofed_uses_exact_subprocess_command() -> None:
    process = type("Process", (), {"pid": 4321})()
    with patch(
        "wesearch.chrome.orphan_probe.subprocess.Popen",
        return_value=process,
    ) as popen:
        assert orphan_probe._spawn_child(proofed=True, hang_seconds=2.5) == 4321
    assert popen.call_args.args[0] == [
        sys.executable,
        orphan_probe.__file__,
        "--sleep",
        "--hang-seconds",
        "2.5",
    ]
    assert popen.call_args.kwargs["preexec_fn"] is die_with_parent


def test_main_normal_mode_prints_child_and_sleeps(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["orphan_probe.py", "--hang-seconds", "3.5"],
    )
    with (
        patch("wesearch.chrome.orphan_probe._spawn_child", return_value=4567),
        patch("wesearch.chrome.orphan_probe.time.sleep") as sleep,
    ):
        assert orphan_probe.main() == 0
    assert capsys.readouterr().out == "4567\n"
    sleep.assert_called_once_with(3.5)


def test_main_forwards_proofed_flag_and_flushes_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["orphan_probe.py", "--proofed", "--hang-seconds", "2.5"],
    )
    with (
        patch(
            "wesearch.chrome.orphan_probe._spawn_child",
            return_value=4567,
        ) as spawn,
        patch("wesearch.chrome.orphan_probe.time.sleep") as sleep,
        patch("wesearch.chrome.orphan_probe.print") as print_pid,
    ):
        assert orphan_probe.main() == 0
    spawn.assert_called_once_with(proofed=True, hang_seconds=2.5)
    print_pid.assert_called_once_with(4567, flush=True)
    sleep.assert_called_once_with(2.5)


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
