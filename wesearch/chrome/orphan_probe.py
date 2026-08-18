#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")/../../.." run --frozen --no-sync python3 "$0" "$@"
Stand-in for Chrome in the process-reaping tests: spawn a child, then hang.

``capture_test`` asserts two properties of a spawned browser -- that killing its
process group takes the processes it started with it, and that the kernel kills
it when its parent dies. Both need a real process tree, and Chrome is neither
required (the mechanism is POSIX process groups and PR_SET_PDEATHSIG, which
Chrome only happens to use) nor installed on every host.

Prints the child's PID on stdout, so the caller watches that exact PID rather
than scanning for one.
'''
# fmt: on

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from wesearch.chrome.capture import die_with_parent


def main() -> int:
    """Start a child that hangs, print its PID, then hang until killed."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 2)[2])
    _add_arguments(parser)
    args = parser.parse_args()

    if args.sleep:
        time.sleep(args.hang_seconds)
        return 0
    child = _spawn_child(proofed=args.proofed, hang_seconds=args.hang_seconds)
    print(child, flush=True)  # noqa: T201 -- this line IS the tool's output.
    time.sleep(args.hang_seconds)
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--proofed",
        action="store_true",
        help=(
            "start the child through subprocess with die_with_parent, the way a "
            "browser is launched; otherwise a plain fork() with nothing armed"
        ),
    )
    parser.add_argument(
        "--sleep",
        action="store_true",
        help="internal: be the child, and just hang",
    )
    parser.add_argument(
        "--hang-seconds",
        type=float,
        default=90.0,
        # Longer than any caller's timeout on purpose: each process must be
        # killed by the test, so its death is evidence about the kill rather
        # than about the sleep expiring.
        help="how long each process sleeps before giving up",
    )


def _spawn_child(*, proofed: bool, hang_seconds: float) -> int:
    """Start the hanging child and return its PID.

    Args:
      proofed: Launch through ``subprocess`` with ``die_with_parent``, matching
        how a browser is started. The unproofed arm forks WITHOUT arming, so it
        stands in for the pre-fix state; the parent-death signal is armed on the
        forking thread, so only the proofed arm arms it against a thread that
        outlives the child.
      hang_seconds: How long the child sleeps.

    Returns:
      pid: The child's process id.

    """
    if not proofed:
        child = os.fork()
        if child == 0:
            time.sleep(hang_seconds)
            # ``os._exit``: the child must not run atexit handlers it inherited
            # or re-flush the parent's stdout buffer.
            os._exit(0)
        return child
    return subprocess.Popen(  # noqa: S603 -- fixed argv, interpreter from sys.
        [sys.executable, __file__, "--sleep", "--hang-seconds", str(hang_seconds)],
        preexec_fn=die_with_parent,  # noqa: PLW1509 -- bare syscalls only; takes no lock a forked thread could hold.
    ).pid


if __name__ == "__main__":
    sys.exit(main())
# vim: ft=python
