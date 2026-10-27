"""Tests for the connections a fetch session owns."""

from __future__ import annotations

import asyncio

import pytest

from wesearch.fetch.types import FetchSession


class _Connection:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _Browsers:
    def __init__(self) -> None:
        self.shutdowns = 0
        self.under_a_loop: list[bool] = []

    def shutdown(self) -> None:
        self.shutdowns += 1
        self.under_a_loop.append(_loop_running())


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_a_session_used_with_with_closes_on_exit() -> None:
    browsers = _Browsers()
    with FetchSession(browsers=browsers) as session:
        assert browsers.shutdowns == 0
    assert browsers.shutdowns == 1
    assert session.browsers is None


@pytest.mark.asyncio
async def test_a_session_used_with_async_with_closes_off_the_running_loop() -> None:
    browsers = _Browsers()
    async with FetchSession(browsers=browsers) as session:
        assert browsers.shutdowns == 0
    assert browsers.under_a_loop == [False]
    assert session.browsers is None


def test_close_releases_every_connection_once_and_empties_the_session() -> None:
    first, second = _Connection(), _Connection()
    browsers = _Browsers()
    session = FetchSession(
        curl={
            ("egress", "first.example", None, 443): first,
            ("egress", "second.example", None, 443): second,
        },
        browsers=browsers,
    )

    session.close()
    session.close()

    assert first.closed == second.closed == browsers.shutdowns == 1
    assert session.curl == {}
    assert session.browsers is None


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
