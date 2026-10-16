"""Tests for the connections a fetch session owns."""

from __future__ import annotations

from wesearch.fetch.types import FetchSession


class _Connection:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _Browsers:
    def __init__(self) -> None:
        self.shutdowns = 0

    def shutdown(self) -> None:
        self.shutdowns += 1


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
