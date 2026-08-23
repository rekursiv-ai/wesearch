"""Every registered extractor satisfies the ``Extract`` protocol."""

from __future__ import annotations

from typing import get_args

import pytest

from wesearch.types.extractor import Extract
from wesearch.types.params import Extractor
from wesearch.web import _EXTRACTORS


def test_every_extractor_name_is_registered() -> None:
    """The registry covers the vocabulary, so no name dispatches to a KeyError."""
    assert set(get_args(Extractor)) == set(_EXTRACTORS)


@pytest.mark.compute_large_fixture
def test_every_extractor_satisfies_the_protocol() -> None:
    """Each implementation is callable with the protocol's exact signature."""
    for extract in _EXTRACTORS.values():
        checked: Extract = extract
        assert isinstance(checked("<html><body>hi</body></html>", url="http://x/"), str)


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
