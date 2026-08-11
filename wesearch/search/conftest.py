"""Hermetic fixtures for the search package's tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from wesearch.search.duckduckgo import _duckduckgo_user_agent


@pytest.fixture(autouse=True)
def clear_duckduckgo_user_agent_cache() -> Iterator[None]:
    """Drop the process-stable DuckDuckGo User-Agent between tests.

    ``_duckduckgo_user_agent`` is ``@cache``d on purpose -- DuckDuckGo derives
    its anti-bot token from ``(query, User-Agent)`` and treats a shifting UA as
    a bot. That cache outlives a test, so a test patching ``user_agent_pool``
    reads whatever an earlier test cached instead of its own stub, and which
    test wins depends on collection order.

    Autouse rather than a per-test call: isolation that each test must remember
    to opt into is isolation the next test will forget.
    """
    _duckduckgo_user_agent.cache_clear()
    yield
    _duckduckgo_user_agent.cache_clear()
