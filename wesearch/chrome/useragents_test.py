"""Unit tests for the User-Agent pools."""

from __future__ import annotations

from collections.abc import Iterator
from email.message import Message
from pathlib import Path
from typing import ClassVar, cast
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import gzip
import io
import json
import urllib.request

import pytest

from wesearch.chrome import useragents
from wesearch.chrome.useragents import (
    UserAgentKind,
    draw_user_agent,
    impersonate_target,
    user_agent_pool,
)


@pytest.fixture(autouse=True)
def clear_pool_cache() -> Iterator[None]:
    """Isolate the module-global ``@cache``: a ``refresh`` test clears it and a
    ``_pool_path`` patch can seed tmp content, so reset it around every test to
    keep that state from leaking into other modules under xdist.
    """
    user_agent_pool.cache_clear()
    yield
    user_agent_pool.cache_clear()


class TestPools:
    def test_desktop_pool_is_nonempty_desktop_chrome(self) -> None:
        pool = user_agent_pool("chrome_desktop")
        assert pool
        ua = pool[0]
        assert "Chrome" in ua
        assert "Mobile" not in ua  # desktop

    def test_android_pool_is_nonempty_mobile_chrome(self) -> None:
        pool = user_agent_pool("chrome_android")
        assert pool
        ua = pool[0]
        assert "Chrome" in ua
        assert "Android" in ua

    def test_pools_are_distinct(self) -> None:
        assert set(user_agent_pool("chrome_desktop")).isdisjoint(
            user_agent_pool("chrome_android")
        )


class TestDraw:
    def test_draws_from_the_requested_pool(self) -> None:
        assert draw_user_agent("chrome_desktop") in user_agent_pool("chrome_desktop")
        assert draw_user_agent("chrome_android") in user_agent_pool("chrome_android")

    @pytest.mark.parametrize("kind", ["chrome_desktop", "chrome_android"])
    def test_draw_delegates_to_rng_choice(self, kind: UserAgentKind) -> None:
        pool = user_agent_pool(kind)
        with patch.object(useragents._RNG, "choice", return_value=pool[-1]) as choice:
            assert draw_user_agent(kind) == pool[-1]

        choice.assert_called_once_with(pool)


class TestImpersonateTarget:
    def test_desktop_maps_to_chrome(self) -> None:
        assert impersonate_target("chrome_desktop") == "chrome"

    def test_android_maps_to_chrome_android(self) -> None:
        assert impersonate_target("chrome_android") == "chrome_android"

    def test_kind_for_impersonate_is_the_inverse(self) -> None:
        # The impersonate<->kind bijection has ONE source of truth: the inverse
        # must round-trip both kinds, so fetch.py can call it instead of
        # inlining a parallel (drift-prone) mapping.
        kind_for_impersonate = useragents.kind_for_impersonate
        for kind in ("chrome_desktop", "chrome_android"):
            assert kind_for_impersonate(impersonate_target(kind)) == kind
        # An unknown impersonate target degrades to desktop.
        assert kind_for_impersonate("chrome") == "chrome_desktop"


class TestRefresh:
    """Refresh pool files from one validated intoli dataset snapshot."""

    _DATASET: ClassVar[list[dict[str, str]]] = [
        {  # desktop Chrome -- kept by desktop, dropped by android
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
            "deviceCategory": "desktop",
        },
        {  # android Chrome -- kept by android, dropped by desktop
            "userAgent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 "
            "Mobile Safari/537.36",
            "deviceCategory": "mobile",
        },
        {  # second desktop identity -- pools must support random selection
            "userAgent": "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
            "deviceCategory": "desktop",
        },
        {  # second Android identity -- pools must support random selection
            "userAgent": "Mozilla/5.0 (Linux; Android 15; Pixel 9) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 "
            "Mobile Safari/537.36",
            "deviceCategory": "mobile",
        },
        {  # desktop Edge -- dropped by both (not plain Chrome)
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 "
            "Safari/537.36 Edg/149.0.0.0",
            "deviceCategory": "desktop",
        },
        {  # vendor-wrapped Android Chrome -- dropped by both
            "userAgent": "Mozilla/5.0 (Linux; Android 14; NOH-NX9) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 "
            "Mobile Safari/537.36 HuaweiBrowser/15.0.0.0",
            "deviceCategory": "mobile",
        },
        {  # embedded newline -- unsafe to serialize as one UA per line
            "userAgent": "Mozilla/5.0 (Linux; Android 14) Chrome/149.0.0.0\n"
            "Injected/1.0 Mobile Safari/537.36",
            "deviceCategory": "mobile",
        },
    ]

    def _refresh(self, kind: str, tmp_path: Path) -> list[str]:
        pool_file = tmp_path / f"{kind}.txt"
        with (
            patch.object(useragents, "_download_records", return_value=self._DATASET),
            patch.object(useragents, "_pool_path", return_value=pool_file),
        ):
            useragents.refresh(cast("UserAgentKind", kind))
        return pool_file.read_text().splitlines()

    def test_desktop_filter_keeps_only_desktop_plain_chrome(
        self, tmp_path: Path
    ) -> None:
        lines = self._refresh("chrome_desktop", tmp_path)
        assert len(lines) == 2
        assert all("Mobile" not in line for line in lines)
        assert all("Edg/" not in line for line in lines)

    def test_android_filter_keeps_only_android_chrome(self, tmp_path: Path) -> None:
        lines = self._refresh("chrome_android", tmp_path)
        assert len(lines) == 2
        assert all("Android" in line for line in lines)

    def test_empty_result_raises(self, tmp_path: Path) -> None:
        pool_file = tmp_path / "chrome_desktop.txt"
        with (
            patch.object(useragents, "_download_records", return_value=[]),
            patch.object(useragents, "_pool_path", return_value=pool_file),
            pytest.raises(RuntimeError, match="fewer than 2 distinct"),
        ):
            useragents.refresh("chrome_desktop")

    def test_refresh_all_downloads_once_and_rewrites_both_pools(
        self, tmp_path: Path
    ) -> None:
        pool_paths = {
            kind: tmp_path / f"{kind}.txt"
            for kind in ("chrome_desktop", "chrome_android")
        }
        with (
            patch.object(
                useragents, "_download_records", return_value=self._DATASET
            ) as download_mock,
            patch.object(useragents, "_pool_path", side_effect=pool_paths.__getitem__),
        ):
            useragents.refresh_all()

        download_mock.assert_called_once_with()
        assert len(pool_paths["chrome_desktop"].read_text().splitlines()) == 2
        assert len(pool_paths["chrome_android"].read_text().splitlines()) == 2

    def test_refresh_rejects_fewer_than_two_distinct_identities(
        self, tmp_path: Path
    ) -> None:
        pool_file = tmp_path / "chrome_desktop.txt"
        duplicate_only = [self._DATASET[0], self._DATASET[0].copy()]
        with (
            patch.object(useragents, "_download_records", return_value=duplicate_only),
            patch.object(useragents, "_pool_path", return_value=pool_file),
            pytest.raises(RuntimeError, match="fewer than 2 distinct"),
        ):
            useragents.refresh("chrome_desktop")

        assert not pool_file.exists()

    def test_refresh_all_validates_both_before_replacing_either_pool(
        self, tmp_path: Path
    ) -> None:
        pool_paths = {
            kind: tmp_path / f"{kind}.txt"
            for kind in ("chrome_desktop", "chrome_android")
        }
        for pool_path in pool_paths.values():
            pool_path.write_text("original\n")
        desktop_only = [self._DATASET[0], self._DATASET[2]]

        with (
            patch.object(useragents, "_download_records", return_value=desktop_only),
            patch.object(useragents, "_pool_path", side_effect=pool_paths.__getitem__),
            pytest.raises(RuntimeError, match="chrome_android"),
        ):
            useragents.refresh_all()

        assert pool_paths["chrome_desktop"].read_text() == "original\n"
        assert pool_paths["chrome_android"].read_text() == "original\n"

    def test_refresh_all_rolls_back_when_second_replacement_fails(
        self, tmp_path: Path
    ) -> None:
        pool_paths = {
            kind: tmp_path / f"{kind}.txt"
            for kind in ("chrome_desktop", "chrome_android")
        }
        originals = {
            "chrome_desktop": "original desktop one\noriginal desktop two\n",
            "chrome_android": "original android one\noriginal android two\n",
        }
        for kind, pool_path in pool_paths.items():
            pool_path.write_text(originals[kind])
        original_replace = Path.replace

        def fail_android_temporary_replace(source: Path, target: Path) -> Path:
            if source.suffix == ".tmp" and target == pool_paths["chrome_android"]:
                raise OSError("injected second replacement failure")
            return original_replace(source, target)

        with patch.object(useragents, "_pool_path", side_effect=pool_paths.__getitem__):
            assert user_agent_pool("chrome_desktop")
            assert user_agent_pool("chrome_android")
        assert user_agent_pool.cache_info().currsize == 2

        with (
            patch.object(useragents, "_download_records", return_value=self._DATASET),
            patch.object(useragents, "_pool_path", side_effect=pool_paths.__getitem__),
            patch.object(
                Path,
                "replace",
                autospec=True,
                side_effect=fail_android_temporary_replace,
            ),
            pytest.raises(OSError, match="injected second replacement failure"),
        ):
            useragents.refresh_all()

        assert pool_paths["chrome_desktop"].read_text() == originals["chrome_desktop"]
        assert pool_paths["chrome_android"].read_text() == originals["chrome_android"]
        assert user_agent_pool.cache_info().currsize == 0
        assert set(tmp_path.iterdir()) == set(pool_paths.values())

    def test_pool_replacement_does_not_leave_temporary_file(
        self, tmp_path: Path
    ) -> None:
        pool_file = tmp_path / "chrome_desktop.txt"
        with (
            patch.object(useragents, "_download_records", return_value=self._DATASET),
            patch.object(useragents, "_pool_path", return_value=pool_file),
        ):
            useragents.refresh("chrome_desktop")

        assert not list(tmp_path.glob("*.tmp"))


class TestDownload:
    """The stdlib downloader parses gzip JSON and retries transient failures."""

    _PAYLOAD = gzip.compress(json.dumps(TestRefresh._DATASET).encode())

    def test_download_uses_stdlib_with_fixed_identity(self) -> None:
        with patch.object(
            urllib.request,
            "urlopen",
            return_value=io.BytesIO(self._PAYLOAD),
        ) as urlopen_mock:
            records = useragents._download_records()

        assert records == TestRefresh._DATASET
        request = urlopen_mock.call_args.args[0]
        assert request.get_header("User-agent")
        assert urlopen_mock.call_args.kwargs == {"timeout": 30}

    def test_download_retries_transient_url_error(self) -> None:
        with patch.object(
            urllib.request,
            "urlopen",
            side_effect=[URLError("temporary"), io.BytesIO(self._PAYLOAD)],
        ) as urlopen_mock:
            assert useragents._download_records() == TestRefresh._DATASET

        assert urlopen_mock.call_count == 2

    @pytest.mark.parametrize("status", [429, 503])
    def test_download_retries_transient_http_error(self, status: int) -> None:
        error = HTTPError(
            "https://example.test/user-agents.json.gz",
            status,
            "Transient",
            hdrs=Message(),
            fp=None,
        )
        with patch.object(
            urllib.request,
            "urlopen",
            side_effect=[error, io.BytesIO(self._PAYLOAD)],
        ) as urlopen_mock:
            assert useragents._download_records() == TestRefresh._DATASET

        assert urlopen_mock.call_count == 2

    @pytest.mark.parametrize("status", [429, 503])
    def test_download_stops_after_three_transient_http_errors(
        self, status: int
    ) -> None:
        error = HTTPError(
            "https://example.test/user-agents.json.gz",
            status,
            "Transient",
            hdrs=Message(),
            fp=None,
        )
        with (
            patch.object(urllib.request, "urlopen", side_effect=error) as urlopen_mock,
            pytest.raises(HTTPError) as raised,
        ):
            useragents._download_records()

        assert raised.value.code == status
        assert urlopen_mock.call_count == 3

    def test_download_stops_after_three_transient_failures(self) -> None:
        with (
            patch.object(
                urllib.request, "urlopen", side_effect=URLError("temporary")
            ) as urlopen_mock,
            pytest.raises(URLError),
        ):
            useragents._download_records()

        assert urlopen_mock.call_count == 3

    def test_download_does_not_retry_other_client_http_error(self) -> None:
        error = HTTPError(
            "https://example.test/user-agents.json.gz",
            404,
            "Not Found",
            hdrs=Message(),
            fp=None,
        )
        with (
            patch.object(urllib.request, "urlopen", side_effect=error) as urlopen_mock,
            pytest.raises(HTTPError),
        ):
            useragents._download_records()

        urlopen_mock.assert_called_once()

    def test_download_rejects_non_array_json(self) -> None:
        payload = gzip.compress(json.dumps({"userAgent": "Chrome/149"}).encode())
        with (
            patch.object(urllib.request, "urlopen", return_value=io.BytesIO(payload)),
            pytest.raises(RuntimeError, match="expected JSON array"),
        ):
            useragents._download_records()


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
