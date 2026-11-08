"""Unit tests for the User-Agent pools."""

from __future__ import annotations

from email.message import Message
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import patch
from urllib import request
from urllib.error import HTTPError, URLError

import gzip
import io
import tempfile

import pytest

from wesearch.chrome import useragents
from wesearch.chrome.headers import ChromePlatform, chrome_user_agent
from wesearch.chrome.useragents import (
    UserAgentKind,
    newest_chrome_major,
    user_agent_pool,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def clear_pool_cache() -> Iterator[None]:
    """Isolate the module-global ``@cache``.

    A ``refresh`` test clears it and a ``_pool_path`` patch can seed tmp content, so
    reset it around every test to keep that state from leaking into other modules under
    xdist.
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
        assert "Mobile" not in ua  # Desktop.

    def test_android_pool_is_nonempty_mobile_chrome(self) -> None:
        pool = user_agent_pool("chrome_android")
        assert pool
        ua = pool[0]
        assert "Chrome" in ua
        assert "Android" in ua

    def test_pools_are_distinct(self) -> None:
        assert set(user_agent_pool("chrome_desktop")).isdisjoint(
            user_agent_pool("chrome_android"),
        )


class TestNewestChromeMajor:
    @pytest.mark.parametrize(
        ("platform", "major"),
        [("Linux", 152), ("Windows", 156), ("Android", 151)],
    )
    def test_is_the_newest_pool_entry_for_the_platform(
        self,
        tmp_path: Path,
        platform: ChromePlatform,
        major: int,
    ) -> None:
        (tmp_path / "chrome_desktop_useragents.txt").write_text(
            f"{chrome_user_agent(150, 'Linux')}\n"
            f"{chrome_user_agent(152, 'Linux')}\n"
            f"{chrome_user_agent(156, 'Windows')}\n"
            "Mozilla/5.0 (X11; CrOS x86_64 1.0) Chrome/160.0.0.0 Safari/537.36\n",
        )
        (tmp_path / "chrome_android_useragents.txt").write_text(
            f"{chrome_user_agent(151, 'Android')}\n",
        )
        with patch.object(useragents, "_CWD", tmp_path):
            assert newest_chrome_major(platform) == major

    def test_shipped_pools_cover_every_platform(self) -> None:
        for platform in ("Windows", "Linux", "macOS", "Android"):
            assert newest_chrome_major(platform) > 0


class TestRefresh:
    """Refresh pool files from one validated intoli dataset snapshot."""

    _DATASET: ClassVar[list[dict[str, str]]] = [
        {  # Desktop Chrome -- kept by desktop, dropped by android.
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
            "deviceCategory": "desktop",
        },
        {  # Android Chrome -- kept by android, dropped by desktop.
            "userAgent": "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 "
            "Mobile Safari/537.36",
            "deviceCategory": "mobile",
        },
        {  # Second desktop identity -- pools must support random selection.
            "userAgent": "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
            "deviceCategory": "desktop",
        },
        {  # Second Android identity -- pools must support random selection.
            "userAgent": "Mozilla/5.0 (Linux; Android 15; Pixel 9) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 "
            "Mobile Safari/537.36",
            "deviceCategory": "mobile",
        },
        {  # Desktop Edge -- dropped by both (not plain Chrome)
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 "
            "Safari/537.36 Edg/149.0.0.0",
            "deviceCategory": "desktop",
        },
        {  # vendor-wrapped Android Chrome -- dropped by both.
            "userAgent": "Mozilla/5.0 (Linux; Android 14; NOH-NX9) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 "
            "Mobile Safari/537.36 HuaweiBrowser/15.0.0.0",
            "deviceCategory": "mobile",
        },
        {  # Facebook in-app browser -- dropped by both.
            "userAgent": "Mozilla/5.0 (Linux; Android 16; motorola razr plus 2025 "
            "Build/W1UXS36H.72-45-10-8-7) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Version/4.0 Chrome/151.0.7922.102 Mobile Safari/537.36 "
            "[FB_IAB/FB4A;FBAV/573.0.0.37.74;IABMV/1;]",
            "deviceCategory": "mobile",
        },
        {  # Embedded newline -- unsafe to serialize as one UA per line.
            "userAgent": "Mozilla/5.0 (Linux; Android 14) Chrome/149.0.0.0\n"
            "Injected/1.0 Mobile Safari/537.36",
            "deviceCategory": "mobile",
        },
    ]

    @pytest.mark.parametrize(
        ("kind", "indices"),
        [("chrome_desktop", [0, 2]), ("chrome_android", [1, 3])],
    )
    def test_selection_keeps_exact_plain_chrome_identities(
        self,
        kind: UserAgentKind,
        indices: list[int],
    ) -> None:
        assert useragents._select_user_agents(list(self._DATASET), kind=kind) == [
            self._DATASET[index]["userAgent"] for index in indices
        ]

    def test_refresh_all_downloads_once_and_rewrites_both_pools(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pool_paths = {
            kind: tmp_path / f"{kind}.txt"
            for kind in ("chrome_desktop", "chrome_android")
        }
        with (
            patch.object(
                useragents,
                "_download_records",
                return_value=self._DATASET,
            ) as download_mock,
            patch.object(useragents, "_pool_path", side_effect=pool_paths.__getitem__),
            caplog.at_level("INFO", logger=useragents.logger.name),
        ):
            useragents.refresh_all()

        download_mock.assert_called_once_with()
        assert [record.getMessage() for record in caplog.records] == [
            f"wrote 2 user agents to {pool_paths['chrome_desktop']}",
            f"wrote 2 user agents to {pool_paths['chrome_android']}",
        ]
        assert len(pool_paths["chrome_desktop"].read_text().splitlines()) == 2
        assert len(pool_paths["chrome_android"].read_text().splitlines()) == 2

    @pytest.mark.parametrize("indices", [[], [0, 0]], ids=["empty", "duplicates"])
    def test_selection_rejects_fewer_than_two_distinct_identities(
        self,
        indices: list[int],
    ) -> None:
        records: list[object] = [self._DATASET[index].copy() for index in indices]
        with pytest.raises(RuntimeError, match="fewer than 2 distinct"):
            useragents._select_user_agents(records, kind="chrome_desktop")

    def test_refresh_does_not_replace_an_invalid_pool(self) -> None:
        invalid = RuntimeError("invalid selection")
        with (
            patch.object(useragents, "_download_records", return_value=[]),
            patch.object(useragents, "_select_user_agents", side_effect=invalid),
            patch.object(useragents, "_replace_pool") as replace,
            pytest.raises(RuntimeError) as raised,
        ):
            useragents.refresh("chrome_desktop")
        assert raised.value is invalid
        replace.assert_not_called()

    def test_refresh_all_validates_both_before_replacing_either_pool(
        self,
        tmp_path: Path,
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
        self,
        tmp_path: Path,
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

    @pytest.mark.parametrize(
        "marker",
        [
            "CriOS/",
            "Edg/",
            "EdgA/",
            "EdgiOS/",
            "FBAN/",
            "FBAV/",
            "FB_IAB/",
            "HeadlessChrome/",
            "HuaweiBrowser/",
            "IABMV/",
            "OPR/",
            "OPT/",
            "SamsungBrowser/",
            "Vivaldi/",
            "YaBrowser/",
        ],
    )
    def test_every_vendor_marker_is_rejected(self, marker: str) -> None:
        ua = f"Mozilla/5.0 Chrome/149.0 {marker}1.0"
        assert useragents._is_plain_chrome(ua) is False

    def test_plain_chrome_is_accepted(self) -> None:
        assert useragents._is_plain_chrome("Mozilla/5.0 Chrome/149.0 Safari/537.36")

    def test_selection_skips_invalid_records_and_preserves_exact_boundaries(
        self,
    ) -> None:
        valid_desktop = self._DATASET[0]["userAgent"]
        second_desktop = self._DATASET[2]["userAgent"]
        records: list[object] = [
            {},
            {"userAgent": None, "deviceCategory": "desktop"},
            {"userAgent": "", "deviceCategory": "desktop"},
            {"userAgent": "  Chrome/1  ", "deviceCategory": "desktop"},
            {"userAgent": "Chrome/1\nInjected", "deviceCategory": "desktop"},
            {"userAgent": valid_desktop, "deviceCategory": "desktop"},
            {"userAgent": second_desktop, "deviceCategory": "desktop"},
        ]
        assert useragents._select_user_agents(records, kind="chrome_desktop") == [
            valid_desktop,
            second_desktop,
        ]

    def test_non_object_dataset_records_are_skipped(self) -> None:
        valid_desktop = self._DATASET[0]["userAgent"]
        second_desktop = self._DATASET[2]["userAgent"]
        records: list[object] = [
            "stray",
            3,
            None,
            [1],
            {"userAgent": valid_desktop, "deviceCategory": "desktop"},
            {"userAgent": second_desktop, "deviceCategory": "desktop"},
        ]
        assert useragents._select_user_agents(records, kind="chrome_desktop") == [
            valid_desktop,
            second_desktop,
        ]

    def test_desktop_mobile_and_android_markers_are_case_sensitive(self) -> None:
        base = self._DATASET[0]["userAgent"]
        second = self._DATASET[2]["userAgent"]
        exact_mobile = base + " Mobile"
        lowercase_mobile = base + " mobile"
        exact_android = base + " Android"
        lowercase_android = base + " android"
        uppercase_android = base + " ANDROID"
        records: list[object] = [
            {"userAgent": exact_mobile, "deviceCategory": "desktop"},
            {"userAgent": lowercase_mobile, "deviceCategory": "desktop"},
            {"userAgent": exact_android, "deviceCategory": "desktop"},
            {"userAgent": lowercase_android, "deviceCategory": "desktop"},
            {"userAgent": uppercase_android, "deviceCategory": "desktop"},
            {"userAgent": second, "deviceCategory": "desktop"},
        ]
        assert useragents._select_user_agents(records, kind="chrome_desktop") == sorted(
            [lowercase_mobile, lowercase_android, uppercase_android, second],
        )

    def test_android_tablet_is_kept_but_android_ten_k_is_rejected(self) -> None:
        tablet = self._DATASET[1]["userAgent"].replace("Android 14", "Android 15")
        android_ten_k = tablet.replace("Android 15; Pixel 8", "Android 10; K")
        records: list[object] = [
            {"userAgent": tablet, "deviceCategory": "tablet"},
            {"userAgent": self._DATASET[3]["userAgent"], "deviceCategory": "mobile"},
            {"userAgent": android_ten_k, "deviceCategory": "mobile"},
        ]
        assert useragents._select_user_agents(records, kind="chrome_android") == sorted(
            [tablet, self._DATASET[3]["userAgent"]],
        )

    def test_refresh_forwards_exact_kind_and_selected_pool(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pool_file = tmp_path / "pool.txt"
        selected = ["a", "b"]
        with (
            patch.object(useragents, "_download_records", return_value=[]),
            patch.object(
                useragents,
                "_select_user_agents",
                return_value=selected,
            ) as select,
            patch.object(useragents, "_replace_pool") as replace,
            patch.object(useragents, "_pool_path", return_value=pool_file) as pool_path,
            caplog.at_level("INFO", logger=useragents.logger.name),
        ):
            useragents.refresh("chrome_android")
        select.assert_called_once_with([], kind="chrome_android")
        replace.assert_called_once_with("chrome_android", selected)
        pool_path.assert_called_with("chrome_android")
        assert caplog.records[0].getMessage() == f"wrote 2 user agents to {pool_file}"

    def test_replace_uses_exact_default_mode_when_new(self, tmp_path: Path) -> None:
        pool_file = tmp_path / "pool.txt"
        with patch.object(useragents, "_pool_path", return_value=pool_file):
            useragents._replace_pool("chrome_desktop", ["a", "b"])
        assert pool_file.read_text() == "a\nb\n"
        assert pool_file.stat().st_mode & 0o777 == 0o644
        assert set(tmp_path.iterdir()) == {pool_file}

    def test_replace_preserves_existing_mode(self, tmp_path: Path) -> None:
        pool_file = tmp_path / "pool.txt"
        pool_file.write_text("old\n")
        pool_file.chmod(0o600)
        with patch.object(useragents, "_pool_path", return_value=pool_file):
            useragents._replace_pool("chrome_desktop", ["a", "b"])
        assert pool_file.read_text() == "a\nb\n"
        assert pool_file.stat().st_mode & 0o777 == 0o600

    def test_restore_handles_absent_and_backup_pools_exactly(
        self,
        tmp_path: Path,
    ) -> None:
        pool_file = tmp_path / "pool.txt"
        with patch.object(useragents, "_pool_path", return_value=pool_file):
            useragents._restore_pool("chrome_desktop", None)
            pool_file.write_text("restored\n")
            backup = tmp_path / ".pool.txt.bak"
            backup.write_text("backup\n")
            useragents._restore_pool("chrome_desktop", backup)
        assert pool_file.read_text() == "backup\n"
        assert not backup.exists()

    def test_backup_does_not_unlink_before_tempfile_exists(
        self,
        tmp_path: Path,
    ) -> None:
        pool_file = tmp_path / "pool.txt"
        pool_file.write_text("old\n")
        with (
            patch.object(useragents, "_pool_path", return_value=pool_file),
            patch.object(
                tempfile,
                "NamedTemporaryFile",
                side_effect=OSError("open failed"),
            ),
            pytest.raises(OSError, match=r"^open failed$"),
        ):
            useragents._backup_pool("chrome_desktop")

    def test_backup_cleans_partial_file_when_metadata_copy_fails(
        self,
        tmp_path: Path,
    ) -> None:
        pool_file = tmp_path / "pool.txt"
        pool_file.write_text("old\n")
        original_unlink = Path.unlink

        def unlink(path: Path, *, missing_ok: bool = False) -> None:
            if missing_ok:
                original_unlink(path)
            else:
                raise FileNotFoundError

        with (
            patch.object(useragents, "_pool_path", return_value=pool_file),
            patch.object(Path, "chmod", side_effect=OSError("chmod failed")),
            patch.object(Path, "unlink", unlink),
            pytest.raises(OSError, match=r"^chmod failed$"),
        ):
            useragents._backup_pool("chrome_desktop")
        assert list(tmp_path.glob("*.bak")) == []

    def test_replace_and_backup_use_exact_atomic_tempfile_shapes(
        self,
        tmp_path: Path,
    ) -> None:
        pool_file = tmp_path / "pool.txt"
        pool_file.write_text("old\n")
        with patch.object(useragents, "_pool_path", return_value=pool_file):
            with patch.object(
                tempfile,
                "NamedTemporaryFile",
                wraps=tempfile.NamedTemporaryFile,
            ) as temporary:
                useragents._replace_pool("chrome_desktop", ["a", "b"])
            replace_call = temporary.call_args
            assert replace_call.kwargs == {
                "mode": "w",
                "encoding": "utf-8",
                "dir": tmp_path,
                "prefix": ".pool.txt.",
                "suffix": ".tmp",
                "delete": False,
            }
            with patch.object(
                tempfile,
                "NamedTemporaryFile",
                wraps=tempfile.NamedTemporaryFile,
            ) as temporary:
                backup = useragents._backup_pool("chrome_desktop")
            assert backup is not None
            assert temporary.call_args.kwargs == {
                "mode": "wb",
                "dir": tmp_path,
                "prefix": ".pool.txt.",
                "suffix": ".bak",
                "delete": False,
            }
            backup.unlink()


class TestDownload:
    """The downloader parses gzip JSON and retries only transient failures."""

    _PAYLOAD: ClassVar[bytes] = gzip.compress(b'[{"userAgent":"Chrome/149"}]')

    def test_download_builds_exact_request_and_refresh_identity(self) -> None:
        sent_request = object()
        with (
            patch.object(request, "Request", return_value=sent_request) as factory,
            patch.object(useragents, "_refresh_user_agent", return_value="UA") as ua,
            patch.object(
                useragents,
                "_read_response",
                side_effect=[
                    self._PAYLOAD,
                    AssertionError("retried a successful read"),
                ],
            ) as read,
        ):
            assert useragents._download_records() == [{"userAgent": "Chrome/149"}]
        ua.assert_called_once_with("chrome_desktop")
        factory.assert_called_once_with(
            "https://raw.githubusercontent.com/intoli/user-agents/main/src/user-agents.json.gz",
            headers={"User-Agent": "UA"},
        )
        read.assert_called_once_with(sent_request)

    @pytest.mark.parametrize(
        ("kind", "platform"),
        [("chrome_desktop", "Linux"), ("chrome_android", "Android")],
    )
    def test_refresh_user_agent_is_the_newest_chrome_for_its_kind(
        self,
        kind: UserAgentKind,
        platform: ChromePlatform,
    ) -> None:
        with patch.object(
            useragents,
            "newest_chrome_major",
            return_value=151,
        ) as newest:
            assert useragents._refresh_user_agent(kind) == chrome_user_agent(
                151,
                platform,
            )
        newest.assert_called_once_with(platform)

    def test_read_response_uses_the_request_and_closes_its_body(self) -> None:
        sent = request.Request("https://example.test/user-agents.json.gz")
        response = io.BytesIO(b"compressed body")
        with patch.object(request, "urlopen", return_value=response) as opened:
            assert useragents._read_response(sent) == b"compressed body"
        opened.assert_called_once_with(sent, timeout=30)
        assert response.closed

    @pytest.mark.parametrize("failure", [URLError, TimeoutError, 429, 500, 503, 599])
    def test_download_retries_transient_failures(
        self,
        failure: int | type[OSError],
    ) -> None:
        error = _download_failure(failure)
        with patch.object(
            useragents,
            "_read_response",
            side_effect=[
                error,
                self._PAYLOAD,
                AssertionError("retried a successful read"),
            ],
        ) as read:
            assert useragents._download_records() == [{"userAgent": "Chrome/149"}]
        assert read.call_count == 2
        assert read.call_args_list[0] == read.call_args_list[1]

    @pytest.mark.parametrize("failure", [URLError, TimeoutError, 429, 503])
    def test_download_stops_after_three_transient_failures(
        self,
        failure: int | type[OSError],
    ) -> None:
        error = _download_failure(failure)
        with (
            patch.object(
                useragents,
                "_read_response",
                side_effect=[
                    error,
                    error,
                    error,
                    AssertionError("exceeded three reads"),
                ],
            ) as read,
            pytest.raises(type(error)) as raised,
        ):
            useragents._download_records()
        assert raised.value is error
        assert read.call_count == 3
        assert read.call_args_list == [read.call_args_list[0]] * 3

    @pytest.mark.parametrize("status", [404, 499, 600])
    def test_download_does_not_retry_other_http_errors(self, status: int) -> None:
        error = _download_failure(status)
        with (
            patch.object(
                useragents,
                "_read_response",
                side_effect=[error, AssertionError("retried a permanent failure")],
            ) as read,
            pytest.raises(HTTPError) as raised,
        ):
            useragents._download_records()
        assert raised.value is error
        read.assert_called_once()

    def test_download_rejects_non_array_json(self) -> None:
        payload = gzip.compress(b'{"userAgent":"Chrome/149"}')
        with (
            patch.object(
                useragents,
                "_read_response",
                side_effect=[payload, AssertionError("retried a malformed body")],
            ) as read,
            pytest.raises(RuntimeError, match=r"^expected JSON array$"),
        ):
            useragents._download_records()
        read.assert_called_once()


def _download_failure(failure: int | type[OSError]) -> OSError:
    if isinstance(failure, int):
        return HTTPError(
            "https://example.test/user-agents.json.gz",
            failure,
            "refused",
            hdrs=Message(),
            fp=None,
        )
    return failure("temporary")


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
