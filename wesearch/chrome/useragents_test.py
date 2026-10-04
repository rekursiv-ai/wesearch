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
import json
import tempfile

import pytest

from wesearch.chrome import useragents
from wesearch.chrome.useragents import (
    UserAgentKind,
    draw_user_agent,
    impersonate_target,
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

    def _refresh(self, kind: UserAgentKind, tmp_path: Path) -> list[str]:
        pool_file = tmp_path / f"{kind}.txt"
        with (
            patch.object(useragents, "_download_records", return_value=self._DATASET),
            patch.object(useragents, "_pool_path", return_value=pool_file),
        ):
            useragents.refresh(kind)
        return pool_file.read_text().splitlines()

    def test_desktop_filter_keeps_only_desktop_plain_chrome(
        self,
        tmp_path: Path,
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

    def test_refresh_rejects_fewer_than_two_distinct_identities(
        self,
        tmp_path: Path,
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

    def test_pool_replacement_does_not_leave_temporary_file(
        self,
        tmp_path: Path,
    ) -> None:
        pool_file = tmp_path / "chrome_desktop.txt"
        with (
            patch.object(useragents, "_download_records", return_value=self._DATASET),
            patch.object(useragents, "_pool_path", return_value=pool_file),
        ):
            useragents.refresh("chrome_desktop")

        assert not list(tmp_path.glob("*.tmp"))

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

    def test_refresh_logs_exact_count_and_path(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pool_file = tmp_path / "pool.txt"
        with (
            patch.object(useragents, "_download_records", return_value=self._DATASET),
            patch.object(useragents, "_pool_path", return_value=pool_file),
            caplog.at_level("INFO", logger=useragents.logger.name),
        ):
            useragents.refresh("chrome_desktop")
        assert [record.getMessage() for record in caplog.records] == [
            f"wrote 2 user agents to {pool_file}",
        ]

    def test_replace_uses_exact_default_mode_when_new(self, tmp_path: Path) -> None:
        pool_file = tmp_path / "pool.txt"
        with patch.object(useragents, "_pool_path", return_value=pool_file):
            useragents._replace_pool("chrome_desktop", ["a", "b"])
        assert pool_file.stat().st_mode & 0o777 == 0o644

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
    """The stdlib downloader parses gzip JSON and retries transient failures."""

    _PAYLOAD = gzip.compress(json.dumps(TestRefresh._DATASET).encode())

    def test_download_builds_exact_request_and_refresh_identity(self) -> None:
        sent_request = object()
        with (
            patch.object(
                request,
                "Request",
                return_value=sent_request,
            ) as request_factory,
            patch.object(
                useragents,
                "_refresh_user_agent",
                return_value="UA",
            ) as refresh_ua,
            patch.object(useragents, "_read_response", return_value=self._PAYLOAD),
        ):
            assert useragents._download_records() == TestRefresh._DATASET
        refresh_ua.assert_called_once_with("chrome_desktop")
        request_factory.assert_called_once_with(
            "https://raw.githubusercontent.com/intoli/user-agents/main/src/user-agents.json.gz",
            headers={"User-Agent": "UA"},
        )

    def test_refresh_user_agent_forwards_exact_kind_mapping(self) -> None:
        with (
            patch.object(
                useragents,
                "impersonate_target",
                return_value="chrome_android",
            ) as target,
            patch.object(
                useragents,
                "impersonate_version_platform",
                return_value=(131, "Android"),
            ) as version,
            patch.object(useragents, "chrome_user_agent", return_value="UA") as ua,
        ):
            assert useragents._refresh_user_agent("chrome_android") == "UA"
        target.assert_called_once_with("chrome_android")
        version.assert_called_once_with("chrome_android")
        ua.assert_called_once_with(131, "Android")

    def test_download_uses_stdlib_with_fixed_identity(self) -> None:
        with patch.object(
            request,
            "urlopen",
            return_value=io.BytesIO(self._PAYLOAD),
        ) as urlopen_mock:
            records = useragents._download_records()

        assert records == TestRefresh._DATASET
        sent_request = urlopen_mock.call_args.args[0]
        assert isinstance(sent_request, request.Request)
        assert sent_request.full_url == (
            "https://raw.githubusercontent.com/intoli/user-agents/"
            "main/src/user-agents.json.gz"
        )
        assert sent_request.get_header("User-agent") == useragents._refresh_user_agent(
            "chrome_desktop",
        )
        assert urlopen_mock.call_args.kwargs == {"timeout": 30}

    def test_download_retries_transient_url_error(self) -> None:
        with patch.object(
            request,
            "urlopen",
            side_effect=[URLError("temporary"), io.BytesIO(self._PAYLOAD)],
        ) as urlopen_mock:
            assert useragents._download_records() == TestRefresh._DATASET

        assert urlopen_mock.call_count == 2

    @pytest.mark.parametrize("status", [429, 500, 503, 599])
    def test_download_retries_transient_http_error(self, status: int) -> None:
        error = HTTPError(
            "https://example.test/user-agents.json.gz",
            status,
            "Transient",
            hdrs=Message(),
            fp=None,
        )
        with patch.object(
            request,
            "urlopen",
            side_effect=[error, io.BytesIO(self._PAYLOAD)],
        ) as urlopen_mock:
            assert useragents._download_records() == TestRefresh._DATASET

        assert urlopen_mock.call_count == 2

    @pytest.mark.parametrize("status", [429, 503])
    def test_download_stops_after_three_transient_http_errors(
        self,
        status: int,
    ) -> None:
        error = HTTPError(
            "https://example.test/user-agents.json.gz",
            status,
            "Transient",
            hdrs=Message(),
            fp=None,
        )
        with (
            patch.object(request, "urlopen", side_effect=error) as urlopen_mock,
            pytest.raises(HTTPError) as raised,
        ):
            useragents._download_records()

        assert raised.value.code == status
        assert urlopen_mock.call_count == 3

    def test_download_does_not_attempt_fourth_transient_request(self) -> None:
        error = HTTPError(
            "https://example.test/user-agents.json.gz",
            503,
            "Transient",
            hdrs=Message(),
            fp=None,
        )
        with (
            patch.object(
                request,
                "urlopen",
                side_effect=[error, error, error, io.BytesIO(self._PAYLOAD)],
            ) as urlopen_mock,
            pytest.raises(HTTPError),
        ):
            useragents._download_records()
        assert urlopen_mock.call_count == 3

    def test_download_stops_after_three_transient_failures(self) -> None:
        with (
            patch.object(
                request,
                "urlopen",
                side_effect=URLError("temporary"),
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
            patch.object(request, "urlopen", side_effect=error) as urlopen_mock,
            pytest.raises(HTTPError),
        ):
            useragents._download_records()

        urlopen_mock.assert_called_once()

    @pytest.mark.parametrize("status", [499, 600])
    def test_download_does_not_retry_boundary_http_errors(self, status: int) -> None:
        error = HTTPError(
            "https://example.test/user-agents.json.gz",
            status,
            "Error",
            hdrs=Message(),
            fp=None,
        )
        with (
            patch.object(request, "urlopen", side_effect=error) as urlopen_mock,
            pytest.raises(HTTPError),
        ):
            useragents._download_records()
        urlopen_mock.assert_called_once()

    def test_download_rejects_non_array_json(self) -> None:
        payload = gzip.compress(json.dumps({"userAgent": "Chrome/149"}).encode())
        with (
            patch.object(request, "urlopen", return_value=io.BytesIO(payload)),
            pytest.raises(RuntimeError, match=r"^expected JSON array$"),
        ):
            useragents._download_records()


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
