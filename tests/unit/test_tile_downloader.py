"""TileDownloader 单元测试."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from seehydro.acquisition.tile_downloader import TileDownloader


def test_request_interval_显式传入时可覆盖_provider默认值() -> None:
    downloader = TileDownloader(provider="tianditu_img", api_key="dummy-key", request_interval=0.0)
    assert downloader.request_interval == 0.0

    downloader2 = TileDownloader(provider="tianditu_img", api_key="dummy-key", request_interval=2.5)
    assert downloader2.request_interval == 2.5


def test_download_tiles_部分瓦片失败时直接报错(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = TileDownloader(provider="google_satellite", request_interval=0.0)

    call_count = {"value": 0}

    def fake_download_single_tile(z: int, x: int, y: int, max_retries: int = 5):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return np.full((256, 256, 3), 255, dtype=np.uint8)
        return None

    monkeypatch.setattr(downloader, "_download_single_tile", fake_download_single_tile)

    with pytest.raises(RuntimeError, match="部分失败"):
        downloader.download_tiles(
            bounds=(114.35, 38.20, 114.36, 38.21),
            zoom=18,
            output_dir=tmp_path,
        )

    assert not any(tmp_path.glob("*.tif"))


def test_download_tiles_全部成功时正常输出(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = TileDownloader(provider="google_satellite", request_interval=0.0)

    def fake_download_single_tile(z: int, x: int, y: int, max_retries: int = 5):
        return np.full((256, 256, 3), 127, dtype=np.uint8)

    monkeypatch.setattr(downloader, "_download_single_tile", fake_download_single_tile)

    output = downloader.download_tiles(
        bounds=(114.35, 38.20, 114.3505, 38.2005),
        zoom=18,
        output_dir=tmp_path,
    )

    assert output.exists()
    with rasterio.open(output) as src:
        assert src.count == 3
        assert src.width > 0
        assert src.height > 0
