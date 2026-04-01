"""TileGenerator 单元测试."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from seehydro.preprocessing.tiling import TileGenerator


@pytest.fixture
def test_geotiff_path(tmp_path: Path) -> Path:
    """创建临时测试 GeoTIFF（3 通道，256x256，随机数据）."""
    raster_path = tmp_path / "test_input.tif"

    height, width, channels = 256, 256, 3
    rng = np.random.default_rng(42)
    data = rng.integers(0, 255, size=(channels, height, width), dtype=np.uint8)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": channels,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(100.0, 30.0, 0.0001, 0.0001),
        "nodata": 0,
    }

    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(data)

    return raster_path


def test_generate_tiles_随机影像_返回有效切片(test_geotiff_path: Path, tmp_path: Path) -> None:
    """测试 generate_tiles: 随机影像可成功切片，数量和文件结果合理."""
    output_dir = tmp_path / "tiles"
    generator = TileGenerator(tile_size=128, overlap=0.0)

    tile_infos = generator.generate_tiles(
        image_path=test_geotiff_path,
        output_dir=output_dir,
        prefix="tile",
        min_valid_ratio=0.5,
    )

    # 256x256 在 tile_size=128, overlap=0 下应为 2x2 共 4 个切片
    assert len(tile_infos) == 4

    for info in tile_infos:
        assert info.tile_path.exists(), f"切片文件不存在: {info.tile_path}"
        assert info.width == 128
        assert info.height == 128
        assert len(info.bounds) == 4

        with rasterio.open(info.tile_path) as src:
            assert src.width == 128
            assert src.height == 128
            assert src.count == 3


def test_保存加载切片索引_有效切片列表_读回一致(test_geotiff_path: Path, tmp_path: Path) -> None:
    """测试 save_tile_index/load_tile_index: CSV 索引可保存并正确读取."""
    output_dir = tmp_path / "tiles"
    csv_path = tmp_path / "tile_index.csv"
    generator = TileGenerator(tile_size=128, overlap=0.0)

    tile_infos = generator.generate_tiles(
        image_path=test_geotiff_path,
        output_dir=output_dir,
        prefix="tile",
        min_valid_ratio=0.5,
    )

    saved_csv = generator.save_tile_index(tile_infos, csv_path)
    assert saved_csv.exists()

    loaded_infos = generator.load_tile_index(saved_csv)

    assert len(loaded_infos) == len(tile_infos)

    original_keys = {(t.row, t.col) for t in tile_infos}
    loaded_keys = {(t.row, t.col) for t in loaded_infos}
    assert loaded_keys == original_keys

    original_paths = {str(t.tile_path) for t in tile_infos}
    loaded_paths = {str(t.tile_path) for t in loaded_infos}
    assert loaded_paths == original_paths


def test_reassemble_预测切片_可正确回拼(test_geotiff_path: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "tiles"
    merged_path = tmp_path / "merged_mask.tif"
    generator = TileGenerator(tile_size=128, overlap=0.0)

    tile_infos = generator.generate_tiles(
        image_path=test_geotiff_path,
        output_dir=output_dir,
        prefix="tile",
        min_valid_ratio=0.5,
    )

    tiles = {}
    for info in tile_infos:
        mask = np.full((info.height, info.width), fill_value=1, dtype=np.uint8)
        mask_path = output_dir / f"{info.tile_path.stem}_mask.tif"
        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            height=info.height,
            width=info.width,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(100.0, 30.0, 0.0001, 0.0001),
        ) as dst:
            dst.write(mask, 1)
        tiles[info.tile_path.stem] = mask_path

    with rasterio.open(test_geotiff_path) as src:
        profile = src.profile.copy()

    merged = generator.reassemble(
        tiles=tiles,
        tile_infos=tile_infos,
        output_path=merged_path,
        original_profile=profile,
    )

    with rasterio.open(merged) as src:
        data = src.read(1)
        assert src.width == 256
        assert src.height == 256
        assert np.all(data == 1)
