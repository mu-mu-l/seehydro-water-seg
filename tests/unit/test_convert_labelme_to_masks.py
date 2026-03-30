"""convert_labelme_to_masks 单元测试."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.convert_labelme_to_masks import _load_labelme_image, _write_mask_tif, _write_rgb_tif


def test_load_labelme_image_读取_geotiff_时保留空间参考(tmp_path: Path) -> None:
    """当 Labelme 指向 GeoTIFF 原图时，应读取到空间参考信息."""
    image_path = tmp_path / "scene.tif"
    data = np.zeros((3, 32, 32), dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "height": 32,
        "width": 32,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(100.0, 30.0, 0.0001, 0.0001),
        "nodata": 0,
    }
    with rasterio.open(image_path, "w", **profile) as dst:
        dst.write(data)

    json_path = tmp_path / "scene.json"
    payload = {"imagePath": "scene.tif", "shapes": []}
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    image, geo_profile = _load_labelme_image(json_path, payload)

    assert image.size == (32, 32)
    assert geo_profile is not None
    assert str(geo_profile["crs"]) == "EPSG:4326"
    assert geo_profile["transform"] == profile["transform"]


def test_write_tif_继承_geotiff_空间参考(tmp_path: Path) -> None:
    """输出 image/mask tif 时应继承 GeoTIFF 的空间参考信息."""
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")
    mask = np.zeros((32, 32), dtype=np.uint8)
    geo_profile = {
        "crs": "EPSG:4326",
        "transform": from_origin(100.0, 30.0, 0.0001, 0.0001),
        "nodata": 0,
        "width": 32,
        "height": 32,
    }

    image_out = tmp_path / "image.tif"
    mask_out = tmp_path / "mask.tif"
    _write_rgb_tif(image, image_out, geo_profile=geo_profile)
    _write_mask_tif(mask, mask_out, geo_profile=geo_profile)

    with rasterio.open(image_out) as src:
        assert str(src.crs) == "EPSG:4326"
        assert src.transform == geo_profile["transform"]
        assert src.count == 3

    with rasterio.open(mask_out) as src:
        assert str(src.crs) == "EPSG:4326"
        assert src.transform == geo_profile["transform"]
        assert src.count == 1
        assert src.nodata == 255


def test_write_tif_普通图片也能正常输出(tmp_path: Path) -> None:
    """没有 GeoTIFF 空间参考时，仍应能写出普通训练 tif."""
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB")
    mask = np.zeros((16, 16), dtype=np.uint8)

    image_out = tmp_path / "plain_image.tif"
    mask_out = tmp_path / "plain_mask.tif"
    _write_rgb_tif(image, image_out, geo_profile=None)
    _write_mask_tif(mask, mask_out, geo_profile=None)

    with rasterio.open(image_out) as src:
        assert src.width == 16
        assert src.height == 16
        assert src.crs is None

    with rasterio.open(mask_out) as src:
        assert src.width == 16
        assert src.height == 16
        assert src.crs is None
