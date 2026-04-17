"""渠道参数提取单元测试."""

import numpy as np
import pytest
from rasterio.transform import from_bounds
from pyproj import Transformer
from shapely.geometry import LineString

from seehydro.extraction.canal_params import (
    extract_centerline,
    extract_canal_params,
    measure_width_profile,
    vectorize_mask_gdf,
)
from seehydro.extraction.geo_measure import measure_distance_m, pixel_to_geo


def _build_horizontal_canal_case():
    """构造一条水平渠道的二值掩膜与仿射变换."""
    height, width = 120, 200
    canal_top, canal_bottom = 50, 70  # 渠道厚度 20 px

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[canal_top:canal_bottom, 10:190] = 1

    # 选择真实经纬度范围，避免投影异常
    west, south, east, north = 114.0, 34.0, 114.02, 34.012
    transform = from_bounds(west, south, east, north, width, height)

    # 估算理论宽度（米）：按同一经度下南北向距离估算
    lon_mid = (west + east) / 2
    lat0 = (south + north) / 2
    lat1 = lat0 + abs(transform.e) * (canal_bottom - canal_top)
    expected_width_m = measure_distance_m((lon_mid, lat0), (lon_mid, lat1))

    return mask, transform, expected_width_m, (canal_top, canal_bottom), (height, width)


def test_extract_centerline_水平渠道_返回非空中心线():
    mask, transform, _, _, _ = _build_horizontal_canal_case()

    centerline = extract_centerline(mask, transform)

    assert centerline is not None
    assert isinstance(centerline, LineString)
    assert len(centerline.coords) >= 2

    xs = [p[0] for p in centerline.coords]
    ys = [p[1] for p in centerline.coords]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)

    # 水平渠道中心线应"横向跨度明显大于纵向起伏"
    assert x_span > 0
    assert y_span < x_span


def test_measure_width_profile_水平渠道_宽度在合理范围内():
    mask, transform, expected_width_m, (canal_top, canal_bottom), (_, width) = _build_horizontal_canal_case()

    # 手动构造水平中心线，聚焦测试宽度量测逻辑
    row_mid = (canal_top + canal_bottom) // 2
    p_start = pixel_to_geo((10, row_mid), transform)
    p_end = pixel_to_geo((width - 10, row_mid), transform)
    centerline = LineString([p_start, p_end])

    profile = measure_width_profile(
        binary_mask=mask,
        centerline=centerline,
        transform=transform,
        interval_m=100,
        max_search_m=500,
    )

    assert len(profile) > 0
    assert "width_m" in profile.columns

    widths = profile["width_m"].to_numpy()
    assert np.all(widths > 0)

    mean_width = float(np.mean(widths))
    assert mean_width == pytest.approx(expected_width_m, rel=0.4)

    # 额外兜底：宽度应在合理量级内（米）
    assert 50 <= mean_width <= 500


def test_measure_width_profile_无有效交点_返回空表而不报错():
    mask, transform, _, _, _ = _build_horizontal_canal_case()
    zero_mask = np.zeros_like(mask)
    centerline = LineString([(114.0, 34.0), (114.01, 34.0)])

    profile = measure_width_profile(
        binary_mask=zero_mask,
        centerline=centerline,
        transform=transform,
        interval_m=100,
        max_search_m=100,
    )

    assert len(profile) == 0
    assert "width_m" in profile.columns


def test_vectorize_mask_gdf_水平渠道_返回单个面要素():
    mask, transform, _, _, _ = _build_horizontal_canal_case()

    gdf = vectorize_mask_gdf(mask, transform, class_name="water")

    assert len(gdf) == 1
    assert "class_name" in gdf.columns
    assert gdf.iloc[0]["class_name"] == "water"
    assert gdf.geometry.iloc[0].geom_type in {"Polygon", "MultiPolygon"}
    assert gdf.geometry.iloc[0].area > 0


def test_extract_centerline_弯曲渠道_不应出现明显回跳():
    height, width = 160, 220
    mask = np.zeros((height, width), dtype=np.uint8)

    for col in range(20, 200):
        row_mid = int(70 + 18 * np.sin((col - 20) / 28))
        mask[max(0, row_mid - 6):min(height, row_mid + 6), col] = 1

    transform = from_bounds(114.0, 34.0, 114.03, 34.02, width, height)
    centerline = extract_centerline(mask, transform)

    assert centerline is not None
    coords = list(centerline.coords)
    xs = np.array([pt[0] for pt in coords])

    diffs = np.diff(xs)
    non_decreasing_ratio = float(np.mean(diffs >= -1e-9))
    non_increasing_ratio = float(np.mean(diffs <= 1e-9))

    assert max(non_decreasing_ratio, non_increasing_ratio) > 0.9


def test_measure_width_profile_投影坐标系下也能得到合理宽度():
    height, width = 120, 200
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[50:70, 10:190] = 1

    lonlat_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    left, bottom = lonlat_to_3857.transform(114.0, 34.0)
    right, top = lonlat_to_3857.transform(114.02, 34.012)
    transform = from_bounds(left, bottom, right, top, width, height)

    row_mid = 60
    p_start = pixel_to_geo((10, row_mid), transform)
    p_end = pixel_to_geo((width - 10, row_mid), transform)
    centerline = LineString([p_start, p_end])

    profile = measure_width_profile(
        binary_mask=mask,
        centerline=centerline,
        transform=transform,
        crs="EPSG:3857",
        interval_m=100,
        max_search_m=1000,
    )

    assert len(profile) > 0
    assert profile["width_m"].mean() > 0


def test_extract_canal_params_中心线失败时仍保留掩膜面结果(tmp_path):
    import rasterio

    mask_path = tmp_path / "tiny_mask.tif"
    data = np.zeros((8, 8), dtype=np.uint8)
    data[3, 3] = 1

    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_bounds(114.0, 34.0, 114.001, 34.001, 8, 8),
        "nodata": 0,
    }

    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(data, 1)

    result = extract_canal_params(mask_path)

    assert result["crs"] == "EPSG:4326"
    assert "water_mask_polygon" in result
    assert len(result["water_mask_polygon"]) == 1
    assert "centerline" not in result
    assert len(result["width_profile"]) == 0
