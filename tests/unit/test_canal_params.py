"""渠道参数提取单元测试."""

import numpy as np
import pytest
from rasterio.transform import from_bounds
from shapely.geometry import LineString

from seehydro.extraction.canal_params import extract_centerline, measure_width_profile
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
