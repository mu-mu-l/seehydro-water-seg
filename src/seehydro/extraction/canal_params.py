"""渠道水面辅助结果提取（中心线、估算水面宽度、估算马道宽度）."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from loguru import logger
from rasterio.features import shapes
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, shape
from shapely.ops import unary_union
from skimage.morphology import skeletonize

from seehydro.extraction.geo_measure import (
    compute_perpendicular,
    get_utm_crs,
    pixel_to_geo,
)


def extract_mask_from_raster(mask_path: str | Path, class_id: int) -> tuple[np.ndarray, dict]:
    """从分割掩膜中提取指定类别的二值掩膜，并返回空间参考信息."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = None if src.crs is None else str(src.crs)
    binary = (mask == class_id).astype(np.uint8)
    return binary, {"profile": profile, "transform": transform, "crs": crs}


def extract_centerline(binary_mask: np.ndarray, transform, crs: str = "EPSG:4326") -> LineString | None:
    """从二值掩膜提取中心线.

    1. 骨架化
    2. 矢量化骨架像素为点
    3. 连接为LineString
    4. B-spline平滑
    """
    if binary_mask.sum() == 0:
        return None

    # 骨架化
    skeleton = skeletonize(binary_mask > 0)

    # 骨架像素坐标
    rows, cols = np.where(skeleton)
    if len(rows) < 3:
        return None

    # 转地理坐标
    points = [pixel_to_geo((int(c), int(r)), transform) for r, c in zip(rows, cols)]

    # 按距离排序（简单贪心最近邻）
    ordered = _order_points_greedy(points)

    if len(ordered) < 3:
        return None

    # B-spline平滑
    smoothed = _smooth_line(ordered)

    return smoothed


def _order_points_greedy(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """贪心最近邻排序点集."""
    remaining = list(points)
    ordered = [remaining.pop(0)]

    while remaining:
        last = ordered[-1]
        dists = [(i, (p[0] - last[0])**2 + (p[1] - last[1])**2) for i, p in enumerate(remaining)]
        nearest_idx = min(dists, key=lambda x: x[1])[0]
        ordered.append(remaining.pop(nearest_idx))

    return ordered


def _smooth_line(points: list[tuple[float, float]], smoothing: float = 0.001) -> LineString:
    """B-spline平滑."""
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    try:
        tck, _ = splprep([x, y], s=smoothing, k=min(3, len(points) - 1))
        t_new = np.linspace(0, 1, max(len(points) // 2, 10))
        x_smooth, y_smooth = splev(t_new, tck)
        return LineString(zip(x_smooth, y_smooth))
    except (ValueError, TypeError):
        return LineString(points)


def measure_width_profile(
    binary_mask: np.ndarray,
    centerline: LineString,
    transform,
    crs: str = "EPSG:4326",
    interval_m: float = 50,
    max_search_m: float = 200,
) -> gpd.GeoDataFrame:
    """沿中心线采样估算水面宽度.

    Args:
        binary_mask: 渠道二值掩膜
        centerline: 中心线 (地理坐标)
        transform: 栅格仿射变换
        crs: 坐标系
        interval_m: 采样间距（米）
        max_search_m: 垂线最大搜索半径（米）

    Returns:
        GeoDataFrame，每行为一个采样点，含 width_m（估算水面宽度）, geometry(Point)
    """
    # 投影到UTM
    centroid = centerline.centroid
    utm_crs = get_utm_crs(centroid.x, centroid.y)

    gdf_line = gpd.GeoDataFrame(geometry=[centerline], crs=crs)
    gdf_line_utm = gdf_line.to_crs(utm_crs)
    line_utm = gdf_line_utm.geometry.iloc[0]

    # 矢量化掩膜边界
    mask_polygons = _vectorize_mask(binary_mask, transform, crs)
    if mask_polygons is None:
        return gpd.GeoDataFrame(columns=["geometry", "width_m", "distance_along_m"], crs=crs)

    mask_gdf = gpd.GeoDataFrame(geometry=[mask_polygons], crs=crs)
    mask_utm = mask_gdf.to_crs(utm_crs).geometry.iloc[0]

    # 沿中心线采样
    total_length = line_utm.length
    num_samples = max(1, int(total_length / interval_m))

    records = []
    for i in range(num_samples + 1):
        d = min(i * interval_m, total_length)
        point = line_utm.interpolate(d)

        # 计算垂线
        perp = compute_perpendicular(line_utm, point, max_search_m)

        # 垂线与掩膜多边形的交集
        intersection = perp.intersection(mask_utm)
        if intersection.is_empty:
            continue

        width = intersection.length

        # 转回地理坐标
        point_geo = gpd.GeoDataFrame(geometry=[point], crs=utm_crs).to_crs(crs).geometry.iloc[0]

        records.append({
            "geometry": point_geo,
            "width_m": round(width, 2),
            "distance_along_m": round(d, 1),
        })

    result = gpd.GeoDataFrame(records, crs=crs)
    if len(result) == 0 or "width_m" not in result.columns:
        logger.warning("未提取到有效的估算水面宽度采样点")
        return gpd.GeoDataFrame(columns=["geometry", "width_m", "distance_along_m"], crs=crs)

    logger.info(f"估算水面宽度采样: {len(result)} 个点, 平均估算宽度 {result['width_m'].mean():.1f}m")
    return result


def _vectorize_mask(
    binary_mask: np.ndarray,
    transform,
    crs: str = "EPSG:4326",
):
    """将二值掩膜矢量化为多边形."""
    mask_shapes = list(shapes(binary_mask.astype(np.int16), transform=transform))
    polygons = [shape(s) for s, v in mask_shapes if v == 1]
    if not polygons:
        return None
    return unary_union(polygons)


def extract_canal_params(
    mask_path: str | Path,
    water_class_id: int = 1,
    berm_class_id: int = 3,
    interval_m: float = 50,
) -> dict:
    """从分割掩膜提取基础渠道辅助结果.

    Returns:
        {
            "centerline": LineString,
            "width_profile": GeoDataFrame,
            "berm_width_profile": GeoDataFrame (如果有马道),
            "mean_estimated_water_surface_width_m": float,
            "mean_estimated_berm_width_m": float,
        }
    """
    result = {}

    # 渠道水面
    water_mask, meta = extract_mask_from_raster(mask_path, water_class_id)
    transform = meta["transform"]
    crs = meta.get("crs") or "EPSG:4326"

    centerline = extract_centerline(water_mask, transform, crs=crs)
    if centerline is None:
        logger.warning("未能提取渠道中心线")
        return result

    result["crs"] = crs
    result["centerline"] = centerline
    result["width_profile"] = measure_width_profile(water_mask, centerline, transform, crs=crs, interval_m=interval_m)
    result["mean_estimated_water_surface_width_m"] = (
        result["width_profile"]["width_m"].mean() if len(result["width_profile"]) > 0 else 0
    )

    # 马道
    berm_mask, _ = extract_mask_from_raster(mask_path, berm_class_id)
    if berm_mask.sum() > 0:
        result["berm_width_profile"] = measure_width_profile(
            berm_mask,
            centerline,
            transform,
            crs=crs,
            interval_m=interval_m,
        )
        result["mean_estimated_berm_width_m"] = (
            result["berm_width_profile"]["width_m"].mean() if len(result["berm_width_profile"]) > 0 else 0
        )

    return result
