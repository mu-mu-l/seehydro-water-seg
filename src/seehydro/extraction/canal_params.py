"""渠道水面辅助结果提取（中心线、估算水面宽度、估算马道宽度）."""

from pathlib import Path
from collections import deque

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
    point_to_wgs84,
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

    pixel_path = _extract_skeleton_path(skeleton)
    if len(pixel_path) < 3:
        return None

    # 转地理坐标
    ordered = [pixel_to_geo((int(c), int(r)), transform) for r, c in pixel_path]

    if len(ordered) < 3:
        return None

    # B-spline平滑
    smoothed = _smooth_line(ordered)

    return smoothed


def _extract_skeleton_path(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """从骨架中提取一条有序主路径.

    基于 8 邻域连通图，优先取最长简单路径近似，避免最近邻串点导致的跨掩膜乱连。
    """
    rows, cols = np.where(skeleton)
    pixels = {(int(r), int(c)) for r, c in zip(rows, cols)}
    if len(pixels) < 3:
        return []

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for pixel in pixels:
        r, c = pixel
        neighbors = []
        for dr, dc in offsets:
            candidate = (r + dr, c + dc)
            if candidate in pixels:
                neighbors.append(candidate)
        adjacency[pixel] = neighbors

    endpoints = [node for node, neighbors in adjacency.items() if len(neighbors) <= 1]
    start = min(endpoints, key=lambda p: (p[1], p[0])) if endpoints else min(pixels, key=lambda p: (p[1], p[0]))

    first = _farthest_node(start, adjacency)
    second, parents = _farthest_node(first, adjacency, return_parents=True)

    path = [second]
    while path[-1] != first:
        parent = parents.get(path[-1])
        if parent is None:
            break
        path.append(parent)
    path.reverse()
    return path


def _farthest_node(
    start: tuple[int, int],
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
    return_parents: bool = False,
):
    """在骨架图上做 BFS，返回最远节点."""
    visited = {start}
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    distance = {start: 0}
    queue = deque([start])
    farthest = start

    while queue:
        node = queue.popleft()
        farthest = max(
            farthest,
            node,
            key=lambda p: (distance[p], -len(adjacency[p]), p[1], p[0]),
        )
        for neighbor in adjacency[node]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parents[neighbor] = node
            distance[neighbor] = distance[node] + 1
            queue.append(neighbor)

    if return_parents:
        return farthest, parents
    return farthest


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
    centroid_lon, centroid_lat = point_to_wgs84(centroid.x, centroid.y, crs)
    utm_crs = get_utm_crs(centroid_lon, centroid_lat)

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


def vectorize_mask_gdf(
    binary_mask: np.ndarray,
    transform,
    crs: str = "EPSG:4326",
    class_name: str = "water",
) -> gpd.GeoDataFrame:
    """将二值掩膜矢量化为 GeoDataFrame."""
    geometry = _vectorize_mask(binary_mask, transform, crs)
    if geometry is None or geometry.is_empty:
        return gpd.GeoDataFrame(columns=["class_name", "pixel_area"], geometry=[], crs=crs)

    records = [{"class_name": class_name, "pixel_area": int(binary_mask.sum())}]
    return gpd.GeoDataFrame(records, geometry=[geometry], crs=crs)


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

    result["crs"] = crs
    result["water_mask_polygon"] = vectorize_mask_gdf(
        water_mask,
        transform,
        crs=crs,
        class_name="water",
    )
    result["water_pixel_area"] = int(water_mask.sum())

    centerline = extract_centerline(water_mask, transform, crs=crs)
    if centerline is None:
        logger.warning("未能提取渠道中心线，将仅保留掩膜面结果")
        result["width_profile"] = gpd.GeoDataFrame(columns=["geometry", "width_m", "distance_along_m"], crs=crs)
        result["mean_estimated_water_surface_width_m"] = 0
        return result

    result["centerline"] = centerline
    result["width_profile"] = measure_width_profile(water_mask, centerline, transform, crs=crs, interval_m=interval_m)
    result["mean_estimated_water_surface_width_m"] = (
        result["width_profile"]["width_m"].mean() if len(result["width_profile"]) > 0 else 0
    )

    # 马道
    berm_mask, _ = extract_mask_from_raster(mask_path, berm_class_id)
    if berm_mask.sum() > 0:
        result["berm_mask_polygon"] = vectorize_mask_gdf(
            berm_mask,
            transform,
            crs=crs,
            class_name="berm",
        )
        result["berm_pixel_area"] = int(berm_mask.sum())
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
