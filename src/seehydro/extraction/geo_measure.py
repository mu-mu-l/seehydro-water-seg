"""GIS量测工具."""

import geopandas as gpd
import numpy as np
from pyproj import Geod, Transformer
from rasterio.transform import Affine
from shapely.geometry import LineString, Point


# WGS84 大地测量计算器
_geod = Geod(ellps="WGS84")


def pixel_to_geo(pixel_xy: tuple[int, int], transform: Affine) -> tuple[float, float]:
    """像素坐标(col, row)转地理坐标(lon, lat)."""
    col, row = pixel_xy
    lon, lat = transform * (col + 0.5, row + 0.5)  # 像素中心
    return lon, lat


def geo_to_pixel(lon: float, lat: float, transform: Affine) -> tuple[int, int]:
    """地理坐标转像素坐标."""
    inv_transform = ~transform
    col, row = inv_transform * (lon, lat)
    return int(col), int(row)


def measure_distance_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """两点间大地线距离（米），输入(lon, lat)."""
    _, _, dist = _geod.inv(p1[0], p1[1], p2[0], p2[1])
    return float(dist)


def measure_line_length_m(line: LineString, crs: str = "EPSG:4326") -> float:
    """计算LineString长度（米）.

    如果CRS是地理坐标系，逐段用大地线计算。
    否则直接用欧氏距离。
    """
    if crs == "EPSG:4326":
        coords = list(line.coords)
        total = 0.0
        for i in range(len(coords) - 1):
            total += measure_distance_m(coords[i][:2], coords[i + 1][:2])
        return total
    return line.length


def get_utm_crs(lon: float, lat: float) -> str:
    """根据经纬度获取UTM CRS."""
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def point_to_wgs84(x: float, y: float, crs: str | None) -> tuple[float, float]:
    """将任意 CRS 下的点坐标转换为 WGS84 经纬度."""
    if crs is None or crs.upper() == "EPSG:4326":
        return x, y

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lon), float(lat)


def compute_perpendicular(
    line: LineString,
    point: Point,
    length_m: float = 200,
) -> LineString:
    """在LineString上某点处计算垂线.

    Args:
        line: 线要素（投影坐标系下）
        point: 线上的点
        length_m: 垂线半长

    Returns:
        垂线LineString
    """
    # 在线上找到最近点的参数位置
    d = line.project(point)

    # 取前后小段计算方向
    epsilon = min(1.0, line.length * 0.001)
    p1 = line.interpolate(max(0, d - epsilon))
    p2 = line.interpolate(min(line.length, d + epsilon))

    # 切线方向
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-10:
        return LineString([(point.x, point.y), (point.x, point.y)])

    # 法线方向（旋转90度）
    nx = -dy / norm * length_m
    ny = dx / norm * length_m

    return LineString([
        (point.x + nx, point.y + ny),
        (point.x - nx, point.y - ny),
    ])
