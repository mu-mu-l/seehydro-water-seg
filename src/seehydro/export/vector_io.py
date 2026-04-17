"""矢量数据输出."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger


def _sanitize_shapefile_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """将字段名调整为 Shapefile 兼容格式.

    ESRI Shapefile 对字段名有 10 个字符限制。这里做确定性截断并处理重名，
    避免导出时由底层驱动隐式改名，导致字段不可预期。
    """
    rename_map: dict[str, str] = {}
    used: set[str] = set()

    for column in gdf.columns:
        if column == gdf.geometry.name:
            continue

        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(column))
        safe = safe[:10] or "field"
        candidate = safe
        suffix = 1
        while candidate in used:
            tail = str(suffix)
            candidate = f"{safe[:10 - len(tail)]}{tail}"
            suffix += 1
        used.add(candidate)
        rename_map[column] = candidate

    sanitized = gdf.rename(columns=rename_map).copy()

    for column in sanitized.columns:
        if column == sanitized.geometry.name:
            continue
        series = sanitized[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            sanitized[column] = series.astype(str)

    return sanitized


def save_geodataframe(
    gdf: gpd.GeoDataFrame,
    output_path: str | Path,
    driver: str | None = None,
) -> Path:
    """保存GeoDataFrame到文件.

    根据后缀名自动选择驱动:
    - .geojson -> GeoJSON
    - .shp -> ESRI Shapefile
    - .gpkg -> GPKG
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if driver is None:
        ext_map = {".geojson": "GeoJSON", ".shp": "ESRI Shapefile", ".gpkg": "GPKG"}
        driver = ext_map.get(output_path.suffix.lower(), "GeoJSON")

    output_gdf = _sanitize_shapefile_columns(gdf) if driver == "ESRI Shapefile" else gdf
    output_gdf.to_file(output_path, driver=driver, index=False)
    logger.info(f"保存矢量数据: {output_path} ({len(gdf)} 条记录)")
    return output_path


def export_all_results(
    results: dict[str, gpd.GeoDataFrame],
    output_dir: str | Path,
    formats: list[str] | None = None,
) -> list[Path]:
    """批量导出所有结果.

    Args:
        results: {name: GeoDataFrame} 字典
        output_dir: 输出目录
        formats: 输出格式列表，如 ["geojson", "shapefile"]

    Returns:
        生成的文件路径列表
    """
    output_dir = Path(output_dir)
    formats = formats or ["geojson"]
    saved_files = []

    format_ext = {
        "geojson": ".geojson",
        "shapefile": ".shp",
        "gpkg": ".gpkg",
    }

    for name, gdf in results.items():
        if gdf is None or len(gdf) == 0:
            logger.warning(f"跳过空结果: {name}")
            continue

        for fmt in formats:
            ext = format_ext.get(fmt, ".geojson")
            out_path = output_dir / f"{name}{ext}"
            save_geodataframe(gdf, out_path)
            saved_files.append(out_path)

    logger.info(f"导出完成: {len(saved_files)} 个文件")
    return saved_files
