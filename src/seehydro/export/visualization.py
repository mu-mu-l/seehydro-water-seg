"""可视化工具."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# 类别颜色映射
SEG_COLORS = {
    0: [0, 0, 0],        # 背景 - 黑色
    1: [0, 0, 255],      # 水面 - 蓝色
    2: [128, 128, 128],  # 边坡 - 灰色
    3: [255, 165, 0],    # 马道 - 橙色
    4: [255, 255, 0],    # 道路 - 黄色
}

DET_COLORS = {
    "highway_bridge": "red",
    "railway_bridge": "darkred",
    "siphon_inlet": "blue",
    "siphon_outlet": "cyan",
    "aqueduct": "green",
    "check_gate": "purple",
    "drain_gate": "magenta",
    "diversion": "orange",
}


def mask_to_rgb(mask: np.ndarray, colors: dict[int, list[int]] | None = None) -> np.ndarray:
    """将类别掩膜转为RGB图像.

    Args:
        mask: (H, W) 类别掩膜
        colors: {class_id: [R, G, B]} 颜色映射

    Returns:
        (H, W, 3) RGB图像, uint8
    """
    colors = colors or SEG_COLORS
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in colors.items():
        rgb[mask == cls_id] = color

    return rgb


def plot_width_profile(
    width_gdf: gpd.GeoDataFrame,
    output_path: str | Path | None = None,
    title: str = "估算水面宽度沿程变化",
) -> plt.Figure:
    """绘制估算水面宽度沿程变化图."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(width_gdf["distance_along_m"], width_gdf["width_m"], "b-o", markersize=3, linewidth=1)
    ax.fill_between(width_gdf["distance_along_m"], 0, width_gdf["width_m"], alpha=0.2, color="blue")

    mean_width = width_gdf["width_m"].mean()
    ax.axhline(y=mean_width, color="r", linestyle="--", label=f"平均估算宽度: {mean_width:.1f}m")

    ax.set_xlabel("沿程距离 (m)", fontsize=12)
    ax.set_ylabel("估算水面宽度 (m)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"保存图表: {output_path}")

    return fig


def create_folium_map(
    canal_gdf: gpd.GeoDataFrame | None = None,
    bridges_gdf: gpd.GeoDataFrame | None = None,
    structures_gdf: gpd.GeoDataFrame | None = None,
    center: tuple[float, float] | None = None,
    zoom_start: int = 10,
) -> "folium.Map":
    """创建folium交互地图.

    Args:
        canal_gdf: 渠道中心线/宽度数据
        bridges_gdf: 桥梁数据
        structures_gdf: 其他建筑物数据
        center: 地图中心 (lat, lon)
        zoom_start: 初始缩放级别
    """
    import folium
    from folium.plugins import MarkerCluster

    # 确定中心点
    if center is None:
        if canal_gdf is not None and len(canal_gdf) > 0:
            centroid = canal_gdf.geometry.unary_union.centroid
            center = (centroid.y, centroid.x)
        else:
            center = (36.0, 113.5)  # 南水北调中线中段

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")

    # 渠道
    if canal_gdf is not None and len(canal_gdf) > 0:
        folium.GeoJson(
            canal_gdf.__geo_interface__,
            name="渠道",
            style_function=lambda _: {"color": "blue", "weight": 3},
        ).add_to(m)

    # 桥梁
    if bridges_gdf is not None and len(bridges_gdf) > 0:
        bridge_cluster = MarkerCluster(name="桥梁").add_to(m)
        for _, row in bridges_gdf.iterrows():
            popup_text = (
                f"类型: {row.get('bridge_type_cn', row.get('bridge_type', ''))}<br>"
                f"跨度: {row.get('span_m', '-')}m<br>"
                f"置信度: {row.get('confidence', '-')}"
            )
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=popup_text,
                icon=folium.Icon(color="red", icon="road", prefix="fa"),
            ).add_to(bridge_cluster)

    # 其他建筑物
    if structures_gdf is not None and len(structures_gdf) > 0:
        struct_cluster = MarkerCluster(name="建筑物").add_to(m)
        icon_map = {
            "inverted_siphon": ("blue", "arrow-down"),
            "aqueduct": ("green", "bridge"),
            "check_gate": ("purple", "bars"),
            "drain_gate": ("orange", "tint"),
            "diversion": ("cadetblue", "code-branch"),
        }
        for _, row in structures_gdf.iterrows():
            stype = row.get("type", "")
            color, icon = icon_map.get(stype, ("gray", "info-sign"))
            popup_text = f"类型: {row.get('type_cn', stype)}<br>置信度: {row.get('confidence', '-')}"
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=popup_text,
                icon=folium.Icon(color=color, icon=icon, prefix="fa"),
            ).add_to(struct_cluster)

    folium.LayerControl().add_to(m)
    return m


def save_map(folium_map, output_path: str | Path) -> Path:
    """保存folium地图为HTML."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folium_map.save(str(output_path))
    logger.info(f"保存交互地图: {output_path}")
    return output_path
