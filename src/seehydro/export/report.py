"""统计报表生成."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger


REPORT_COLUMNS = ["类别", "子类", "数量", "指标项", "指标值", "单位", "备注"]


def generate_summary_report(
    canal_params: dict | None = None,
    bridges: gpd.GeoDataFrame | None = None,
    siphons: gpd.GeoDataFrame | None = None,
    aqueducts: gpd.GeoDataFrame | None = None,
    gates: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """生成工程参数汇总报表.

    Returns:
        汇总DataFrame
    """
    rows = []

    # 渠道参数
    if canal_params:
        rows.append({
            "类别": "渠道",
            "子类": "水面",
            "数量": 1,
            "指标项": "平均估算宽度",
            "指标值": canal_params.get("mean_estimated_water_surface_width_m", "-"),
            "单位": "m",
            "备注": f"采样点数: {len(canal_params.get('width_profile', []))}",
        })
        if "mean_estimated_berm_width_m" in canal_params:
            rows.append({
                "类别": "渠道",
                "子类": "马道",
                "数量": 1,
                "指标项": "平均估算宽度",
                "指标值": canal_params["mean_estimated_berm_width_m"],
                "单位": "m",
                "备注": "",
            })

    # 桥梁
    if bridges is not None and len(bridges) > 0:
        for btype in bridges["bridge_type"].unique():
            subset = bridges[bridges["bridge_type"] == btype]
            rows.append({
                "类别": "桥梁",
                "子类": subset["bridge_type_cn"].iloc[0] if "bridge_type_cn" in subset.columns else btype,
                "数量": len(subset),
                "指标项": "平均跨度",
                "指标值": round(subset["span_m"].mean(), 1),
                "单位": "m",
                "备注": f"跨度范围: {subset['span_m'].min():.1f}-{subset['span_m'].max():.1f}m",
            })

    # 倒虹吸
    if siphons is not None and len(siphons) > 0:
        matched = siphons[siphons["type"] == "inverted_siphon"]
        if len(matched) > 0:
            rows.append({
                "类别": "倒虹吸",
                "子类": "已匹配",
                "数量": len(matched),
                "指标项": "平均长度",
                "指标值": round(matched["length_m"].mean(), 1),
                "单位": "m",
                "备注": "",
            })

    # 渡槽
    if aqueducts is not None and len(aqueducts) > 0:
        rows.append({
            "类别": "渡槽",
            "子类": "-",
            "数量": len(aqueducts),
            "指标项": "平均长度",
            "指标值": round(aqueducts["length_m"].mean(), 1),
            "单位": "m",
            "备注": "",
        })

    # 闸门
    if gates is not None and len(gates) > 0:
        for gtype in gates["type"].unique():
            subset = gates[gates["type"] == gtype]
            rows.append({
                "类别": "闸门",
                "子类": subset["type_cn"].iloc[0] if "type_cn" in subset.columns else gtype,
                "数量": len(subset),
                "指标项": "-",
                "指标值": "-",
                "单位": "-",
                "备注": "",
            })

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def save_report(
    summary_df: pd.DataFrame,
    output_dir: str | Path,
    name: str = "工程参数汇总",
) -> list[Path]:
    """保存报表为CSV和Excel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # CSV
    csv_path = output_dir / f"{name}.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    saved.append(csv_path)

    # Excel
    xlsx_path = output_dir / f"{name}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="汇总", index=False)
    saved.append(xlsx_path)

    logger.info(f"报表已保存: {[str(p) for p in saved]}")
    return saved
