#!/usr/bin/env python3
"""将 Labelme JSON 批量转换为 SeeHydro 可用的 image/mask 数据集."""

from __future__ import annotations

import base64
import io
import json
import shutil
from pathlib import Path
from typing import Annotated

import numpy as np
import rasterio
import typer
from PIL import Image, ImageDraw

app = typer.Typer(help="把 Labelme 标注批量转换成 SeeHydro 训练数据目录")


def _load_labelme_image(json_path: Path, payload: dict) -> Image.Image:
    """优先从 imageData 读取图像，否则从 imagePath 读取原图."""
    image_data = payload.get("imageData")
    if image_data:
        raw = base64.b64decode(image_data)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    image_path = payload.get("imagePath")
    if not image_path:
        raise ValueError(f"{json_path} 缺少 imageData 和 imagePath，无法恢复原图")

    source_image = (json_path.parent / image_path).resolve()
    if not source_image.exists():
        raise FileNotFoundError(f"{json_path} 指向的原图不存在: {source_image}")
    return Image.open(source_image).convert("RGB")


def _draw_mask(payload: dict, size: tuple[int, int], label_map: dict[str, int]) -> np.ndarray:
    """将 Labelme shapes 渲染为类别掩膜."""
    width, height = size
    mask = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask)

    for shape in payload.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if label not in label_map:
            continue

        points = shape.get("points", [])
        if len(points) < 2:
            continue

        polygon = [(float(x), float(y)) for x, y in points]
        class_id = int(label_map[label])
        shape_type = str(shape.get("shape_type", "polygon")).strip()

        if shape_type == "rectangle" and len(polygon) >= 2:
            draw.rectangle([polygon[0], polygon[1]], fill=class_id)
        else:
            draw.polygon(polygon, fill=class_id)

    return np.array(mask, dtype=np.uint8)


def _write_rgb_tif(image: Image.Image, output_path: Path) -> None:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"原图必须是 RGB 三通道，得到 shape={arr.shape}")

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 3,
        "dtype": "uint8",
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr.transpose(2, 0, 1))


def _write_mask_tif(mask: np.ndarray, output_path: Path) -> None:
    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)


@app.command()
def main(
    input_dir: Annotated[Path, typer.Option("--input-dir", help="Labelme JSON 所在目录")] = Path("labelme_work"),
    output_root: Annotated[Path, typer.Option("--output-root", help="输出数据集根目录")] = Path("data/seg_water"),
    water_label: Annotated[str, typer.Option("--water-label", help="Labelme 中代表水面的标签名")] = "water",
    image_suffix: Annotated[str, typer.Option("--image-suffix", help="输出影像后缀")] = ".tif",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="仅检查，不写文件")] = False,
) -> None:
    """把 Labelme JSON 转为 images/*.tif + masks/*.tif."""
    if not input_dir.exists():
        typer.echo(f"错误：输入目录不存在 {input_dir}", err=True)
        raise typer.Exit(code=1)

    if image_suffix.lower() != ".tif":
        typer.echo("错误：当前脚本仅支持输出 .tif", err=True)
        raise typer.Exit(code=1)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        typer.echo(f"错误：未在 {input_dir} 找到 json 标注文件", err=True)
        raise typer.Exit(code=1)

    label_map = {water_label: 1}
    image_dir = output_root / "images"
    mask_dir = output_root / "masks"

    if not dry_run:
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for json_path in json_files:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        image = _load_labelme_image(json_path, payload)
        mask = _draw_mask(payload, image.size, label_map)

        stem = json_path.stem
        image_out = image_dir / f"{stem}.tif"
        mask_out = mask_dir / f"{stem}.tif"

        if not dry_run:
            _write_rgb_tif(image, image_out)
            _write_mask_tif(mask, mask_out)

        converted += 1

    typer.echo(f"转换完成: {converted} 个样本 -> {output_root}")
    typer.echo(f"影像目录: {image_dir}")
    typer.echo(f"掩膜目录: {mask_dir}")


if __name__ == "__main__":
    app()
