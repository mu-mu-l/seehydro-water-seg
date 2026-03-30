#!/usr/bin/env python3
"""分割数据集自检脚本."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import numpy as np
import rasterio
import typer

app = typer.Typer(help="检查分割训练数据目录是否满足 SeeHydro 要求")


def _collect_tifs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.tif") if p.is_file())


@app.command()
def main(
    image_dir: Annotated[Path, typer.Option("--image-dir", help="影像目录")] = Path("data/seg_water/images"),
    mask_dir: Annotated[Path, typer.Option("--mask-dir", help="掩膜目录")] = Path("data/seg_water/masks"),
    num_classes: Annotated[int, typer.Option("--num-classes", help="类别总数")] = 2,
) -> None:
    """检查影像/掩膜配对、通道数、掩膜类别值和尺寸一致性。"""
    if num_classes <= 0:
        typer.echo(f"错误：num_classes 必须大于 0，当前为 {num_classes}", err=True)
        raise typer.Exit(code=1)

    if not image_dir.exists() or not mask_dir.exists():
        typer.echo(f"错误：目录不存在 image_dir={image_dir}, mask_dir={mask_dir}", err=True)
        raise typer.Exit(code=1)

    image_files = _collect_tifs(image_dir)
    mask_files = _collect_tifs(mask_dir)
    image_map = {p.stem: p for p in image_files}
    mask_map = {p.stem: p for p in mask_files}

    image_stems = set(image_map)
    mask_stems = set(mask_map)
    common_stems = sorted(image_stems & mask_stems)
    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)

    typer.echo(f"影像数量: {len(image_files)}")
    typer.echo(f"掩膜数量: {len(mask_files)}")
    typer.echo(f"有效配对: {len(common_stems)}")

    if missing_masks:
        typer.echo(f"缺少掩膜的影像: {missing_masks[:10]}", err=True)
    if missing_images:
        typer.echo(f"缺少影像的掩膜: {missing_images[:10]}", err=True)
    if not common_stems:
        raise typer.Exit(code=1)

    class_counter: Counter[int] = Counter()
    invalid_class_files: list[str] = []
    shape_mismatches: list[str] = []
    channel_mismatches: list[str] = []

    for stem in common_stems:
        image_path = image_map[stem]
        mask_path = mask_map[stem]

        with rasterio.open(image_path) as img_src:
            image_shape = (img_src.height, img_src.width)
            image_count = img_src.count

        with rasterio.open(mask_path) as mask_src:
            mask_shape = (mask_src.height, mask_src.width)
            mask_count = mask_src.count
            mask = mask_src.read(1)

        if image_shape != mask_shape:
            shape_mismatches.append(stem)
        if image_count < 1 or mask_count != 1:
            channel_mismatches.append(stem)

        unique_values = np.unique(mask)
        class_counter.update(int(v) for v in unique_values.tolist())
        if unique_values.min() < 0 or unique_values.max() >= num_classes:
            invalid_class_files.append(stem)

    typer.echo(f"掩膜类别值统计: {dict(sorted(class_counter.items()))}")

    if shape_mismatches:
        typer.echo(f"尺寸不一致样本: {shape_mismatches[:10]}", err=True)
    if channel_mismatches:
        typer.echo(f"通道不合法样本: {channel_mismatches[:10]}", err=True)
    if invalid_class_files:
        typer.echo(f"类别值越界样本: {invalid_class_files[:10]}", err=True)

    has_error = bool(missing_masks or missing_images or shape_mismatches or channel_mismatches or invalid_class_files)
    if has_error:
        raise typer.Exit(code=1)

    typer.echo("数据集检查通过")


if __name__ == "__main__":
    app()
