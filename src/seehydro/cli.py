# -*- coding: utf-8 -*-
"""命令行入口。"""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from seehydro import __version__
from seehydro.utils.config import load_config

# ---------------------------------------------------------------------------
# 应用与子命令组
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="SeeHydro - 南水北调中线工程参数遥感提取系统",
    add_completion=False,
)

download_app = typer.Typer(help="数据下载")
preprocess_app = typer.Typer(help="数据预处理")
train_app = typer.Typer(help="模型训练")

app.add_typer(download_app, name="download")
app.add_typer(preprocess_app, name="preprocess")
app.add_typer(train_app, name="train")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"SeeHydro {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="显示版本号",
        ),
    ] = None,
) -> None:
    """SeeHydro - 南水北调中线工程参数遥感提取系统."""


@app.command()
def info() -> None:
    """显示项目信息与默认配置文件内容。"""
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "default.yaml"

    typer.echo("=== SeeHydro 信息 ===")
    typer.echo(f"版本       : {__version__}")
    typer.echo(f"项目根目录 : {project_root}")
    typer.echo(f"默认配置   : {config_path}")
    typer.echo("")

    if not config_path.exists():
        typer.echo("未找到配置文件")
        return

    typer.echo("=== configs/default.yaml ===")
    typer.echo(config_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# download 子命令
# ---------------------------------------------------------------------------


@download_app.command("route")
def download_route(
    source: Annotated[
        str,
        typer.Option("--source", help="数据源：osm 或 local"),
    ] = "osm",
    output: Annotated[
        Path,
        typer.Option("--output", help="输出路径"),
    ] = Path("data/route"),
) -> None:
    """下载南水北调中线线路数据。"""
    if source not in {"osm", "local"}:
        typer.echo(
            f"错误：--source 只能是 osm 或 local，当前值为 {source!r}",
            err=True,
        )
        raise typer.Exit(code=1)

    logger.info("开始执行线路数据下载 ... source={}, output={}", source, output)
    typer.echo(f"TODO: 实现线路数据下载（source={source}, output={output}）")
    # TODO: 调用 seehydro.acquisition 模块实现具体逻辑


@download_app.command("sentinel2")
def download_sentinel2(
    config: Annotated[
        Path,
        typer.Option("--config", help="下载配置文件路径"),
    ] = Path("configs/default.yaml"),
) -> None:
    """下载 Sentinel-2 多光谱影像（通过 Google Earth Engine）。"""
    logger.info("开始执行 Sentinel-2 数据下载 ... config={}", config)
    typer.echo(f"TODO: 实现 Sentinel-2 数据下载（config={config}）")
    # TODO: 调用 seehydro.acquisition 模块实现具体逻辑


# ---------------------------------------------------------------------------
# preprocess 子命令
# ---------------------------------------------------------------------------


@preprocess_app.command("clip")
def preprocess_clip(
    input: Annotated[
        Path,
        typer.Option("--input", help="输入影像目录"),
    ],
    route: Annotated[
        Path,
        typer.Option("--route", help="线路矢量文件路径（GeoJSON / Shapefile）"),
    ],
    buffer: Annotated[
        float,
        typer.Option("--buffer", help="沿线路的缓冲区宽度（米）"),
    ] = 2000.0,
    output: Annotated[
        Path,
        typer.Option("--output", help="裁剪结果输出目录"),
    ] = Path("data/clipped"),
) -> None:
    """按线路缓冲区裁剪影像。"""
    logger.info(
        "开始执行影像裁剪 ... input={}, route={}, buffer={}m, output={}",
        input, route, buffer, output,
    )
    typer.echo("TODO: 实现影像裁剪预处理")
    # TODO: 调用 seehydro.preprocessing 模块实现具体逻辑


@preprocess_app.command("tile")
def preprocess_tile(
    input: Annotated[
        Path,
        typer.Option("--input", help="输入影像目录"),
    ],
    size: Annotated[
        int,
        typer.Option("--size", help="切片尺寸（像素，正方形）"),
    ] = 512,
    overlap: Annotated[
        float,
        typer.Option("--overlap", help="相邻切片的重叠比例，取値范围 [0, 1)"),
    ] = 0.25,
    output: Annotated[
        Path,
        typer.Option("--output", help="切片输出目录"),
    ] = Path("data/tiles"),
) -> None:
    """滑窗切片，将大幅影像拆分为固定尺寸的小块。"""
    from seehydro.preprocessing.tiling import TileGenerator

    if not (0.0 <= overlap < 1.0):
        typer.echo(
            f"错误：--overlap 需在 [0, 1) 范围内，当前值为 {overlap}",
            err=True,
        )
        raise typer.Exit(code=1)

    logger.info(
        "开始执行影像切片 ... input={}, size={}, overlap={}, output={}",
        input, size, overlap, output,
    )
    input_files = []
    if input.is_dir():
        input_files = sorted(p for p in input.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    elif input.is_file():
        input_files = [input]
    else:
        typer.echo(f"错误：输入路径不存在 {input}", err=True)
        raise typer.Exit(code=1)

    if not input_files:
        typer.echo(f"错误：未在 {input} 找到可切片的 tif/tiff 影像", err=True)
        raise typer.Exit(code=1)

    generator = TileGenerator(tile_size=size, overlap=overlap)
    all_tile_infos = []
    for image_path in input_files:
        tile_infos = generator.generate_tiles(image_path=image_path, output_dir=output, prefix=image_path.stem)
        all_tile_infos.extend(tile_infos)

    index_path = output / "tile_index.csv"
    generator.save_tile_index(all_tile_infos, index_path)
    typer.echo(f"切片完成，共 {len(all_tile_infos)} 个切片，索引保存在 {index_path}")


# ---------------------------------------------------------------------------
# train 子命令
# ---------------------------------------------------------------------------


@train_app.command("segmentation")
def train_segmentation(
    config: Annotated[
        Path,
        typer.Option("--config", help="训练配置文件路径（YAML）"),
    ] = Path("configs/default.yaml"),
) -> None:
    """训练语义分割模型（水面、边坡、马道等）。"""
    from seehydro.training.train_seg import train_segmentation as run_segmentation_training

    logger.info("开始执行分割模型训练 ... config={}", config)
    cfg = load_config(config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    output_cfg = cfg.get("output", {})

    image_dir = Path(data_cfg.get("image_dir", "data/tiles/highres/images"))
    mask_dir = Path(data_cfg.get("mask_dir", "data/tiles/highres/masks"))
    output_dir = Path(output_cfg.get("checkpoint_dir", "models/trained"))

    if not image_dir.exists() or not mask_dir.exists():
        typer.echo(f"错误：训练数据目录不存在 image_dir={image_dir}, mask_dir={mask_dir}", err=True)
        raise typer.Exit(code=1)

    merged_cfg = {
        **model_cfg,
        **train_cfg,
    }
    best_path = run_segmentation_training(
        image_dir=image_dir,
        mask_dir=mask_dir,
        config=merged_cfg,
        output_dir=output_dir,
    )
    typer.echo(f"分割训练完成，最佳模型保存在 {best_path}")


@train_app.command("detection")
def train_detection(
    config: Annotated[
        Path,
        typer.Option("--config", help="训练配置文件路径（YAML）"),
    ] = Path("configs/default.yaml"),
) -> None:
    """训练目标检测模型（桥梁、倒虹吸、渡槽等建筑物）。"""
    from seehydro.training.train_det import train_detection as run_detection_training

    logger.info("开始执行检测模型训练 ... config={}", config)
    cfg = load_config(config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    output_cfg = cfg.get("output", {})

    data_yaml = Path(data_cfg.get("data_yaml", "datasets/detection/data.yaml"))
    output_dir = Path(output_cfg.get("project_dir", "models/trained/detection"))

    if not data_yaml.exists():
        typer.echo(f"错误：检测数据配置不存在 {data_yaml}", err=True)
        raise typer.Exit(code=1)

    merged_cfg = {
        "model_name": model_cfg.get("model_name", "yolov8m"),
        "input_size": model_cfg.get("input_size", 1024),
        "batch_size": train_cfg.get("batch_size", 4),
        "epochs": train_cfg.get("epochs", 200),
        "lr": train_cfg.get("lr", 1e-3),
        "experiment_name": train_cfg.get("experiment_name", "snbd_det"),
    }
    best_path = run_detection_training(
        data_yaml=data_yaml,
        config=merged_cfg,
        output_dir=output_dir,
    )
    typer.echo(f"检测训练完成，最佳模型保存在 {best_path}")


# ---------------------------------------------------------------------------
# infer（直接挂在 app 上）
# ---------------------------------------------------------------------------


@app.command("infer")
def run_infer(
    input: Annotated[
        Path,
        typer.Option("--input", help="待推理的影像切片目录"),
    ],
    model_seg: Annotated[
        Path,
        typer.Option("--model-seg", help="分割模型权重文件路径（.pth）"),
    ],
    model_det: Annotated[
        Path,
        typer.Option("--model-det", help="检测模型权重文件路径（.pt）"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="推理结果输出目录"),
    ] = Path("outputs/infer"),
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="推理批次大小"),
    ] = 4,
    device: Annotated[
        str,
        typer.Option("--device", help="推理设备：cpu / cuda / cuda:0 / cuda:1 ..."),
    ] = "cpu",
) -> None:
    """对切片影像执行分割与检测推理，生成掩膜和检测框。"""
    logger.info(
        "开始执行推理预测 ... input={}, model_seg={}, model_det={},"
        " output={}, batch_size={}, device={}",
        input, model_seg, model_det, output, batch_size, device,
    )
    typer.echo("TODO: 实现推理预测")
    # TODO: 调用 seehydro.models 推理管线实现具体逻辑


# ---------------------------------------------------------------------------
# extract（直接挂在 app 上）
# ---------------------------------------------------------------------------


@app.command("extract")
def run_extract(
    input: Annotated[
        Path,
        typer.Option("--input", help="推理结果目录（掩膜 + 检测框）"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="参数提取结果输出目录"),
    ] = Path("outputs/extraction"),
    sample_interval: Annotated[
        float,
        typer.Option("--sample-interval", help="横断面采样间距（米）"),
    ] = 50.0,
) -> None:
    """从推理结果中提取渠道宽度、水面面积等水文参数。"""
    logger.info(
        "开始执行参数提取 ... input={}, output={}, sample_interval={}m",
        input, output, sample_interval,
    )
    typer.echo("TODO: 实现参数提取")
    # TODO: 调用 seehydro.extraction 模块实现具体逻辑


# ---------------------------------------------------------------------------
# export（直接挂在 app 上）
# ---------------------------------------------------------------------------

_VALID_FORMATS = {"geojson", "shapefile"}


@app.command("export")
def run_export(
    input: Annotated[
        Path,
        typer.Option("--input", help="参数提取结果目录"),
    ],
    fmt: Annotated[
        str,
        typer.Option("--format", help="矢量输出格式：geojson 或 shapefile"),
    ] = "geojson",
    report: Annotated[
        Path | None,
        typer.Option("--report", help="报告输出目录（可选，生成 HTML / Excel 报告）"),
    ] = None,
) -> None:
    """将提取参数导出为矢量文件，并可选生成分析报告。"""
    if fmt not in _VALID_FORMATS:
        typer.echo(
            f"错误：--format 只能是 geojson 或 shapefile，当前值为 {fmt!r}",
            err=True,
        )
        raise typer.Exit(code=1)

    logger.info("开始执行结果导出 ... input={}, format={}, report={}", input, fmt, report)
    typer.echo("TODO: 实现结果导出")
    # TODO: 调用 seehydro.export 模块实现具体逻辑


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
