# -*- coding: utf-8 -*-
"""命令行入口。"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from seehydro import __version__
from seehydro.acquisition.route import load_route
from seehydro.utils.config import load_config

# ---------------------------------------------------------------------------
# 应用与子命令组
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="SeeHydro - 南水北调中线渠道水面识别与辅助分析工具",
    add_completion=False,
)

download_app = typer.Typer(help="数据下载")
preprocess_app = typer.Typer(help="数据预处理")
train_app = typer.Typer(help="模型训练")
pipeline_app = typer.Typer(help="端到端流程")

app.add_typer(download_app, name="download")
app.add_typer(preprocess_app, name="preprocess")
app.add_typer(train_app, name="train")
app.add_typer(pipeline_app, name="pipeline")


def _run_local_script(script_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """以当前 Python 解释器运行仓库内脚本。"""
    cmd = [sys.executable, str(script_path), *args]
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    """解析 west,south,east,north 格式的 bbox 字符串."""
    try:
        west, south, east, north = [float(v.strip()) for v in bbox.split(",")]
    except ValueError as exc:
        raise typer.BadParameter("bbox 需为 west,south,east,north，例如 114.35,38.20,114.39,38.23") from exc

    if west >= east or south >= north:
        raise typer.BadParameter("bbox 范围非法，需满足 west < east 且 south < north")
    return west, south, east, north


def _load_seg_inference_config(config_path: Path | None) -> dict:
    """读取推理阶段使用的分割模型配置，兼容训练配置结构."""
    if config_path is None:
        return {}

    cfg = load_config(config_path)
    if "model" in cfg:
        model_cfg = cfg.get("model", {})
        preprocessing_cfg = cfg.get("preprocessing", {}).get("normalize", {})
    else:
        model_cfg = cfg.get("segmentation", {})
        preprocessing_cfg = cfg.get("preprocessing", {}).get("normalize", {})

    result = dict(model_cfg)
    if "method" in preprocessing_cfg:
        result["normalize_method"] = preprocessing_cfg.get("method")
    return result


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
    """SeeHydro - 南水北调中线渠道水面识别与辅助分析工具."""


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
        typer.Option("--source", help="数据源：osm / geojson / shapefile"),
    ] = "osm",
    output: Annotated[
        Path,
        typer.Option("--output", help="输出路径"),
    ] = Path("data/route"),
    path: Annotated[
        Path | None,
        typer.Option("--path", help="本地线路文件路径（source=geojson/shapefile 时必填）"),
    ] = None,
) -> None:
    """下载南水北调中线线路数据。"""
    source = source.strip().lower()
    if source not in {"osm", "geojson", "shapefile"}:
        typer.echo(
            f"错误：--source 只能是 osm / geojson / shapefile，当前值为 {source!r}",
            err=True,
        )
        raise typer.Exit(code=1)

    logger.info("开始执行线路数据加载 ... source={}, path={}, output={}", source, path, output)
    route_gdf = load_route(source=source, path=path)

    if output.suffix.lower() in {".geojson", ".json"}:
        output_path = output
    elif output.suffix.lower() == ".shp":
        output_path = output
    else:
        output_path = output / "snbd_centerline.geojson"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    driver = "GeoJSON" if output_path.suffix.lower() in {".geojson", ".json"} else "ESRI Shapefile"
    route_gdf.to_file(output_path, driver=driver)

    typer.echo(f"线路数据已保存到 {output_path}，共 {len(route_gdf)} 条要素")


@download_app.command("sentinel2")
def download_sentinel2(
    config: Annotated[
        Path,
        typer.Option("--config", help="下载配置文件路径"),
    ] = Path("configs/default.yaml"),
    bbox: Annotated[
        str | None,
        typer.Option("--bbox", help="下载范围 west,south,east,north；不传则尝试按线路缓冲下载"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="输出 GeoTIFF 路径或输出目录"),
    ] = None,
) -> None:
    """下载 Sentinel-2 多光谱影像（通过 Google Earth Engine）。"""
    from seehydro.acquisition.gee import GEEDownloader

    try:
        import ee
        import geopandas as gpd
        from shapely.geometry import box, mapping
    except ImportError as exc:
        typer.echo(f"错误：缺少 Sentinel-2 下载依赖: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    logger.info("开始执行 Sentinel-2 数据下载 ... config={}, bbox={}, output={}", config, bbox, output)
    cfg = load_config(config)
    s2_cfg = cfg.get("sentinel2", {})

    downloader = GEEDownloader(project_id=s2_cfg.get("project_id"))
    date_range = tuple(s2_cfg.get("date_range", ["2024-01-01", "2025-12-31"]))
    cloud_pct_max = int(s2_cfg.get("cloud_pct_max", 10))
    bands = list(s2_cfg.get("bands", ["B2", "B3", "B4", "B8", "B11"]))
    resolution = int(s2_cfg.get("resolution", 10))

    if bbox:
        west, south, east, north = _parse_bbox(bbox)
        geometry = ee.Geometry(mapping(box(west, south, east, north)))
        output_path = output or Path(s2_cfg.get("output_dir", "data/sentinel2")) / "sentinel2_bbox.tif"
        if output_path.suffix.lower() not in {".tif", ".tiff"}:
            output_path = output_path / "sentinel2_bbox.tif"

        image = downloader.get_sentinel2(
            geometry=geometry,
            date_range=date_range,
            cloud_pct_max=cloud_pct_max,
            bands=bands,
        )
        saved = downloader.download_image(image=image, geometry=geometry, output_path=output_path, scale=resolution)
        typer.echo(f"Sentinel-2 下载完成：{saved}")
        return

    route_cfg = cfg.get("route", {})
    source = str(route_cfg.get("source", "osm"))
    local_path = route_cfg.get("local_path")
    route_path = None if local_path is None else Path(local_path)

    try:
        route_gdf = load_route(source=source, path=route_path if route_path and route_path.exists() else None)
    except Exception:
        if route_path and route_path.exists():
            raise
        route_gdf = load_route(source="osm")

    output_dir = output or Path(s2_cfg.get("output_dir", "data/sentinel2"))
    if output_dir.suffix.lower() in {".tif", ".tiff"}:
        output_dir = output_dir.parent

    downloaded = downloader.download_by_segments(
        route_gdf=route_gdf if isinstance(route_gdf, gpd.GeoDataFrame) else route_gdf,
        segment_length_m=float(s2_cfg.get("segment_length", 10000)),
        buffer_m=float(s2_cfg.get("buffer_width", 2000)),
        date_range=date_range,
        output_dir=output_dir,
    )
    typer.echo(f"Sentinel-2 分段下载完成，共 {len(downloaded)} 个文件，输出目录：{output_dir}")


@download_app.command("tiles")
def download_tiles(
    bbox: Annotated[
        str,
        typer.Option("--bbox", help="下载范围 west,south,east,north"),
    ],
    provider: Annotated[
        str,
        typer.Option("--provider", help="瓦片服务商：tianditu_img / google_satellite"),
    ] = "tianditu_img",
    zoom: Annotated[
        int,
        typer.Option("--zoom", help="瓦片级别"),
    ] = 18,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="输出目录"),
    ] = Path("raw_geotiff"),
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="天地图 key；不传则读取 TDT_KEY"),
    ] = None,
) -> None:
    """下载在线地图瓦片并拼接为 GeoTIFF，优先用于天地图底图抓取。"""
    from seehydro.acquisition.tile_downloader import TileDownloader

    bounds = _parse_bbox(bbox)
    resolved_key = api_key or os.getenv("TDT_KEY")
    if provider.startswith("tianditu") and not resolved_key:
        typer.echo("错误：天地图下载需要 --api-key 或环境变量 TDT_KEY", err=True)
        raise typer.Exit(code=1)

    logger.info(
        "开始执行在线瓦片下载 ... provider={}, zoom={}, bbox={}, output_dir={}",
        provider, zoom, bounds, output_dir,
    )
    downloader = TileDownloader(provider=provider, api_key=resolved_key)
    saved = downloader.download_tiles(bounds=bounds, zoom=zoom, output_dir=output_dir)
    typer.echo(f"瓦片下载并拼接完成：{saved}")


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
    from seehydro.preprocessing.clip import batch_clip

    if not input.exists():
        typer.echo(f"错误：输入路径不存在 {input}", err=True)
        raise typer.Exit(code=1)
    if not route.exists():
        typer.echo(f"错误：线路文件不存在 {route}", err=True)
        raise typer.Exit(code=1)
    if buffer <= 0:
        typer.echo(f"错误：--buffer 必须大于 0，当前值为 {buffer}", err=True)
        raise typer.Exit(code=1)

    route_suffix = route.suffix.lower()
    if route_suffix in {".geojson", ".json"}:
        route_gdf = load_route(source="geojson", path=route)
    elif route_suffix == ".shp":
        route_gdf = load_route(source="shapefile", path=route)
    else:
        typer.echo(f"错误：不支持的线路文件格式 {route.suffix}", err=True)
        raise typer.Exit(code=1)

    logger.info(
        "开始执行影像裁剪 ... input={}, route={}, buffer={}m, output={}",
        input, route, buffer, output,
    )
    clipped_files = batch_clip(
        raster_dir=input,
        route_gdf=route_gdf,
        buffer_m=buffer,
        output_dir=output,
    )
    typer.echo(f"裁剪完成，共输出 {len(clipped_files)} 个文件到 {output}")


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


@train_app.command("prepare-seg-data")
def prepare_segmentation_data(
    labelme_dir: Annotated[
        Path,
        typer.Option("--labelme-dir", help="Labelme 标注目录，包含 json 和对应原图"),
    ] = Path("labelme_work"),
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="输出训练数据根目录"),
    ] = Path("data/seg_water"),
    water_label: Annotated[
        str,
        typer.Option("--water-label", help="Labelme 中代表水面的标签名"),
    ] = "water",
    num_classes: Annotated[
        int,
        typer.Option("--num-classes", help="分割类别总数"),
    ] = 2,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="仅检查转换输入，不写训练数据"),
    ] = False,
) -> None:
    """把 Labelme 标注转换成训练数据并自动做一次数据检查。"""
    project_root = Path(__file__).resolve().parents[2]
    convert_script = project_root / "scripts" / "convert_labelme_to_masks.py"
    validate_script = project_root / "scripts" / "validate_seg_dataset.py"

    if not labelme_dir.exists():
        typer.echo(f"错误：Labelme 目录不存在 {labelme_dir}", err=True)
        raise typer.Exit(code=1)

    logger.info(
        "开始整理分割训练数据 ... labelme_dir={}, output_root={}, water_label={}, num_classes={}, dry_run={}",
        labelme_dir, output_root, water_label, num_classes, dry_run,
    )

    convert_args = [
        "--input-dir", str(labelme_dir),
        "--output-root", str(output_root),
        "--water-label", water_label,
    ]
    if dry_run:
        convert_args.append("--dry-run")

    convert_result = _run_local_script(convert_script, convert_args)
    if convert_result.stdout:
        typer.echo(convert_result.stdout.rstrip())
    if convert_result.returncode != 0:
        if convert_result.stderr:
            typer.echo(convert_result.stderr.rstrip(), err=True)
        raise typer.Exit(code=convert_result.returncode)

    if dry_run:
        return

    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    validate_args = [
        "--image-dir", str(image_dir),
        "--mask-dir", str(mask_dir),
        "--num-classes", str(num_classes),
    ]
    validate_result = _run_local_script(validate_script, validate_args)
    if validate_result.stdout:
        typer.echo(validate_result.stdout.rstrip())
    if validate_result.returncode != 0:
        if validate_result.stderr:
            typer.echo(validate_result.stderr.rstrip(), err=True)
        raise typer.Exit(code=validate_result.returncode)


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
        Path | None,
        typer.Option("--model-seg", help="分割模型权重文件路径（.pth）"),
    ] = None,
    model_det: Annotated[
        Path | None,
        typer.Option("--model-det", help="检测模型权重文件路径（.pt）"),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="分割模型配置文件路径，用于按训练配置加载推理模型"),
    ] = Path("configs/segmentation_binary_water.yaml"),
    output: Annotated[
        Path,
        typer.Option("--output", help="推理结果输出目录"),
    ] = Path("outputs/infer"),
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="推理批次大小"),
    ] = 4,
    device: Annotated[
        str | None,
        typer.Option("--device", help="推理设备：cpu / cuda / cuda:0 / cuda:1 ...；默认自动选择"),
    ] = None,
) -> None:
    """对切片影像执行分割与检测推理，生成分割掩膜和检测结果。"""
    from seehydro.models.inference import InferencePipeline

    has_seg = model_seg is not None and model_seg.exists()
    has_det = model_det is not None and model_det.exists()

    if not has_seg and not has_det:
        typer.echo("错误：至少提供一个存在的模型权重：--model-seg 或 --model-det", err=True)
        raise typer.Exit(code=1)
    if not input.exists():
        typer.echo(f"错误：输入路径不存在 {input}", err=True)
        raise typer.Exit(code=1)

    import torch

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "开始执行推理预测 ... input={}, model_seg={}, model_det={}, config={}, output={}, batch_size={}, device={}",
        input, model_seg, model_det, config, output, batch_size, resolved_device,
    )
    seg_cfg = _load_seg_inference_config(config) if has_seg else {}
    pipeline = InferencePipeline(
        seg_model_path=model_seg if has_seg else None,
        det_model_path=model_det if has_det else None,
        seg_config=seg_cfg if has_seg else None,
        device=resolved_device,
    )

    normalize_method = str(seg_cfg.get("normalize_method", "percentile"))
    result = pipeline.run_full_pipeline(
        tile_dir=input,
        output_dir=output,
        normalize_method=normalize_method,
        batch_size=batch_size,
    )
    summary: dict[str, object] = {}

    if "segmentation" in result:
        seg_output = output / "segmentation"
        summary["segmentation_output_dir"] = str(seg_output)
        summary["segmentation_count"] = len(result["segmentation"])

        tile_index_path = input / "tile_index.csv"
        if tile_index_path.exists():
            from seehydro.preprocessing.tiling import TileGenerator
            from seehydro.utils.raster_utils import read_raster

            generator = TileGenerator()
            tile_infos = generator.load_tile_index(tile_index_path)
            grouped_infos: dict[Path, list] = {}
            for info in tile_infos:
                grouped_infos.setdefault(info.source_path, []).append(info)

            merged_outputs = []
            merged_dir = output / "merged"
            for source_path, source_tile_infos in grouped_infos.items():
                source_data, source_profile = read_raster(source_path)
                source_profile.update(
                    height=source_data.shape[1],
                    width=source_data.shape[2],
                    count=1,
                    dtype="uint8",
                    nodata=255,
                )
                merged_path = merged_dir / f"{source_path.stem}_merged_mask.tif"
                generator.reassemble(
                    tiles=result["segmentation"],
                    tile_infos=source_tile_infos,
                    output_path=merged_path,
                    original_profile=source_profile,
                )
                merged_outputs.append(str(merged_path))

            summary["merged_segmentation_outputs"] = merged_outputs

    if "detection" in result:
        det_output = output / "detection.json"
        det_output.parent.mkdir(parents=True, exist_ok=True)
        det_serializable = {k: v for k, v in result["detection"].items()}
        det_output.write_text(json.dumps(det_serializable, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["detection_output_json"] = str(det_output)
        summary["detection_tile_count"] = len(result["detection"])

    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


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
    """从分割掩膜中提取中心线与估算水面宽度等基础结果。"""
    import geopandas as gpd
    from shapely.geometry import LineString

    from seehydro.extraction.canal_params import extract_canal_params
    from seehydro.export.report import generate_summary_report, save_report
    from seehydro.export.vector_io import save_geodataframe

    if not input.exists():
        typer.echo(f"错误：输入路径不存在 {input}", err=True)
        raise typer.Exit(code=1)
    if sample_interval <= 0:
        typer.echo(f"错误：--sample-interval 必须大于 0，当前值为 {sample_interval}", err=True)
        raise typer.Exit(code=1)

    logger.info(
        "开始执行参数提取 ... input={}, output={}, sample_interval={}m",
        input, output, sample_interval,
    )
    if input.is_dir():
        merged_masks = sorted(input.glob("*_merged_mask.tif")) + sorted(input.glob("*_merged_mask.tiff"))
        mask_files = merged_masks if merged_masks else []
    else:
        mask_files = [input]

    if not mask_files:
        typer.echo(f"错误：未在 {input} 找到可提取的回拼掩膜（*_merged_mask.tif/tiff）", err=True)
        raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)
    vectors_dir = output / "vectors"
    reports_dir = output / "reports"

    summary_rows = []
    saved_vector_files = []

    for mask_path in mask_files:
        canal_params = extract_canal_params(mask_path=mask_path, interval_m=sample_interval)
        if not canal_params:
            logger.warning("跳过无法提取水面辅助参数的掩膜: {}", mask_path)
            continue

        prefix = mask_path.stem
        mask_crs = canal_params.get("crs", "EPSG:4326")

        if "centerline" in canal_params and isinstance(canal_params["centerline"], LineString):
            centerline_gdf = gpd.GeoDataFrame(
                [{
                    "name": prefix,
                    "mean_estimated_water_surface_width_m": canal_params.get("mean_estimated_water_surface_width_m", 0.0),
                }],
                geometry=[canal_params["centerline"]],
                crs=mask_crs,
            )
            saved_vector_files.append(save_geodataframe(centerline_gdf, vectors_dir / f"{prefix}_centerline.geojson"))

        if "width_profile" in canal_params and len(canal_params["width_profile"]) > 0:
            saved_vector_files.append(
                save_geodataframe(canal_params["width_profile"], vectors_dir / f"{prefix}_width_profile.geojson")
            )

        if "berm_width_profile" in canal_params and len(canal_params["berm_width_profile"]) > 0:
            saved_vector_files.append(
                save_geodataframe(canal_params["berm_width_profile"], vectors_dir / f"{prefix}_berm_width_profile.geojson")
            )

        summary_rows.append({
            "mask": str(mask_path),
            "mean_estimated_water_surface_width_m": canal_params.get("mean_estimated_water_surface_width_m", 0.0),
            "sample_count": len(canal_params.get("width_profile", [])),
            "mean_estimated_berm_width_m": canal_params.get("mean_estimated_berm_width_m", 0.0),
        })

        summary_df = generate_summary_report(canal_params=canal_params)
        report_name = f"{prefix}_summary"
        save_report(summary_df, reports_dir, name=report_name)

    if not summary_rows:
        typer.echo("未能从输入掩膜中提取到有效水面辅助参数", err=True)
        raise typer.Exit(code=1)

    summary_json_path = output / "summary.json"
    summary_json_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(
        json.dumps(
            {
                "processed_masks": len(summary_rows),
                "vector_outputs": len(saved_vector_files),
                "summary_json": str(summary_json_path),
                "vectors_dir": str(vectors_dir),
                "reports_dir": str(reports_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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
    """将已提取的辅助分析结果导出为矢量文件，并可选生成汇总报告。"""
    from seehydro.export.report import save_report
    from seehydro.export.vector_io import export_all_results

    if fmt not in _VALID_FORMATS:
        typer.echo(
            f"错误：--format 只能是 geojson 或 shapefile，当前值为 {fmt!r}",
            err=True,
        )
        raise typer.Exit(code=1)
    if not input.exists():
        typer.echo(f"错误：输入路径不存在 {input}", err=True)
        raise typer.Exit(code=1)

    logger.info("开始执行结果导出 ... input={}, format={}, report={}", input, fmt, report)
    import geopandas as gpd
    import pandas as pd

    geojson_files = sorted(input.glob("*.geojson")) if input.is_dir() else [input]
    if not geojson_files:
        typer.echo(f"错误：未在 {input} 找到可导出的 GeoJSON 文件", err=True)
        raise typer.Exit(code=1)

    results = {}
    for path in geojson_files:
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("跳过无法读取的矢量文件 {}: {}", path, exc)
            continue
        results[path.stem] = gdf

    if not results:
        typer.echo("错误：没有可导出的有效矢量结果", err=True)
        raise typer.Exit(code=1)

    export_dir = input / f"export_{fmt}" if input.is_dir() else input.parent / f"export_{fmt}"
    saved_files = export_all_results(results=results, output_dir=export_dir, formats=[fmt])

    report_files = []
    if report is not None:
        summary_json = input.parent / "summary.json" if input.is_dir() else input.parent / "summary.json"
        if summary_json.exists():
            summary_records = json.loads(summary_json.read_text(encoding="utf-8"))
            summary_df = pd.DataFrame(summary_records)
            report_files = [str(p) for p in save_report(summary_df, report, name="extract_summary")]

    typer.echo(
        json.dumps(
            {
                "exported_files": [str(p) for p in saved_files],
                "report_files": report_files,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# pipeline（端到端流程）
# ---------------------------------------------------------------------------


@pipeline_app.command("quickstart")
def pipeline_quickstart(
    bbox: Annotated[
        str | None,
        typer.Option("--bbox", help="下载范围 west,south,east,north；不传则跳过在线下载"),
    ] = None,
    provider: Annotated[
        str,
        typer.Option("--provider", help="在线底图来源：tianditu_img / google_satellite"),
    ] = "tianditu_img",
    zoom: Annotated[
        int,
        typer.Option("--zoom", help="在线底图瓦片级别"),
    ] = 18,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="天地图 key；不传则读取 TDT_KEY"),
    ] = None,
    raw_input: Annotated[
        Path,
        typer.Option("--raw-input", help="原始 GeoTIFF 输入目录"),
    ] = Path("raw_geotiff"),
    route: Annotated[
        Path | None,
        typer.Option("--route", help="线路文件路径；传入后执行沿线路裁剪"),
    ] = None,
    buffer: Annotated[
        float,
        typer.Option("--buffer", help="线路裁剪缓冲区宽度（米）"),
    ] = 2000.0,
    tile_size: Annotated[
        int,
        typer.Option("--tile-size", help="切片大小"),
    ] = 512,
    overlap: Annotated[
        float,
        typer.Option("--overlap", help="切片重叠比例"),
    ] = 0.25,
    model_seg: Annotated[
        Path | None,
        typer.Option("--model-seg", help="已有分割模型权重；提供后执行推理"),
    ] = None,
    labelme_dir: Annotated[
        Path | None,
        typer.Option("--labelme-dir", help="可选：Labelme 标注目录；提供后自动整理训练数据"),
    ] = None,
    seg_output_root: Annotated[
        Path,
        typer.Option("--seg-output-root", help="整理后的训练数据输出根目录"),
    ] = Path("data/seg_water"),
    config: Annotated[
        Path | None,
        typer.Option("--config", help="分割模型配置文件路径，用于按训练配置执行推理"),
    ] = Path("configs/segmentation_binary_water.yaml"),
    sample_interval: Annotated[
        float,
        typer.Option("--sample-interval", help="提参采样间距（米）"),
    ] = 50.0,
    device: Annotated[
        str | None,
        typer.Option("--device", help="推理设备；默认自动选择 cuda 或 cpu"),
    ] = None,
    workspace: Annotated[
        Path,
        typer.Option("--workspace", help="流程输出工作目录"),
    ] = Path("outputs/pipeline_run"),
) -> None:
    """快速串联下载、裁剪、切片、训练数据整理、推理、提取基础辅助结果的最小工程流程."""
    from seehydro.acquisition.tile_downloader import TileDownloader
    from seehydro.preprocessing.clip import batch_clip
    from seehydro.preprocessing.tiling import TileGenerator

    workspace.mkdir(parents=True, exist_ok=True)
    raw_dir = workspace / "raw_geotiff"
    clipped_dir = workspace / "clipped"
    tiles_dir = workspace / "tiles"
    infer_dir = workspace / "infer"
    extract_dir = workspace / "extraction"

    outputs: dict[str, object] = {
        "workspace": str(workspace),
    }

    current_input_dir = raw_input

    if bbox:
        bounds = _parse_bbox(bbox)
        resolved_key = api_key or os.getenv("TDT_KEY")
        if provider.startswith("tianditu") and not resolved_key:
            typer.echo("错误：天地图下载需要 --api-key 或环境变量 TDT_KEY", err=True)
            raise typer.Exit(code=1)

        raw_dir.mkdir(parents=True, exist_ok=True)
        downloader = TileDownloader(provider=provider, api_key=resolved_key)
        downloaded_path = downloader.download_tiles(bounds=bounds, zoom=zoom, output_dir=raw_dir)
        current_input_dir = raw_dir
        outputs["downloaded_geotiff"] = str(downloaded_path)
    elif not raw_input.exists():
        typer.echo("错误：未提供 --bbox，且 --raw-input 不存在，无法开始流程", err=True)
        raise typer.Exit(code=1)

    if route is not None:
        if not route.exists():
            typer.echo(f"错误：线路文件不存在 {route}", err=True)
            raise typer.Exit(code=1)

        route_suffix = route.suffix.lower()
        if route_suffix in {".geojson", ".json"}:
            route_gdf = load_route(source="geojson", path=route)
        elif route_suffix == ".shp":
            route_gdf = load_route(source="shapefile", path=route)
        else:
            typer.echo(f"错误：不支持的线路文件格式 {route.suffix}", err=True)
            raise typer.Exit(code=1)

        clipped_files = batch_clip(
            raster_dir=current_input_dir,
            route_gdf=route_gdf,
            buffer_m=buffer,
            output_dir=clipped_dir,
        )
        outputs["clipped_files"] = [str(p) for p in clipped_files]
        current_input_dir = clipped_dir

    if labelme_dir is not None and not labelme_dir.exists():
        typer.echo(f"错误：Labelme 目录不存在 {labelme_dir}", err=True)
        raise typer.Exit(code=1)

    generator = TileGenerator(tile_size=tile_size, overlap=overlap)
    input_files = sorted(
        p for p in Path(current_input_dir).iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    if not input_files:
        typer.echo(f"错误：未在 {current_input_dir} 找到可切片 tif/tiff 影像", err=True)
        raise typer.Exit(code=1)

    all_tile_infos = []
    for image_path in input_files:
        tile_infos = generator.generate_tiles(image_path=image_path, output_dir=tiles_dir, prefix=image_path.stem)
        all_tile_infos.extend(tile_infos)

    tile_index_path = tiles_dir / "tile_index.csv"
    generator.save_tile_index(all_tile_infos, tile_index_path)
    outputs["tile_count"] = len(all_tile_infos)
    outputs["tile_index"] = str(tile_index_path)

    if labelme_dir is not None:
        project_root = Path(__file__).resolve().parents[2]
        convert_script = project_root / "scripts" / "convert_labelme_to_masks.py"
        validate_script = project_root / "scripts" / "validate_seg_dataset.py"

        convert_result = _run_local_script(
            convert_script,
            [
                "--input-dir", str(labelme_dir),
                "--output-root", str(seg_output_root),
                "--water-label", "water",
            ],
        )
        if convert_result.stdout:
            typer.echo(convert_result.stdout.rstrip())
        if convert_result.returncode != 0:
            if convert_result.stderr:
                typer.echo(convert_result.stderr.rstrip(), err=True)
            raise typer.Exit(code=convert_result.returncode)

        validate_result = _run_local_script(
            validate_script,
            [
                "--image-dir", str(seg_output_root / "images"),
                "--mask-dir", str(seg_output_root / "masks"),
                "--num-classes", "2",
            ],
        )
        if validate_result.stdout:
            typer.echo(validate_result.stdout.rstrip())
        if validate_result.returncode != 0:
            if validate_result.stderr:
                typer.echo(validate_result.stderr.rstrip(), err=True)
            raise typer.Exit(code=validate_result.returncode)

        outputs["prepared_segmentation_dataset"] = str(seg_output_root)

    if model_seg is not None:
        has_seg = model_seg.exists()
        if not has_seg:
            typer.echo(f"错误：分割模型不存在 {model_seg}", err=True)
            raise typer.Exit(code=1)

        from seehydro.models.inference import InferencePipeline
        from seehydro.utils.raster_utils import read_raster

        seg_cfg = _load_seg_inference_config(config)
        import torch

        pipeline_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = InferencePipeline(seg_model_path=model_seg, seg_config=seg_cfg, device=pipeline_device)
        seg_results = pipeline.run_segmentation(
            tile_dir=tiles_dir,
            output_dir=infer_dir / "segmentation",
            normalize_method=str(seg_cfg.get("normalize_method", "percentile")),
            batch_size=1,
        )
        outputs["segmentation_count"] = len(seg_results)
        outputs["segmentation_output_dir"] = str(infer_dir / "segmentation")

        tile_infos = generator.load_tile_index(tile_index_path)
        grouped_infos: dict[Path, list] = {}
        for info in tile_infos:
            grouped_infos.setdefault(info.source_path, []).append(info)

        merged_outputs = []
        merged_dir = infer_dir / "merged"
        for source_path, source_tile_infos in grouped_infos.items():
            source_data, source_profile = read_raster(source_path)
            source_profile.update(
                height=source_data.shape[1],
                width=source_data.shape[2],
                count=1,
                dtype="uint8",
                nodata=255,
            )
            merged_path = merged_dir / f"{source_path.stem}_merged_mask.tif"
            generator.reassemble(
                tiles=seg_results,
                tile_infos=source_tile_infos,
                output_path=merged_path,
                original_profile=source_profile,
            )
            merged_outputs.append(str(merged_path))
        outputs["merged_segmentation_outputs"] = merged_outputs

        if merged_outputs:
            from seehydro.extraction.canal_params import extract_canal_params
            from seehydro.export.report import generate_summary_report, save_report
            from seehydro.export.vector_io import save_geodataframe
            import geopandas as gpd
            from shapely.geometry import LineString

            vectors_dir = extract_dir / "vectors"
            reports_dir = extract_dir / "reports"
            summary_rows = []

            for merged_mask in merged_outputs:
                merged_mask_path = Path(merged_mask)
                canal_params = extract_canal_params(merged_mask_path, interval_m=sample_interval)
                if not canal_params:
                    continue

                prefix = merged_mask_path.stem
                if "centerline" in canal_params and isinstance(canal_params["centerline"], LineString):
                    centerline_gdf = gpd.GeoDataFrame(
                        [{
                            "name": prefix,
                            "mean_estimated_water_surface_width_m": canal_params.get("mean_estimated_water_surface_width_m", 0.0),
                        }],
                        geometry=[canal_params["centerline"]],
                        crs=canal_params.get("crs", "EPSG:4326"),
                    )
                    save_geodataframe(centerline_gdf, vectors_dir / f"{prefix}_centerline.geojson")
                if "width_profile" in canal_params and len(canal_params["width_profile"]) > 0:
                    save_geodataframe(canal_params["width_profile"], vectors_dir / f"{prefix}_width_profile.geojson")

                summary_rows.append({
                    "mask": str(merged_mask_path),
                    "mean_estimated_water_surface_width_m": canal_params.get("mean_estimated_water_surface_width_m", 0.0),
                    "sample_count": len(canal_params.get("width_profile", [])),
                })
                report_df = generate_summary_report(canal_params=canal_params)
                save_report(report_df, reports_dir, name=f"{prefix}_summary")

            if summary_rows:
                summary_json_path = extract_dir / "summary.json"
                summary_json_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
                outputs["extraction_summary"] = str(summary_json_path)
                outputs["extraction_vectors_dir"] = str(vectors_dir)
                outputs["extraction_reports_dir"] = str(reports_dir)

    typer.echo(json.dumps(outputs, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
