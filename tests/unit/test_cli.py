"""CLI 基础行为测试."""

from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from seehydro.cli import _load_seg_inference_config, _parse_bbox, app


runner = CliRunner()


def _combined_output(result) -> str:
    return f"{result.stdout}\n{getattr(result, 'stderr', '')}"


def test_parse_bbox_合法输入_返回四元组() -> None:
    bounds = _parse_bbox("114.35,38.20,114.39,38.23")
    assert bounds == (114.35, 38.20, 114.39, 38.23)


def test_load_seg_inference_config_读取二分类训练配置() -> None:
    cfg = _load_seg_inference_config(PROJECT_ROOT / "configs" / "segmentation_binary_water.yaml")
    assert cfg["model_name"] == "Unet"
    assert cfg["encoder"] == "resnet18"
    assert cfg["num_classes"] == 2


def test_parse_bbox_非法顺序_抛出异常() -> None:
    try:
        _parse_bbox("114.39,38.23,114.35,38.20")
    except Exception as exc:  # noqa: BLE001
        assert "bbox 范围非法" in str(exc)
    else:
        raise AssertionError("应当抛出 bbox 非法异常")


def test_download_tiles_缺少天地图key_直接报错() -> None:
    result = runner.invoke(
        app,
        [
            "download",
            "tiles",
            "--bbox",
            "114.35,38.20,114.39,38.23",
            "--provider",
            "tianditu_img",
        ],
    )
    assert result.exit_code == 1
    assert "天地图下载需要 --api-key 或环境变量 TDT_KEY" in _combined_output(result)


def test_infer_缺少模型参数_直接报错(tmp_path: Path) -> None:
    input_dir = tmp_path / "tiles"
    input_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "infer",
            "--input",
            str(input_dir),
        ],
    )
    assert result.exit_code == 1
    assert "至少提供一个存在的模型权重" in _combined_output(result)


def test_preprocess_clip_缺少线路文件_直接报错(tmp_path: Path) -> None:
    input_dir = tmp_path / "rasters"
    input_dir.mkdir()
    route_path = tmp_path / "missing.geojson"

    result = runner.invoke(
        app,
        [
            "preprocess",
            "clip",
            "--input",
            str(input_dir),
            "--route",
            str(route_path),
        ],
    )
    assert result.exit_code == 1
    assert "线路文件不存在" in _combined_output(result)


def test_extract_输入不存在_直接报错(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    result = runner.invoke(
        app,
        [
            "extract",
            "--input",
            str(missing),
        ],
    )
    assert result.exit_code == 1
    assert "输入路径不存在" in _combined_output(result)


def test_extract_目录下没有回拼掩膜_直接报错(tmp_path: Path) -> None:
    input_dir = tmp_path / "infer"
    input_dir.mkdir()
    (input_dir / "plain.tif").write_bytes(b"dummy")

    result = runner.invoke(
        app,
        [
            "extract",
            "--input",
            str(input_dir),
        ],
    )
    assert result.exit_code == 1
    assert "回拼掩膜" in _combined_output(result)


def test_export_输入不存在_直接报错(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    result = runner.invoke(
        app,
        [
            "export",
            "--input",
            str(missing),
        ],
    )
    assert result.exit_code == 1
    assert "输入路径不存在" in _combined_output(result)


def test_pipeline_quickstart_缺少输入_直接报错(tmp_path: Path) -> None:
    missing_raw = tmp_path / "missing_raw"
    result = runner.invoke(
        app,
        [
            "pipeline",
            "quickstart",
            "--raw-input",
            str(missing_raw),
        ],
    )
    assert result.exit_code == 1
    assert "未提供 --bbox，且 --raw-input 不存在" in _combined_output(result)


def test_prepare_seg_data_labelme目录不存在_直接报错(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing_labelme"
    result = runner.invoke(
        app,
        [
            "train",
            "prepare-seg-data",
            "--labelme-dir",
            str(missing_dir),
        ],
    )
    assert result.exit_code == 1
    assert "Labelme 目录不存在" in _combined_output(result)


def test_pipeline_quickstart_labelme目录不存在_直接报错(tmp_path: Path) -> None:
    raw_input = tmp_path / "raw_geotiff"
    raw_input.mkdir()
    missing_labelme = tmp_path / "missing_labelme"

    result = runner.invoke(
        app,
        [
            "pipeline",
            "quickstart",
            "--raw-input",
            str(raw_input),
            "--labelme-dir",
            str(missing_labelme),
        ],
    )
    assert result.exit_code == 1
    assert "Labelme 目录不存在" in _combined_output(result)
