"""训练与推理关键防护的回归测试."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from seehydro.models.inference import InferencePipeline
from seehydro.models.seg_model import SegmentationModel
from seehydro.training.augmentation import get_seg_train_transform, get_seg_val_transform
from seehydro.training.dataset import SegmentationDataset


def test_segmentation_dataset_transform_可独立设置而不串用(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    data = np.ones((3, 32, 32), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "height": 32,
        "width": 32,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(100.0, 30.0, 1.0, 1.0),
    }

    for idx in range(2):
        with rasterio.open(image_dir / f"sample_{idx}.tif", "w", **profile) as dst:
            dst.write(data)
        mask_profile = dict(profile)
        mask_profile.update(count=1)
        with rasterio.open(mask_dir / f"sample_{idx}.tif", "w", **mask_profile) as dst:
            dst.write(mask, 1)

    base_dataset = SegmentationDataset(image_dir, mask_dir, in_channels=3)
    train_dataset = SegmentationDataset(image_dir, mask_dir, in_channels=3, transform=get_seg_train_transform(64))
    val_dataset = SegmentationDataset(image_dir, mask_dir, in_channels=3, transform=get_seg_val_transform(64))

    assert base_dataset.transform is None
    assert train_dataset.transform is not val_dataset.transform


def test_segmentation_transforms_小图会先补边再裁剪() -> None:
    image = np.ones((32, 48, 3), dtype=np.uint8)
    mask = np.ones((32, 48), dtype=np.uint8)

    train_augmented = get_seg_train_transform(64)(image=image, mask=mask)
    val_augmented = get_seg_val_transform(64)(image=image, mask=mask)

    assert train_augmented["image"].shape == (64, 64, 3)
    assert train_augmented["mask"].shape == (64, 64)
    assert val_augmented["image"].shape == (64, 64, 3)
    assert val_augmented["mask"].shape == (64, 64)


def test_inference_pipeline_输入通道不足时提前报错(tmp_path: Path) -> None:
    tile_dir = tmp_path / "tiles"
    output_dir = tmp_path / "out"
    tile_dir.mkdir()

    profile = {
        "driver": "GTiff",
        "height": 32,
        "width": 32,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(100.0, 30.0, 1.0, 1.0),
    }
    with rasterio.open(tile_dir / "tile_0001.tif", "w", **profile) as dst:
        dst.write(np.ones((3, 32, 32), dtype=np.uint8))

    pipeline = InferencePipeline()
    pipeline.seg_model = SimpleNamespace(
        model=SimpleNamespace(encoder=SimpleNamespace(_in_channels=5)),
        predict=lambda tensor: torch.zeros((tensor.shape[-2], tensor.shape[-1]), dtype=torch.uint8),
    )

    with pytest.raises(ValueError, match="输入影像通道数不足"):
        pipeline.run_segmentation(tile_dir=tile_dir, output_dir=output_dir)


def test_inference_pipeline_支持_tiff_且按批推理(tmp_path: Path) -> None:
    tile_dir = tmp_path / "tiles"
    output_dir = tmp_path / "out"
    tile_dir.mkdir()

    profile = {
        "driver": "GTiff",
        "height": 16,
        "width": 16,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(100.0, 30.0, 1.0, 1.0),
    }
    for name in ("tile_a.tif", "tile_b.tiff"):
        with rasterio.open(tile_dir / name, "w", **profile) as dst:
            dst.write(np.ones((3, 16, 16), dtype=np.uint8))

    observed_batches: list[int] = []

    def fake_predict(tensor: torch.Tensor) -> torch.Tensor:
        observed_batches.append(int(tensor.shape[0]))
        return torch.zeros((tensor.shape[0], tensor.shape[-2], tensor.shape[-1]), dtype=torch.uint8)

    pipeline = InferencePipeline()
    pipeline.seg_model = SimpleNamespace(
        model=SimpleNamespace(encoder=SimpleNamespace(_in_channels=3)),
        predict=fake_predict,
    )

    results = pipeline.run_segmentation(tile_dir=tile_dir, output_dir=output_dir, batch_size=2)

    assert len(results) == 2
    assert observed_batches == [2]
    assert (output_dir / "tile_a.tif").exists()
    assert (output_dir / "tile_b.tiff").exists()


def test_segmentation_model_load_weights_兼容旧版torch参数() -> None:
    model = SegmentationModel.__new__(SegmentationModel)
    model.device = "cpu"
    model.model = SimpleNamespace(load_state_dict=lambda state_dict: state_dict)

    calls: list[dict[str, object]] = []

    def fake_torch_load(path, map_location=None, weights_only=None):  # noqa: ANN001
        calls.append(
            {
                "path": path,
                "map_location": map_location,
                "weights_only": weights_only,
            }
        )
        if weights_only is True:
            raise TypeError("weights_only unsupported")
        return {"ok": True}

    with patch("seehydro.models.seg_model.torch.load", side_effect=fake_torch_load):
        SegmentationModel.load_weights(model, "dummy.pth")

    assert calls[0]["weights_only"] is True
    assert calls[1]["weights_only"] is None
