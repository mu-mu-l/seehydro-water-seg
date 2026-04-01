"""统一推理管线，整合分割和检测模型."""

from pathlib import Path

import numpy as np
import rasterio
import torch
from loguru import logger
from tqdm import tqdm

from seehydro.models.det_model import DetectionModel
from seehydro.models.seg_model import SegmentationModel
from seehydro.preprocessing.normalize import normalize_image


class InferencePipeline:
    """统一推理管线，对切片进行分割和检测."""

    def __init__(
        self,
        seg_model_path: str | Path | None = None,
        det_model_path: str | Path | None = None,
        seg_config: dict | None = None,
        det_config: dict | None = None,
        device: str | None = None,
    ):
        """初始化推理管线.

        Args:
            seg_model_path: 分割模型权重路径
            det_model_path: 检测模型权重路径
            seg_config: 分割模型配置
            det_config: 检测模型配置
            device: 推理设备
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seg_model = None
        self.det_model = None

        if seg_model_path:
            cfg = seg_config or {}
            self.seg_model = SegmentationModel(
                model_name=cfg.get("model_name", "DeepLabV3Plus"),
                encoder=cfg.get("encoder", "resnet101"),
                encoder_weights=cfg.get("encoder_weights", "imagenet"),
                in_channels=cfg.get("in_channels", 3),
                num_classes=cfg.get("num_classes", 5),
                device=self.device,
            )
            self.seg_model.load_weights(seg_model_path)

        if det_model_path:
            self.det_model = DetectionModel(
                model_path=det_model_path,
                device=self.device,
                conf_threshold=(det_config or {}).get("conf_threshold", 0.25),
            )

    def run_segmentation(
        self,
        tile_dir: str | Path,
        output_dir: str | Path,
        normalize_method: str = "percentile",
    ) -> dict[str, Path]:
        """对切片目录中所有影像运行分割推理.

        Args:
            tile_dir: 切片目录
            output_dir: 输出目录（分割掩膜）
            normalize_method: 归一化方法

        Returns:
            {tile_name: mask_path} 映射
        """
        if self.seg_model is None:
            raise RuntimeError("分割模型未加载")

        tile_dir = Path(tile_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tile_files = sorted(tile_dir.glob("*.tif"))
        results = {}

        for tile_path in tqdm(tile_files, desc="分割推理"):
            with rasterio.open(tile_path) as src:
                data = src.read().astype(np.float32)  # (C, H, W)
                profile = src.profile.copy()

            # 归一化
            data = normalize_image(data, method=normalize_method)

            # 只取前N个通道
            in_channels = self.seg_model.model.encoder._in_channels if hasattr(self.seg_model.model, 'encoder') else 3
            if data.shape[0] > in_channels:
                data = data[:in_channels]

            # 推理
            tensor = torch.from_numpy(data).float()
            mask = self.seg_model.predict(tensor)  # (H, W)

            # 保存掩膜
            mask_path = output_dir / tile_path.name
            profile.update(count=1, dtype="uint8", nodata=255)
            with rasterio.open(mask_path, "w", **profile) as dst:
                dst.write(mask.numpy().astype(np.uint8), 1)

            results[tile_path.stem] = mask_path

        logger.info(f"分割推理完成: {len(results)} 个切片")
        return results

    def run_detection(
        self,
        tile_dir: str | Path,
        conf: float | None = None,
    ) -> dict[str, list[dict]]:
        """对切片目录中所有影像运行目标检测.

        Returns:
            {tile_name: [detection_dict, ...]} 映射
        """
        if self.det_model is None:
            raise RuntimeError("检测模型未加载")

        tile_dir = Path(tile_dir)
        tile_files = sorted(tile_dir.glob("*.tif"))
        results = {}

        for tile_path in tqdm(tile_files, desc="检测推理"):
            detections = self.det_model.predict(str(tile_path), conf=conf)
            if detections:
                results[tile_path.stem] = detections

        logger.info(f"检测推理完成: {len(tile_files)} 个切片, 检出 {sum(len(v) for v in results.values())} 个目标")
        return results

    def run_full_pipeline(
        self,
        tile_dir: str | Path,
        output_dir: str | Path,
        normalize_method: str = "percentile",
    ) -> dict:
        """运行完整推理管线（分割 + 检测）.

        Returns:
            {"segmentation": seg_results, "detection": det_results}
        """
        output_dir = Path(output_dir)
        result = {}

        if self.seg_model:
            seg_output = output_dir / "segmentation"
            result["segmentation"] = self.run_segmentation(tile_dir, seg_output, normalize_method=normalize_method)

        if self.det_model:
            result["detection"] = self.run_detection(tile_dir)

        return result
