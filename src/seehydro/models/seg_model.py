"""语义分割模型封装（DeepLabV3+ / U-Net）."""

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from loguru import logger


# 分割类别定义
SEG_CLASSES = {
    0: "background",
    1: "canal_water",      # 渠道水面
    2: "canal_slope",      # 渠道边坡
    3: "berm",             # 马道
    4: "service_road",     # 管理道路
}


def create_seg_model(
    model_name: str = "DeepLabV3Plus",
    encoder: str = "resnet101",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 5,
) -> nn.Module:
    """创建分割模型.

    Args:
        model_name: 模型名称，支持 DeepLabV3Plus / Unet / UnetPlusPlus / FPN
        encoder: 编码器骨干网络
        encoder_weights: 预训练权重
        in_channels: 输入通道数
        num_classes: 分割类别数
    """
    model_map = {
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
        "FPN": smp.FPN,
    }

    if model_name not in model_map:
        raise ValueError(f"不支持的模型: {model_name}，可选: {list(model_map.keys())}")

    model = model_map[model_name](
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    logger.info(f"创建分割模型: {model_name}, encoder={encoder}, in_channels={in_channels}, classes={num_classes}")
    return model


class SegmentationModel:
    """分割模型的高级封装，提供训练和推理接口."""

    def __init__(
        self,
        model_name: str = "DeepLabV3Plus",
        encoder: str = "resnet101",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 5,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_seg_model(model_name, encoder, encoder_weights, in_channels, num_classes)
        self.model.to(self.device)
        self.num_classes = num_classes
        logger.info(f"分割模型已加载到 {self.device}")

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """单张/批量推理.

        Args:
            image: (B, C, H, W) 或 (C, H, W) 的张量，值域 [0, 1]

        Returns:
            (B, H, W) 或 (H, W) 的类别掩膜
        """
        self.model.eval()
        single = image.dim() == 3
        if single:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        with torch.no_grad():
            logits = self.model(image)  # (B, C, H, W)
            preds = torch.argmax(logits, dim=1)  # (B, H, W)

        if single:
            preds = preds.squeeze(0)
        return preds.cpu()

    def predict_proba(self, image: torch.Tensor) -> torch.Tensor:
        """推理并返回概率图.

        Returns:
            (B, num_classes, H, W) 的概率张量
        """
        self.model.eval()
        single = image.dim() == 3
        if single:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        with torch.no_grad():
            logits = self.model(image)
            probs = torch.softmax(logits, dim=1)

        if single:
            probs = probs.squeeze(0)
        return probs.cpu()

    def load_weights(self, path: str | Path) -> None:
        """加载模型权重."""
        path = Path(path)
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            # 兼容不支持 weights_only 参数的旧版 PyTorch。
            state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"加载分割模型权重: {path}")

    def save_weights(self, path: str | Path) -> None:
        """保存模型权重."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"保存分割模型权重: {path}")
