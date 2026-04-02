"""分割模型训练脚本."""

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, random_split

from seehydro.models.seg_model import SegmentationModel
from seehydro.training.augmentation import get_seg_train_transform, get_seg_val_transform
from seehydro.training.dataset import SegmentationDataset
from seehydro.training.metrics import SegmentationMetrics


class DiceCELoss(nn.Module):
    """Dice + CrossEntropy 组合损失."""

    def __init__(self, num_classes: int, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        # Dice loss
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        total = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (total + 1e-6)
        dice_loss = 1 - dice.mean()

        return (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss


def train_segmentation(
    image_dir: str | Path,
    mask_dir: str | Path,
    config: dict,
    output_dir: str | Path = "models/trained",
) -> Path:
    """训练分割模型.

    Args:
        image_dir: 影像切片目录
        mask_dir: 掩膜目录
        config: 训练配置字典
        output_dir: 模型输出目录

    Returns:
        最佳模型权重路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置参数
    model_name = config.get("model_name", "DeepLabV3Plus")
    encoder = config.get("encoder", "resnet101")
    in_channels = config.get("in_channels", 3)
    num_classes = config.get("num_classes", 5)
    input_size = config.get("input_size", 512)
    batch_size = config.get("batch_size", 8)
    epochs = config.get("epochs", 100)
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    patience = config.get("early_stopping_patience", 15)
    val_split = config.get("val_split", 0.2)
    random_seed = config.get("random_seed", 42)
    num_workers = config.get("num_workers", 4)
    encoder_weights = config.get("encoder_weights", "imagenet")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"设备: {device}")

    # 数据集
    train_transform = get_seg_train_transform(input_size)
    val_transform = get_seg_val_transform(input_size)

    full_dataset = SegmentationDataset(image_dir, mask_dir, in_channels=in_channels)
    if len(full_dataset) < 2:
        raise ValueError("分割训练至少需要 2 个样本，才能划分训练集和验证集。")

    val_size = max(1, int(len(full_dataset) * val_split))
    if val_size >= len(full_dataset):
        val_size = 1
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed)
    )

    # random_split 返回的 Subset 共享同一个 dataset，需要拆成独立实例避免 transform 串用。
    train_dataset = deepcopy(full_dataset)
    train_dataset.image_files = [full_dataset.image_files[i] for i in train_subset.indices]
    train_dataset.mask_files = [full_dataset.mask_files[i] for i in train_subset.indices]
    train_dataset.transform = train_transform

    val_dataset = deepcopy(full_dataset)
    val_dataset.image_files = [full_dataset.image_files[i] for i in val_subset.indices]
    val_dataset.mask_files = [full_dataset.mask_files[i] for i in val_subset.indices]
    val_dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"训练集: {train_size}, 验证集: {val_size}")

    # 模型
    model = SegmentationModel(model_name, encoder, encoder_weights, in_channels, num_classes, device)

    # 损失和优化器
    criterion = DiceCELoss(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 训练循环
    best_miou = 0.0
    no_improve_count = 0
    best_path = output_dir / "seg_best.pth"

    for epoch in range(epochs):
        # 训练
        model.model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model.model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # 验证
        model.model.eval()
        metrics = SegmentationMetrics(num_classes)
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model.model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                targets = masks.cpu().numpy()
                for i in range(preds.shape[0]):
                    metrics.update(preds[i], targets[i])

        val_loss /= len(val_loader)
        result = metrics.compute()
        miou = result["miou"]

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
            f"mIoU: {miou:.4f}, PA: {result['pixel_accuracy']:.4f}"
        )

        # Early stopping
        if miou > best_miou:
            best_miou = miou
            no_improve_count = 0
            model.save_weights(best_path)
            logger.info(f"保存最佳模型 mIoU={best_miou:.4f}")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    logger.info(f"训练完成，最佳mIoU: {best_miou:.4f}")
    return best_path
