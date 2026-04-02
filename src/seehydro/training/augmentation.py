"""数据增强配置."""

import albumentations as A


def get_seg_train_transform(input_size: int = 512) -> A.Compose:
    """分割训练数据增强."""
    return A.Compose([
        A.PadIfNeeded(min_height=input_size, min_width=input_size, border_mode=0, fill=0, fill_mask=0, p=1.0),
        A.RandomCrop(height=input_size, width=input_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
    ])


def get_seg_val_transform(input_size: int = 512) -> A.Compose:
    """分割验证数据增强（仅裁剪）."""
    return A.Compose([
        A.PadIfNeeded(min_height=input_size, min_width=input_size, border_mode=0, fill=0, fill_mask=0, p=1.0),
        A.CenterCrop(height=input_size, width=input_size, p=1.0),
    ])
