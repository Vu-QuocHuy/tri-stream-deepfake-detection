"""Albumentations pipelines for training, validation, and TTA."""

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 380,
    use_heavy_augmentation: bool = False,
    augmentation_level: str = 'medium',
) -> A.Compose:
    """Build the training augmentation pipeline."""
    if use_heavy_augmentation:
        augmentation_level = 'heavy'

    _norm = [
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    if augmentation_level == 'light':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.7, 1.0),
                p=0.5,
            ),
        ] + _norm)

    if augmentation_level == 'medium':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=0.5,
            ),
            A.Rotate(limit=10, p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                A.Downscale(
                    scale_min=0.4,
                    scale_max=0.9,
                    interpolation={
                        "downscale": cv2.INTER_LINEAR,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=1.0,
                ),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.6),
            A.GaussNoise(var_limit=(5.0, 40.0), p=0.3),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.4,
            ),
        ] + _norm)

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.05),
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.5, 1.0),
            ratio=(0.9, 1.1),
            p=0.5,
        ),
        A.Rotate(limit=15, p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.3,
        ),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.1, p=1.0),
            A.GridDistortion(p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.ImageCompression(quality_lower=10, quality_upper=70, p=1.0),
            A.GaussianBlur(blur_limit=(3, 11), p=1.0),
            A.Downscale(
                scale_min=0.3,
                scale_max=0.8,
                interpolation={
                    "downscale": cv2.INTER_LINEAR,
                    "upscale": cv2.INTER_LINEAR,
                },
                p=1.0,
            ),
            A.MotionBlur(blur_limit=9, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
        ], p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.3,
        ),
        A.RGBShift(
            r_shift_limit=15,
            g_shift_limit=15,
            b_shift_limit=15,
            p=0.2,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.CLAHE(p=0.2),
        A.CoarseDropout(
            max_holes=8,
            max_height=image_size // 8,
            max_width=image_size // 8,
            fill_value=0,
            p=0.3,
        ),
    ] + _norm)


def get_val_transforms(image_size: int = 380) -> A.Compose:
    """Build the validation/test preprocessing pipeline."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_test_time_augmentation_transforms(image_size: int = 380) -> list:
    """Build test-time augmentation variants."""
    return [
        get_val_transforms(image_size),

        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),

        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.0, p=1.0),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),

        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.1, p=1.0),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]),
    ]
