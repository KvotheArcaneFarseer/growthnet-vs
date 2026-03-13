"""
MONAI Transform Pipeline Definitions

This module provides transformation pipelines for medical image preprocessing and augmentation
using the MONAI framework. It includes functions for building transforms for various training
scenarios including supervised training, evaluation, synthetic data handling, and self-supervised
pretraining.

The transforms handle 3D medical images and perform operations such as:
- Spatial resampling and normalization
- Data augmentation (flipping, rotation, intensity adjustments)
- Cropping strategies (foreground-based, positive/negative sampling)
- Contrastive learning preprocessing

Functions:
    build_train_transform: Creates transform pipeline for supervised training
    build_eval_transform: Creates transform pipeline for model evaluation
    get_train_transform_synthetic: Creates transform pipeline for synthetic training data
    get_val_transform_synthetic: Creates transform pipeline for synthetic validation data
    build_pretrain_transform: Creates transform pipeline for self-supervised pretraining
    build_preval_transform: Creates transform pipeline for pretraining validation
"""

import numpy as np
import monai
from monai.transforms import (
    Compose,
    CopyItemsd,
    CropForegroundd, 
    DeleteItemsd, 
    EnsureTyped, 
    NormalizeIntensityd,
    OneOf,
    Orientationd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    RandFlipd, 
    RandRotated, 
    RandScaleIntensityd, 
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld, 
    RandZoomd, 
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandGaussianSharpend, 
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd, 
    Transform, 
)
from src.data.utils import CenterCropByLabeld, CenterCropByLabelDeterministicd

def build_train_transform(
        target_spacing: tuple[int, int, int] = (1.0, 1.0, 2.0),
        roi_size: tuple[int, int, int] = (128, 128, 64),
        pos_neg: list[float] = [3.0, 1.0],
        vit_patch_div: tuple[int, int, int] = (16, 16, 16)
) -> Transform:
    """
    Builds a composition of transforms for training on medical image data with labels.
    
    This pipeline performs spatial preprocessing (orientation, spacing, normalization),
    foreground cropping, spatial padding, and various data augmentation techniques
    including random flipping, rotation, and intensity adjustments. The pipeline
    uses positive/negative label sampling for patch extraction.
    
    Args:
        target_spacing : tuple[int, int, int] : Target voxel spacing in mm for resampling 
            (x, y, z). Default is (1.0, 1.0, 2.0).
        roi_size : tuple[int, int, int] : Size of the region of interest to crop in voxels
            (H, W, D). Default is (128, 128, 64).
        pos_neg : list[float] : Ratio for sampling positive (foreground) vs negative 
            (background) patches. Default is [3.0, 1.0] meaning 3:1 ratio.
        vit_patch_div : tuple[int, int, int] : Divisibility constraint for Vision Transformer
            patch size (H, W, D). Ensures cropped dimensions are divisible by these values.
            Default is (16, 16, 16).
    
    Returns:
        transform : monai.transforms.Transform : A composed transform pipeline that takes
            a dictionary with 'image' and 'label' keys and returns augmented versions.
    """
    return Compose([
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="label", margin=16, k_divisible=vit_patch_div),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=roi_size,  # Ensures minimum size for cropping
            mode="constant",
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=pos_neg[0],
            neg=pos_neg[1],
            num_samples=1,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotated(keys=["image", "label"], range_x=np.pi/16, range_y=np.pi/16, range_z=np.pi/16, prob=0.3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.25),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.25),
        RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.8, 1.2))
    ])

def build_eval_transform(
        target_spacing: tuple[int, int, int] = (1.0, 1.0, 2.0),
        roi_size: tuple[int, int, int] = (128, 128, 64),
        vit_patch_div: tuple[int, int, int] = (16, 16, 16)
) -> Transform:
    """
    Builds a composition of transforms for evaluation/validation on medical image data.
    
    This pipeline performs deterministic preprocessing including spatial resampling,
    normalization, foreground cropping, and deterministic center cropping. Unlike the
    training pipeline, no random augmentation is applied to ensure reproducible evaluation.
    
    Args:
        target_spacing : tuple[int, int, int] : Target voxel spacing in mm for resampling
            (x, y, z). Default is (1.0, 1.0, 2.0).
        roi_size : tuple[int, int, int] : Size of the region of interest to crop in voxels
            (H, W, D). Default is (128, 128, 64).
        vit_patch_div : tuple[int, int, int] : Divisibility constraint for Vision Transformer
            patch size (H, W, D). Ensures cropped dimensions are divisible by these values.
            Default is (16, 16, 16).
    
    Returns:
        transform : monai.transforms.Transform : A composed transform pipeline that takes
            a dictionary with 'image' and 'label' keys and returns preprocessed versions
            with deterministic center cropping.
    """
    return Compose([
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="label", margin=16, k_divisible=vit_patch_div),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=roi_size,  # Ensures minimum size for cropping
            mode="constant",
        ),
        CenterCropByLabelDeterministicd(
            keys=["image", "label"],
            label_key="label",
            roi_size=roi_size,
            k_divisible=vit_patch_div
        )
    ])

def get_train_transform_synthetic(
        spatial_size: tuple[int]
) -> monai.transforms.Transform:
    """
    Builds a composition of transforms for training on synthetic medical image data.
    
    This pipeline is optimized for synthetic data and includes resizing to a fixed
    spatial size, orientation standardization, random flipping augmentations, and
    intensity normalization with augmentation. Designed for data that may not require
    the same preprocessing as real medical scans.
    
    Args:
        spatial_size : tuple[int] : Target spatial dimensions for resizing in voxels
            (H, W, D). All images will be resized to this shape.
    
    Returns:
        transform : monai.transforms.Transform : A composed transform pipeline that takes
            a dictionary with 'image' and 'label' keys and returns augmented synthetic data
            with fixed spatial dimensions.
    """
    return Compose([
        Resized(
            keys=["image", "label"],
            spatial_size=spatial_size,
            mode=["trilinear", "nearest"]
        ),
        Orientationd(
            keys=["image", "label"], 
            axcodes="RAS"
        ),
        RandFlipd(
            keys=["image", "label"],
            prob=0.5,
            spatial_axis=0
        ),
        RandFlipd(
            keys=["image", "label"],
            prob=0.5,
            spatial_axis=1
        ),
        RandFlipd(
            keys=["image", "label"],
            prob=0.5,
            spatial_axis=2
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])

def get_val_transform_synthetic(
        spatial_size: tuple[int]
) -> monai.transforms.Transform:
    """
    Builds a composition of transforms for validation on synthetic medical image data.
    
    This pipeline provides deterministic preprocessing for synthetic validation data,
    including resizing to a fixed spatial size, orientation standardization, and
    intensity normalization. No augmentation is applied for reproducible evaluation.
    
    Args:
        spatial_size : tuple[int] : Target spatial dimensions for resizing in voxels
            (H, W, D). All images will be resized to this shape.
    
    Returns:
        transform : monai.transforms.Transform : A composed transform pipeline that takes
            a dictionary with 'image' and 'label' keys and returns preprocessed synthetic
            data with fixed spatial dimensions and no augmentation.
    """
    return Compose([
        Resized(
            keys=["image", "label"],
            spatial_size=spatial_size,
            mode="trilinear"
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

def build_pretrain_transform(
        target_spacing: tuple[int, int, int] = (1.0, 1.0, 2.0),
        roi_size: tuple[int, int, int] = (128, 128, 64),
) -> Transform:
    """
    Builds a composition of transforms for self-supervised pretraining on unlabeled data.
    
    This pipeline creates multiple augmented views of the same image for contrastive
    learning. It generates three versions: two heavily augmented views ('image', 'image_2')
    and one ground truth view ('gt_image'). The augmented views include coarse dropout
    and shuffling to encourage learning robust representations. Labels are removed as
    they are not used in self-supervised pretraining.
    
    Args:
        target_spacing : tuple[int, int, int] : Target voxel spacing in mm for resampling
            (x, y, z). Default is (1.0, 1.0, 2.0).
        roi_size : tuple[int, int, int] : Size of the region of interest to crop in voxels
            (H, W, D). Default is (128, 128, 64).
    
    Returns:
        transform : monai.transforms.Transform : A composed transform pipeline that takes
            a dictionary with 'image' and 'label' keys and returns a dictionary with
            three image views: 'image', 'image_2', and 'gt_image', each with shape 
            matching roi_size. The 'label' key is removed from the output.
    """
    return Compose([
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="label", allow_smaller=True),
        DeleteItemsd(keys=["label"]),
        SpatialPadd(
            keys=["image"],
            spatial_size=roi_size,  # Ensures minimum size for cropping
            mode="constant",
        ),
        # We now add the transforms to create the contrastive learning images
        RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=roi_size,
            random_size=False,
            num_samples=2
        ),
        CopyItemsd(
            keys=["image"], 
            times=2, 
            names=["gt_image", "image_2"], 
            allow_missing_keys=False
        ),
        RandFlipd(keys=["image", "image_2", "gt_image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "image_2", "gt_image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "image_2", "gt_image"], prob=0.5, spatial_axis=2),
        RandRotated(
            keys=["image", "image_2", "gt_image"], 
            range_x=np.pi/16, 
            range_y=np.pi/16, 
            range_z=np.pi/16, 
            prob=0.3
        ),
        RandScaleIntensityd(keys=["image", "image_2", "gt_image"], factors=0.1, prob=0.25),
        RandShiftIntensityd(keys=["image", "image_2", "gt_image"], offsets=0.1, prob=0.25),
        RandAdjustContrastd(keys=["image", "image_2", "gt_image"], prob=0.2, gamma=(0.8, 1.2)),
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image"], 
                    prob=1.0, 
                    holes=6, 
                    spatial_size=5, 
                    dropout_holes=True, 
                    max_spatial_size=32
                ),
                RandCoarseDropoutd(
                    keys=["image"], 
                    prob=1.0, 
                    holes=6, 
                    spatial_size=20,
                    dropout_holes=False,
                    max_spatial_size=64
                ),
            ]
        ),
        RandCoarseShuffled(
            keys=["image"],
            prob=0.8,
            holes=10,
            spatial_size=8
        ),
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image_2"], 
                    prob=1.0, 
                    holes=6, 
                    spatial_size=5, 
                    dropout_holes=True, 
                    max_spatial_size=32
                ),
                RandCoarseDropoutd(
                    keys=["image_2"], 
                    prob=1.0, 
                    holes=6, 
                    spatial_size=20,
                    dropout_holes=False,
                    max_spatial_size=64
                ),
            ]
        ),
        RandCoarseShuffled(
            keys=["image_2"],
            prob=0.8,
            holes=10,
            spatial_size=8
        )
    ])

def build_preval_transform(
        target_spacing: tuple[int, int, int] = (1.0, 1.0, 2.0),
        roi_size: tuple[int, int, int] = (128, 128, 64),
) -> Transform:
    """
    Builds a composition of transforms for validation during self-supervised pretraining.
    
    This pipeline creates multiple views of the same image for validation of contrastive
    learning models, but without the heavy augmentations used in training. It generates
    three versions ('image', 'image_2', 'gt_image') using only basic preprocessing
    operations to enable consistent evaluation during pretraining. Labels are removed
    as they are not used in self-supervised validation.
    
    Args:
        target_spacing : tuple[int, int, int] : Target voxel spacing in mm for resampling
            (x, y, z). Default is (1.0, 1.0, 2.0).
        roi_size : tuple[int, int, int] : Size of the region of interest to crop in voxels
            (H, W, D). Default is (128, 128, 64).
    
    Returns:
        transform : monai.transforms.Transform : A composed transform pipeline that takes
            a dictionary with 'image' and 'label' keys and returns a dictionary with
            three minimally augmented image views: 'image', 'image_2', and 'gt_image',
            each with shape matching roi_size. The 'label' key is removed from the output.
    """
    return Compose([
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="label", allow_smaller=True),
        DeleteItemsd(keys=["label"]),
        SpatialPadd(
            keys=["image"],
            spatial_size=roi_size,  # Ensures minimum size for cropping
            mode="constant",
        ),
        # We now add the transforms to create the contrastive learning images
        RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=roi_size,
            random_size=False,
            num_samples=2
        ),
        CopyItemsd(
            keys=["image"], 
            times=2, 
            names=["gt_image", "image_2"], 
            allow_missing_keys=False
        )
    ])