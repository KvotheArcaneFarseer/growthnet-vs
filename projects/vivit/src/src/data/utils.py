"""
Medical Image Processing Utilities for Sequential MRI Analysis

This module provides custom MONAI transforms and utility functions for processing
sequential medical imaging data, particularly for brain MRI scans. It includes:

- Custom cropping transforms that center on labeled regions (tumors)
- Sequence-based augmentation transforms for temporal consistency
- Data loading utilities for variable-length sequences
- Tensor manipulation functions for downsampling and post-processing

Key Components:
    - CenterCropByLabeld: Crops around positive labels with dynamic centering
    - CenterCropByLabelDeterministicd: Deterministic label-based cropping with k-divisible alignment
    - TransformSequence: Applies spatial transforms consistently across image sequences
    - Collation and downsampling utilities for batch processing

Dependencies:
    - torch: PyTorch for tensor operations
    - monai: Medical Open Network for AI transforms
    - numpy: Numerical operations
    - scipy: Scientific computing utilities
"""

import torch
import monai
import numpy as np
from copy import deepcopy
from typing import Any
from torch.nn.utils.rnn import pad_sequence
from scipy import ndimage
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose, 
    MapTransform, 
    SpatialCropd,
    Transform
)

class CenterCropByLabeld(MapTransform):
    """
    Center crop around regions where labels are positive (non-zero).
    
    This transform identifies the center of mass of positive label values and crops
    the image and label around that center. If no positive labels exist, it defaults
    to cropping around the geometric center of the volume.
    
    Args:
        keys : list[str] : List of keys in the data dictionary to apply cropping to
        label_key : str : Key in the data dictionary containing the label/segmentation mask
        roi_size : tuple[int, int, int] : Size of the region of interest to crop (D, H, W)
        allow_missing_keys : bool : Whether to allow keys that are not present in the data dictionary (default: False)
    """
    def __init__(
            self, 
            keys: list[str], 
            label_key: str,
            roi_size: tuple[int, int, int],
            allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.label_key = label_key
        self.roi_size = roi_size
    
    def __call__(self, data: dict) -> dict:
        """
        Apply center cropping to the data dictionary.
        
        Args:
            data : dict : Dictionary containing image and label data with keys specified during initialization
        
        Returns:
            cropped_data : dict : Dictionary with cropped image and label tensors
        """
        d = dict(data)
        label = d[self.label_key]

        # Handle tensor to numpy conversion and squeeze channel dimension
        if isinstance(label, torch.Tensor):
            label_array = label[0].cpu().numpy()
        else:
            label_array = label[0] if label.ndim == 4 else label
        
        shape = label.shape  # (C, D, H, W)
        
        if not np.any(label_array > 0):
            center = [shape[1] // 2, shape[2] // 2, shape[3] // 2]
        else:        
            center = ndimage.center_of_mass(label_array > 0)
            center = list(center)  # Already in (D, H, W) order

        # Clamp center
        for i in range(3):
            half_roi = self.roi_size[i] // 2
            center[i] = max(half_roi, min(center[i], shape[i+1] - half_roi))

        # Use spatial crop with computed center
        cropper = SpatialCropd(
            keys=self.keys,
            roi_center=tuple(center),
            roi_size=self.roi_size
        )

        return cropper(d)
    
class CenterCropByLabelDeterministicd(MapTransform):
    """
    Deterministic center crop around positive label regions with optional k-divisible alignment.
    
    This transform computes the center of mass of positive labels and performs a deterministic
    crop around that center. It supports optional k-divisible alignment for the crop start
    position, which is useful for certain neural network architectures that require specific
    spatial dimensions.
    
    Args:
        keys : list[str] : List of keys in the data dictionary to apply cropping to
        label_key : str : Key in the data dictionary containing the label/segmentation mask
        roi_size : tuple[int, int, int] : Size of the region of interest to crop (H, W, D)
        k_divisible : tuple[int, int, int] | None : Divisibility constraint for crop start position (H, W, D), or None for no constraint (default: None)
        allow_missing_keys : bool : Whether to allow keys that are not present in the data dictionary (default: False)
    """
    def __init__(
            self, 
            keys: list[str], 
            label_key: str, 
            roi_size: tuple[int, int, int], 
            k_divisible: tuple[int, int, int] | None = None, 
            allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.label_key = label_key
        self.roi_size = np.asarray(roi_size, dtype=int)
        self.k_divisible = np.asarray(k_divisible, dtype=int) if k_divisible is not None else None

    def __call__(self, data):
        """
        Apply deterministic center cropping with optional k-divisible alignment.
        
        Args:
            data : dict : Dictionary containing image and label data with keys specified during initialization
        
        Returns:
            cropped_data : dict : Dictionary with cropped image and label tensors aligned to k-divisible boundaries if specified
        """
        d = dict(data)
        lbl = d[self.label_key]

        # Move to numpy for processing
        if hasattr(lbl, "cpu") and hasattr(lbl, "numpy"):
            lbl_np = lbl.detach().cpu().numpy()
        else:
            lbl_np = np.asarray(lbl)

        # HWD or CHWD shape
        if lbl_np.ndim == 4:
            lbl_for_com = (lbl_np > 0).any(axis=0).astype(np.uint8)
            H, W, D = lbl_for_com.shape
        elif lbl_np.ndim == 3:
            lbl_for_com = (lbl_np > 0).astype(np.uint8)
            H, W, D = lbl_for_com.shape
        else:
            raise RuntimeError(f"Label must be HWD or CHWD, got {lbl_np.shape}")
        
        # Make sure there is a tumor
        if lbl_for_com.any():
            c_h, c_w, c_d = ndimage.center_of_mass(lbl_for_com)
        else:
            c_h, c_w, c_d = H / 2.0, W / 2.0, D / 2.0

        # Center around whereever the tumor might be
        center = np.array([round(c_h), round(c_w), round(c_d)], dtype=int)

        # K-divisible start alignment (optional)
        half = self.roi_size // 2
        if self.k_divisible is not None:
            start = center - half
            start = (start // self.k_divisible) * self.k_divisible
            center = start + half

        # Clamp using ref image shape (HWD or CHWD)
        ref = d[self.keys[0]]
        if ref.ndim == 4:
            _, Hx, Wx, Dx = ref.shape
        elif ref.ndim == 3:
            Hx, Wx, Dx = ref.shape
        else:
            raise RuntimeError(f"Image must be HWD or CHWD, got {ref.shape}")
        
        lo = half
        hi = np.array([Hx, Wx, Dx]) - half
        center = np.clip(center, lo, hi - 1)

        cropper = SpatialCropd(
            keys=self.keys, 
            roi_center=tuple(center.tolist()),
            roi_size=tuple(self.roi_size.tolist())
        )
                               
        return cropper(d)

class TransformSequence(MapTransform):
    """
    Applies spatial transforms consistently across temporal image sequences.
    
    This transform ensures that augmentations are applied uniformly across all timesteps
    in a sequence by using the same random seed. This maintains temporal consistency in
    sequential medical imaging data.
    
    Args:
        keys : list[str] : List of keys to process, must include 'images', 'labels', and 'dates'
        spatial_transforms : Transform : MONAI transform or composition to apply to each image-label pair
        allow_missing_keys : bool : Whether to allow keys that are not present in the data dictionary (default: False)
    """
    def __init__(
            self, 
            keys: list[str], 
            spatial_transforms: Transform,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.spatial_transforms = spatial_transforms
    
    def __call__(
            self, 
            data: dict
    ) -> dict:
        """
        Apply transforms across all images in the sequence with temporal consistency.
        
        Args:
            data : dict : Dictionary containing 'images' (list of arrays), 'labels' (list of arrays), and 'dates' (list or tensor)
        
        Returns:
            transformed_data : dict : Dictionary with stacked tensors - 'images' shape (T, C, H, W, D), 'labels' shape (T, C, H, W, D), 'dates' shape (T,)
        """
        # Set random seed for when applying transforms
        if hasattr(self.spatial_transforms, 'set_random_state'):
            seed = np.random.randint(0, 2**32)

        # Get the images and labels
        if "images" in self.keys:
            image_key = "images"
        else:
            raise KeyError(f"key `images` not in {self.keys}.")
        
        if "labels" in self.keys:
            label_key = "labels"
        else:
            raise KeyError(f"key `labels` not in {self.keys}.")
        
        if "dates" in self.keys:
            date_key = "dates"
        else:
            raise KeyError(f"key `dates` not in {self.keys}.")
        
        # Copy the data
        data = deepcopy(data)

        # Apply the augmentation sequence
        images, labels = data[image_key], data[label_key]
        augmented_images, augmented_labels = apply_augmentation_sequence(
            images, labels, seed, self.spatial_transforms
        )
        data[image_key] = augmented_images
        data[label_key] = augmented_labels

        # Loop through and execute keys
        for key in self.keys:
            if key in data:
                if key == image_key or key == label_key:
                    data[key] = torch.stack(data[key], dim=0).float()
                elif key == date_key:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].clone().detach().to(torch.float32)
                    else:
                        data[key] = torch.tensor(data[key], dtype=torch.float32)
            else:
                raise KeyError(f"key {key} not in data dictionary.")
        
        return data

def images_only_dataset(
        dataset: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Convert a sequential dataset into individual image-label pairs.
    
    Flattens a dataset of sequences into a dataset of individual timesteps,
    useful for training models that process single images rather than sequences.
    
    Args:
        dataset : list[dict[str, Any]] : List of dictionaries, each containing 'images' (list of arrays) and 'labels' (list of arrays)
    
    Returns:
        image_only_dataset : list[dict[str, Any]] : Flattened list of dictionaries, each containing a single 'image' array and 'label' array
    """
    image_only_dataset = []

    for data in dataset:
        images, labels = data["images"], data["labels"]
        for img, label in zip(images, labels):
            image_only_dataset.append({
                "image": img,
                "label": label
            })
    
    return image_only_dataset

def apply_augmentation_sequence(
        images: list[np.ndarray],
        labels: list[np.ndarray],
        seed: int,
        transforms: Transform
) -> torch.Tensor:
    """
    Apply transforms uniformly across an image-label sequence.
    
    Uses a consistent random seed for all timesteps to ensure the same spatial
    transformations are applied across the entire sequence, maintaining temporal
    consistency.
    
    Args:
        images : list[np.ndarray] : List of image arrays, each with shape (C, H, W, D)
        labels : list[np.ndarray] : List of label arrays, each with shape (C, H, W, D)
        seed : int : Random seed for reproducible transformations
        transforms : Transform : MONAI transform or composition to apply
    
    Returns:
        augmented_images : list[torch.Tensor] : List of transformed image tensors, each with shape (C, H, W, D)
        augmented_labels : list[torch.Tensor] : List of transformed label tensors, each with shape (C, H, W, D)
    """
    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        # Set the same seed
        transforms.set_random_state(seed=seed)

        # Create a data_dict for processing
        data_dict = {"image": img, "label": label}

        # Apply transforms
        augmented = transforms(data_dict)

        # Check if list
        if isinstance(augmented, list):
            augmented = augmented[0]
        
        # Add to list
        augmented_images.append(augmented["image"])
        augmented_labels.append(augmented["label"])

    return augmented_images, augmented_labels

def pad_sequence_collate_fn(
        batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Custom collate function for variable-length sequential MRI data.
    
    Pads sequences to the maximum length in the batch and creates a batch tensor.
    Useful for DataLoader when sequences have different numbers of timesteps.
    
    Args:
        batch : list[dict[str, Any]] : List of dictionaries, each containing 'images' (tensor of shape (T, C, H, W, D)), 'labels' (tensor of shape (T, C, H, W, D)), and 'dates' (tensor of shape (T,))
    
    Returns:
        collated_batch : dict[str, Any] : Dictionary containing:
            - 'images' : torch.Tensor with shape (B, T_max, C, H, W, D) - padded image sequences
            - 'labels' : torch.Tensor with shape (B, T_max, C, H, W, D) - padded label sequences
            - 'dates' : torch.Tensor with shape (B, T_max) - padded dates with -1.0 for padding
            - 'sequence_lengths' : torch.Tensor with shape (B,) - actual length of each sequence
    """
    collated_batch = {}

    keys = batch[0].keys()

    # Pad the sequences
    for key in keys:
        if key in ["images", "labels"]:
            collated_batch[key] = pad_sequence(
                [sample[key] for sample in batch],
                batch_first=True
            )
        elif key == "dates":
            collated_batch[key] = pad_sequence(
                [sample[key] for sample in batch],
                batch_first=True,
                padding_value=-1.0
            )
    
    # Get sequence lengths
    collated_batch["sequence_lengths"] = torch.tensor(
        [len(sample["images"]) for sample in batch],
        dtype=torch.long
    )

    return collated_batch

def down_sample(
        x: torch.Tensor,
        down_size: tuple[int] = (128, 128, 128)
) -> torch.Tensor:
    """
    Downsample 3D volumetric data to reduce memory consumption.
    
    Uses trilinear interpolation to downsample each volume in the batch and sequence
    to the specified size. Useful for processing large medical images with limited GPU memory.
    
    Args:
        x : torch.Tensor : Input tensor with shape (B, T, C, H, W, D) where B is batch size, T is sequence length, C is channels, and H, W, D are spatial dimensions
        down_size : tuple[int] : Target spatial dimensions (H_new, W_new, D_new) (default: (128, 128, 128))
    
    Returns:
        x_downsampled : torch.Tensor : Downsampled tensor with shape (B, T, C, H_new, W_new, D_new)
    """
    B, T = x.shape[0], x.shape[1]

    # Merge batch
    x = x.reshape(-1, *x.shape[2:])

    # Interpolate
    x_downsampled = torch.nn.functional.interpolate(
        x,
        down_size,
        mode='trilinear', 
        align_corners=False
    )

    # Split batch and return
    x_downsampled = x_downsampled.reshape(B, T, *x_downsampled.shape[1:]).to(x.device)

    return x_downsampled

def get_post_transform():
    """
    Get post-processing transform for model predictions.
    
    Creates a composition of transforms to convert model logits to binary predictions.
    Applies sigmoid activation followed by thresholding at 0.5.
    
    Returns:
        transform : Compose : MONAI composition that applies sigmoid activation and discrete thresholding
    """
    return Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5)
    ])