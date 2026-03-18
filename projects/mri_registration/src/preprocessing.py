#!/usr/bin/env python3
"""
Longitudinal MRI Preprocessing and Coregistration Pipeline
===========================================================

This script performs comprehensive longitudinal MRI preprocessing with hierarchical 
coregistration of multiple timepoints for each patient to a common template space.

Workflow Overview:
-----------------
1. **Template Loading**: 
   - Loads MNI T1/T2 templates and their brain-extracted (BET) variants
   - Templates serve as the final common space for all registrations

2. **Patient Processing Loop**:
   For each patient:
   
   a) **Patient Anchor Selection**:
      - Selects the largest T2 image with segmentation as the patient anchor
      - This becomes the primary reference for all timepoints of this patient
   
   b) **Patient Anchor → Template Registration**:
      - Two-step registration: DenseRigid initialization + Similarity refinement
      - Registers patient anchor to MNI template (or previously coregistered image)
      - Saves transformation for propagating all patient images to template space
   
   c) **Timepoint Processing**:
      For each timepoint with multiple images:
      
      i) **Timepoint Anchor Selection**:
         - Selects largest T2 image with segmentation as timepoint anchor
         - Skip if timepoint anchor is same as patient anchor
      
      ii) **Timepoint Anchor → Patient Anchor Registration**:
          - Two-step registration: DenseRigid initialization + Similarity refinement
          - Creates transformation chain to template space
      
      iii) **Other Studies → Timepoint Anchor Registration**:
           - Registers remaining studies (T1, other T2, etc.) to timepoint anchor
           - Single-step DenseRigid registration
           - Complete transformation chain: Study → TP Anchor → Patient Anchor → Template

3. **Output Generation**:
   - Coregistered images and segmentations saved with preserved directory structure
   - Quality control overlay plots (axial, sagittal, coronal views)
   - Mutual information (MI) metrics for registration quality assessment
   - Comprehensive error logging for debugging

Registration Methods:
--------------------
- **DenseRigid**: Rigid transformation (rotation + translation) with dense sampling
- **Similarity**: Similarity transformation (rotation + translation + isotropic scaling)
- **Initialization**: Faster, coarser registration to initialize main registration
- **Main Registration**: Fine-tuned registration with higher iterations

Preprocessing Pipeline:
----------------------
For each raw image, the following steps are applied:
1. **Reorientation**: Reorient to LPI (Left-Posterior-Inferior) standard orientation
2. **Resampling**: Resample to target spacing (default 0.5mm isotropic)
3. **Denoising**: Rician noise model denoising
4. **Bias Correction**: N4 bias field correction
5. **Percentile-based Intensity Standardization**: Map 1st-99th percentiles to [0-1000] range
   - Preserves contrast better than simple truncation/normalization
6. **Brain-Masked Histogram Matching**: Match intensity histogram to template (same modality)
   - Uses brain mask to prevent washing out background/skull
   - Falls back to full-image matching if brain extraction fails
7. **Head Masking**: Remove background outside head region (air/noise)
   - Uses dilated brain extraction masks when available
   - Falls back to Otsu intensity thresholding when masks unavailable

Key Features:
------------
- Hierarchical registration strategy minimizes error accumulation
- Brain extraction masks for robust registration
- Head masking removes background noise for cleaner outputs
- Percentile-based intensity standardization preserves contrast
- Brain-masked histogram matching prevents image washout
- Parallel processing support with configurable worker count
- Resume capability: skips already coregistered images
- Memory management: explicit garbage collection for large datasets
- Multi-orientation QC plots for visual inspection
- Detailed error logging with context for troubleshooting

Configuration:
-------------
Modify the CONFIGURATION section below to:
- Change registration methods (DenseRigid, Similarity, Affine, SyN, etc.)
- Adjust registration parameters (iterations, shrink factors, sampling, etc.)
- Set input/output paths
- Limit processing to subset of patients for testing
- Enable TEST_MODE for dry-run without actual registration
- Enable/disable histogram matching (ENABLE_HISTOGRAM_MATCHING)
- Enable/disable head masking for background removal (ENABLE_HEAD_MASKING)
  Note: This flag is checked at multiple levels (defense-in-depth):
    1. In get_ants_image() before calling get_head_mask()
    2. In get_head_mask() as an early-return guard
    3. In apply_head_mask() as a safety check
- Configure per-level center alignment before registration (CENTER_ALIGNMENT_CONFIG)
  Each registration level (pt_anchor_to_template, tp_anchor_to_pt_anchor,
  study_to_tp_anchor) can be independently enabled/disabled.
  When enabled, uses mask-based rigid translation to align image centers first.
  When disabled, registrations start with initial_transform=None (or Identity).
- Set head mask dilation radius (HEAD_MASK_DILATION_MM)
- Set minimum mask coverage threshold (HEAD_MASK_MIN_COVERAGE)
- Configure parallel processing (PARALLEL_WORKERS, THREADS)

Author: Matthew Nguyen
Date: 2026
Version: 6.8
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU, use CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logging (3 = ERROR only)
import re
import logging
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
import ants
import antspynet
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import traceback
from datetime import datetime, date

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set a reproducible seed and number of threads for ITK/ANTs operations
SEED = 42
THREADS = 8  # Threads per worker process for ITK/ANTs operations

# Set random seeds for reproducibility across all libraries
import random
random.seed(SEED)
np.random.seed(SEED)

# TensorFlow/Keras seed (used by antspynet)
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    # For TensorFlow 2.x, also set this for better reproducibility
    os.environ['PYTHONHASHSEED'] = str(SEED)
except ImportError:
    pass  # TensorFlow not available

# Parallel processing configuration
# Number of patients to process in parallel. Each worker uses THREADS cores.
# Total CPU usage = PARALLEL_WORKERS * THREADS (e.g., 4 workers * 16 threads = 64 cores)
# Set to 1 for sequential processing (original behavior)
PARALLEL_WORKERS = 8

# Set to True to skip image loading and registrations for testing file paths
# Useful for validating workflow logic without time-consuming registrations
TEST_MODE = False

# Set to True to prefer using already coregistered images when available
# This allows resuming processing by using previously coregistered images as anchors
# Set to False to always use raw images, ignoring any existing coregistered images
PREFER_COREGISTERED = False

# Preprocessing version tracking
# Change this version when preprocessing methods are updated (e.g., new intensity normalization)
# This allows detecting which images need re-preprocessing with new methods
PREPROCESSING_VERSION = "6.8"

# Incremental processing modes for adding new timepoints/modalities
# Controls behavior when some studies are already coregistered
INCREMENTAL_MODE = "force_reprocess"  # Options:
                            # - "skip_all": Skip entire patient if ANY study is coregistered (original behavior)
                            # - "process_new": Only process studies without coregistered_image (resume mode)
                            # - "smart": Process new studies + re-preprocess old ones if version changed
                            # - "force_reprocess": Re-preprocess ALL studies, even if coregistered

# When re-preprocessing coregistered images with new methods:
REUSE_TRANSFORMATIONS = True  # If True, reuse existing transformations when re-preprocessing
                              # If False, recompute transformations (slower but more accurate)

#Experiment Suffix for further studies
EXPERIMENT_SUFFIX = f"_{date.today().strftime('%m_%d_%Y')}_V{PREPROCESSING_VERSION}"  # Suffix for output files to distinguish experiments


# Registration method configurations
# 
# Registration Strategy:
# - pt_anchor_to_template: Patient anchor → Template (inter-patient, may use existing coregistered image)
# - tp_anchor_to_pt_anchor: Timepoint anchor → Patient anchor (intra-patient, inter-timepoint)
# - study_to_tp_anchor: Other studies → Timepoint anchor (intra-patient, intra-timepoint)
REGISTRATION_CONFIG = {
    'pt_anchor_to_template': {
        'initializer': 'DenseRigid',  # Fast rigid alignment to get close to solution
        'main': 'Similarity',          # Fine-tuned similarity with scaling allowed
    },
    'tp_anchor_to_pt_anchor': {
        'initializer': 'DenseRigid',  # Fast rigid alignment between timepoints
        'main': 'Similarity',          # Fine-tuned similarity for intra-patient alignment
    },
    'study_to_tp_anchor': {
        'method': 'DenseRigid',        # Single-step rigid for intra-timepoint studies
    }
}

# Registration parameters (iterations, shrink factors, smoothing, sampling)
# Higher iterations = more precise but slower
# Shrink factors: multi-resolution pyramid (4=1/4 resolution, 1=full resolution)
# Smoothing sigmas: Gaussian smoothing at each resolution level (0=no smoothing)
# Sampling: number of voxels sampled per iteration (higher=slower but more robust)
# Random sampling rate: fraction of voxels to use (0.7 = 70% of sampled points)
# Grad step: gradient descent step size (smaller=more stable, larger=faster convergence)
REGISTRATION_PARAMS = {
    'DenseRigid': {
        'aff_iterations': (100000, 5000, 2500, 1000),
        'aff_shrink_factors': (2, 2, 1, 1),
        'aff_smoothing_sigmas': (1.5, 1, 0.5, 0),
        'aff_sampling': 128,
        'aff_random_sampling_rate': 0.8,
        'grad_step': 0.1,
    },
    'Similarity': {
        'aff_iterations': (100000, 5000, 2500, 1000),
        'aff_shrink_factors': (2, 2, 1, 1),
        'aff_smoothing_sigmas': (1.5, 1, 0.5, 0),
        'aff_sampling': 128,
        'aff_random_sampling_rate': 0.8,
        'grad_step': 0.1,
    },
    'Affine': {
        'aff_iterations': (100000, 5000, 2500, 1000),
        'aff_shrink_factors': (2, 2, 1, 1),
        'aff_smoothing_sigmas': (1.5, 1, 0.5, 0),
        'aff_sampling': 128,
        'aff_random_sampling_rate': 0.8,
        'grad_step': 0.1,
    },
}

# Initializer parameters (lighter/faster initialization step)
INITIALIZER_PARAMS = {
    'DenseRigid': {
        'aff_iterations': (100000, 5000, 2500, 1000),
        'aff_shrink_factors': (2, 2, 1, 1),
        'aff_smoothing_sigmas': (1.5, 1, 0.5, 0),
        'aff_sampling': 128,
        'aff_random_sampling_rate': 0.7,
        'grad_step': 0.1
    }
}
 
BASE_DIR = '/standard/gam_ai_group/dataset_curated/uva_vs_notreat'
HD_OUTPUT_DIR = os.path.join(BASE_DIR, 'hd_output')

# Input/Output paths
INPUT_CSV = 'meta_df_entropy.csv'                     # Metadata CSV with image paths
OUTPUT_CSV = f'df_coregistered{EXPERIMENT_SUFFIX}.csv'    # Output CSV with coregistered paths
TEMPLATE_FOLDER = 'Template'                               # Folder containing MNI templates
COREGISTERED_OUTPUT_DIR = os.path.join(BASE_DIR, f'df_coregistered{EXPERIMENT_SUFFIX}')  # Root output directory
OVERLAY_OUTPUT_DIR = os.path.join(COREGISTERED_OUTPUT_DIR, 'overlay_plots')  # QC plots
# Removed: ERROR_LOG_CSV - errors now go to OUTPUT_CSV

# Processing limits (set to None for no limit)
MAX_PATIENTS = None  # Set to integer (e.g., 5) for testing on subset

# Study type filtering
# Set to None to allow all study types, or provide a list of allowed types
# Examples: ['T2'], ['T1', 'T2'], ['T2_FLAIR', 'T2'], etc.
# Matching is case-insensitive and uses substring matching (e.g., 'T2' matches 'T2_FLAIR')
ALLOWED_STUDY_TYPES = ['t2_thin', 't1+_thin','t1+_thick_ax','t1+_thick_cor']  # e.g., ['T2'] to only process T2 studies

# Resampling configuration
TARGET_SPACING = (0.5, 0.5, 0.5)  # Target voxel spacing in mm (isotropic 0.5mm)

# Histogram matching configuration
# When enabled, matches the intensity histogram of the moving image to the fixed image
# before registration. This improves same-modality registration by normalizing intensities.
# Only applied when both images have the same study type (e.g., T2 to T2, T1 to T1).
#
# NEW: Brain-masked histogram matching prevents image washout by matching only within brain tissue.
# Modality (T1 or T2) is automatically detected from 'Study Type' field (e.g., 'T1_thin' -> T1, 'T2_FLAIR' -> T2).
# Brain extraction uses ANTsPyNet with modality-specific models for both T1 and T2 images.
ENABLE_HISTOGRAM_MATCHING = False  # Set to True to enable brain-masked histogram matching
HISTOGRAM_MATCHING_BINS = 1024     # Number of histogram bins for matching
HISTOGRAM_MATCHING_POINTS = 256    # Number of quantile points for matching

# Head mask / background removal configuration
# When enabled, removes background (air) outside the head region from images.
# Uses dilated brain extraction masks when available, falls back to Otsu thresholding.
#
# What ENABLE_HEAD_MASKING controls:
# - Background removal during preprocessing (zeroes out air/noise outside head)
# - Applied via get_head_mask() -> apply_head_mask() pipeline
# - Uses brain extraction + dilation or Otsu thresholding as fallback
#
# What it does NOT control (always active regardless of this flag):
# - Brain extraction masks used for registration guidance (get_bet_image)
# - Histogram matching masks (uses brain masks, not head masks)
# - Registration metric computation
ENABLE_HEAD_MASKING = False           # Master switch for head region extraction/background removal
HEAD_MASK_DILATION_MM = 25.0         # Dilation radius in mm (enough to include skull from brain)
HEAD_MASK_MIN_COVERAGE = 5.0         # Minimum % coverage to consider mask valid

# Center alignment configuration
# Controls where align_image_centers is applied in the registration hierarchy.
# Each level can be independently enabled/disabled:
#   - pt_anchor_to_template:  Patient anchor → Template registration
#   - tp_anchor_to_pt_anchor: Timepoint anchor → Patient anchor registration
#   - study_to_tp_anchor:     Study → Timepoint anchor registration
# When enabled, aligns image centers before registration using mask-based rigid translation.
# This helps prevent "no valid points" errors when images start far apart.
# When disabled for a level, that registration uses no initial transform (initial_transform=None).
CENTER_ALIGNMENT_CONFIG = {
    'pt_anchor_to_template':  True,   # Recommended: images may be far from template space
    'tp_anchor_to_pt_anchor': False,   # Recommended: different timepoints may differ in position
    'study_to_tp_anchor':     False,  # Usually unnecessary: studies share space with their TP anchor
}

# Anchor selection metric configuration
# Controls which column from meta_df_entropy is used to rank candidates when selecting anchors.
# 'column': the DataFrame column to sort by (e.g., 'file_size_mb', 'entropy', 'voxel_volume')
# 'ascending': sort order — False selects the highest value, True selects the lowest value
# Examples:
#   Largest file size (default):  {'column': 'file_size_mb', 'ascending': False}
#   Lowest entropy:               {'column': 'entropy',      'ascending': True}
#   Largest voxel volume:         {'column': 'voxel_volume', 'ascending': False}
ANCHOR_SELECTION_METRIC = {
    'column': 'file_size_mb',
    'ascending': False,
}

# Template Z-axis cropping configuration
# When enabled, crops templates along the z-axis (inferior-superior) after loading.
# Useful for removing inferior slices (e.g., neck/lower brainstem) that can hurt registration.
# Set TEMPLATE_Z_CROP to None to disable, or a tuple (start, end) for the slice range.
TEMPLATE_Z_CROP = (0, 140)           # Crop z-axis to slices 0:140; set to None to disable

# ============================================================================
# SETUP
# ============================================================================

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(THREADS)
log.info(f"Set ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS to {THREADS}.")


# ============================================================================
# BRAIN EXTRACTION CACHING
# ============================================================================

def _cached_brain_extraction(image, modality, cache_key=None, cache_path=None):
    """
    Run antspynet.brain_extraction with disk caching.

    Args:
        image: ANTs image object
        modality: Modality string for brain_extraction (e.g. 't1', 't2')
        cache_key: A unique string used to build the cache filename.
                   If None, a hash of image metadata is used.
        cache_path: Full path for the cache file. If provided, overrides
                    cache_key. Useful when callers already know the exact
                    path they want (e.g. filename-based caching).

    Returns:
        ANTs image: Brain extraction mask
    """
    # Build cache path
    if cache_path is None:
        if cache_key is None:
            # Derive a key from image geometry so repeated runs on the same data hit cache
            meta = f"{image.origin}_{image.spacing}_{image.direction.tobytes()}_{image.shape}"
            cache_key = hashlib.md5(meta.encode()).hexdigest()

        cache_dir = HD_OUTPUT_DIR
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_key}_brain_mask.nii.gz")
    else:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        except Exception as e:
            log.warning(f"Failed to create cache directory for {cache_path}: {e}")

    if os.path.isfile(cache_path):
        log.info(f"Brain extraction cache hit: {cache_path}")
        try:
            return ants.image_read(cache_path)
        except Exception as e:
            log.warning(f"Failed to read cached brain mask {cache_path}: {e}")

    log.info(f"Brain extraction cache miss, computing (modality={modality}, cache={cache_path})")
    mask = antspynet.brain_extraction(image, modality=modality)

    try:
        ants.image_write(mask, cache_path)
        log.info(f"Cached brain extraction mask to {cache_path}")
    except Exception as e:
        log.warning(f"Failed to write brain mask cache {cache_path}: {e}")

    return mask


# ============================================================================
# ERROR LOGGING
# ============================================================================

def log_error_to_df(meta_df, row_index, error_details):
    """
    Log error information directly to the DataFrame.
    
    Args:
        meta_df: The main DataFrame being processed
        row_index: Index of the row where error occurred (can be None for patient-level errors)
        error_details: Free-text description of the error with context
    
    Returns:
        None (modifies meta_df in place)
    """
    timestamp = datetime.now().isoformat()
    
    if row_index is not None and row_index in meta_df.index:
        meta_df.loc[row_index, 'error_details'] = error_details
        meta_df.loc[row_index, 'error_timestamp'] = timestamp
        log.error(f"Error recorded for row index {row_index}")
    else:
        log.error(f"Error occurred (no specific row): {error_details[:100]}...")


# ============================================================================
# TEMPLATE LOADING
# ============================================================================

def load_templates(template_folder):
    """
    Load and preprocess MNI template images.
    
    Args:
        template_folder: Path to folder containing template NIfTI files
        
    Returns:
        dict: Dictionary of processed template images and BET variants
    """
    if not os.path.isdir(template_folder):
        raise RuntimeError(f"Template folder not found: {template_folder}")

    template_files = [f for f in os.listdir(template_folder) if f.endswith('.nii.gz')]
    
    templates = {}
    template_bets = {}

    for file in template_files:
        file_path = os.path.join(template_folder, file)
        lower_file = file.lower()

        # Read image
        img = ants.image_read(file_path)

        # Crop along z-axis if configured
        if TEMPLATE_Z_CROP is not None:
            z_start, z_end = TEMPLATE_Z_CROP
            z_end = min(z_end, img.shape[2])  # Clamp to actual z dimension
            img = ants.crop_indices(img, (0, 0, z_start), (img.shape[0], img.shape[1], z_end))
            log.info(f"Cropped template {file} z-axis to [{z_start}:{z_end}], new shape: {img.shape}")

        # Store based on filename content
        if "bet.nii.gz" in lower_file:
            if "t1" in lower_file:
                template_bets['t1_bet'] = img
            elif "t2" in lower_file:
                template_bets['t2_bet'] = img
        else:
            if "t1" in lower_file:
                templates['t1'] = img
                template_bets['t1'] = img
            elif "t2" in lower_file:
                templates['t2'] = img
                template_bets['t2'] = img

    # Ensure template_bets has both plain and bet variants if available
    for mod in ('t1', 't2'):
        if mod in templates:
            if mod not in template_bets:
                template_bets[mod] = templates[mod]
            suff = f"{mod}_bet"
            if suff not in template_bets:
                template_bets[suff] = templates[mod]

    log.info("Loaded templates: %s", list(template_bets.keys()))
    return template_bets


# ============================================================================
# PREPROCESSING HELPERS
# ============================================================================

def fix_zero_spacing(image, img_path="unknown"):
    """
    Fix invalid spacing values in an ANTs image.
    """
    if image is None:
        return None
    
    spacing = list(image.spacing)
    needs_fix = False
    
    MIN_VALID_SPACING = 1e-6
    MAX_VALID_SPACING = 100.0
    
    def is_invalid_spacing(s):
        """Check if a spacing value is invalid."""
        if s is None:
            return True, "None"
        if np.isnan(s):
            return True, "NaN"
        if np.isinf(s):
            return True, "infinity"
        if s < 0:
            return True, "negative"
        if s < MIN_VALID_SPACING:
            return True, "zero/near-zero"
        if s > MAX_VALID_SPACING:
            return True, "too large"
        return False, None
    
    # First pass: identify all invalid spacings and collect valid ones
    invalid_dims = []
    valid_spacings = []
    
    for i, s in enumerate(spacing):
        is_bad, reason = is_invalid_spacing(s)
        if is_bad:
            invalid_dims.append((i, s, reason))
            needs_fix = True
        else:
            valid_spacings.append(s)
    
    # Second pass: fix invalid spacings
    if needs_fix:
        log.critical(f"Found invalid spacing in {os.path.basename(img_path)}: {image.spacing}")
        for i, original_value, reason in invalid_dims:
            log.info(f"  [FIX] Dimension {i}: {original_value} ({reason})")

            if valid_spacings:
                replacement = float(np.median(valid_spacings))
            else:
                # All spacings are invalid - use a reasonable default (1.0mm)
                replacement = 1.0
                log.warning(f"  No valid spacings found, using default: {replacement}")

            spacing[i] = replacement

        log.info(f"  [FIXED] New spacing: {tuple(spacing)}")
        image.set_spacing(tuple(spacing))
    
    return image


def validate_image_for_registration(image, image_name="image"):
    """
    Validate that an image is suitable for registration.
    Raises ValueError if image has invalid properties.
    
    Args:
        image: ANTs image object
        image_name: Name for error messages
    
    Returns:
        ANTs image: Validated (and potentially fixed) image
    
    Raises:
        ValueError: If image is None or has unfixable invalid properties
    """
    if image is None:
        raise ValueError(f"{image_name}: Image is None")
    
    # Check spacing for invalid values
    spacing = image.spacing
    for i, s in enumerate(spacing):
        if np.isnan(s) or np.isinf(s) or s <= 0:
            raise ValueError(f"{image_name}: Invalid spacing in dimension {i}: {s}")
    
    # Check for NaN/Inf in image data
    img_array = image.numpy()
    has_nan = np.any(np.isnan(img_array))
    has_inf = np.any(np.isinf(img_array))
    
    if has_nan or has_inf:
        if has_nan:
            log.warning(f"{image_name}: Contains NaN values, replacing with 0")
        if has_inf:
            log.warning(f"{image_name}: Contains Inf values, clipping")
        
        img_array = np.nan_to_num(
            img_array, 
            nan=0.0, 
            posinf=np.finfo(np.float32).max, 
            neginf=np.finfo(np.float32).min
        )
        image = ants.from_numpy(
            img_array, 
            origin=image.origin, 
            spacing=image.spacing, 
            direction=image.direction
        )
    
    return image


def validate_warped_image(image, image_name="warped_image"):
    """
    Validate that a warped image (output of apply_transforms) is valid.
    This catches issues like zero-valued spacing that can occur after transformations.

    Args:
        image: ANTs image object (result of apply_transforms)
        image_name: Name for error messages

    Returns:
        bool: True if valid, False if invalid
    """
    if image is None:
        log.warning(f"{image_name}: Image is None after apply_transforms")
        return False

    try:
        # Check spacing for invalid values (zero, NaN, Inf)
        spacing = image.spacing
        for i, s in enumerate(spacing):
            if np.isnan(s) or np.isinf(s) or s <= 0:
                log.error(f"{image_name}: Invalid spacing in dimension {i}: {s}. Full spacing: {spacing}")
                return False

        # Check image dimensions
        shape = image.shape
        if any(dim <= 0 for dim in shape):
            log.error(f"{image_name}: Invalid shape: {shape}")
            return False

        return True
    except Exception as e:
        log.error(f"{image_name}: Validation failed with exception: {e}")
        return False


def align_image_centers(fixed, moving, fixed_mask=None, moving_mask=None,
                        fixed_study_type=None, moving_study_type=None):
    """
    Create an initial rigid transform by registering the BET masks of both images.

    Args:
        fixed: Fixed ANTs image
        moving: Moving ANTs image
        fixed_mask: Brain extraction mask for fixed image
        moving_mask: Brain extraction mask for moving image
        fixed_study_type: Study type of fixed image (for metric selection)
        moving_study_type: Study type of moving image (for metric selection)

    Returns:
        str or None: Path to the rigid transform file, or None on failure.
    """
    if fixed is None or moving is None:
        return None

    try:
        # Prepare fixed mask
        if fixed_mask is not None:
            fixed_bet = ants.resample_image_to_target(fixed_mask, fixed, interp_type='nearestNeighbor')
        else:
            log.info("align_image_centers: No fixed mask provided, computing brain extraction")
            fixed_bet = _cached_brain_extraction(fixed, modality='t2')

        # Prepare moving mask
        if moving_mask is not None:
            moving_bet = ants.resample_image_to_target(moving_mask, moving, interp_type='nearestNeighbor')
        else:
            log.info("align_image_centers: No moving mask provided, computing brain extraction")
            moving_bet = _cached_brain_extraction(moving, modality='t2')

        aff_metric = get_aff_metric(fixed_study_type, moving_study_type)
        log.info(f"Performing rigid registration on BET masks for initial alignment (aff_metric='{aff_metric}')")

        reg_result = ants.registration(
            fixed=fixed_bet,
            moving=moving_bet,
            type_of_transform='Rigid',
            aff_metric=aff_metric,
            verbose=False,
            random_seed=SEED
        )

        return reg_result['fwdtransforms'][0]

    except Exception as e:
        log.warning(f"Failed to align image centers via rigid registration: {e}")
        return None


def standardize_intensity_percentiles(image, lower_percentile=1, upper_percentile=99,
                                       target_min=0, target_max=1):
    """
    Standardize intensity by mapping specific percentiles to target values.
    Preserves contrast better than full histogram matching.

    Args:
        image: ANTs image object
        lower_percentile: Lower percentile for clipping (default: 1)
        upper_percentile: Upper percentile for clipping (default: 99)
        target_min: Target minimum value after rescaling (default: 0)
        target_max: Target maximum value after rescaling (default: 1000)

    Returns:
        ANTs image: Intensity-standardized image
    """
    img_array = image.numpy()

    # Get percentile values
    p_low = np.percentile(img_array, lower_percentile)
    p_high = np.percentile(img_array, upper_percentile)

    # Clip and rescale
    img_array = np.clip(img_array, p_low, p_high)
    if p_high > p_low:  # Avoid division by zero
        img_array = (img_array - p_low) / (p_high - p_low) * (target_max - target_min) + target_min

    # Create new ANTs image with standardized intensities
    return ants.from_numpy(img_array, origin=image.origin, spacing=image.spacing, direction=image.direction)


def histogram_match_masked(source, reference, mask=None, bins=1024, points=256):
    """
    Apply histogram matching only within masked region (e.g., brain only).
    Preserves background and overall contrast better than full-image matching.

    Args:
        source: ANTs image to be matched
        reference: ANTs image to match to
        mask: Optional binary mask (ANTs image). If None, matches full image.
        bins: Number of histogram bins (default: 1024)
        points: Number of quantile points for matching (default: 256)

    Returns:
        ANTs image: Histogram-matched source image
    """
    if mask is None:
        # Fallback to standard histogram matching if no mask provided
        return ants.histogram_match_image(source, reference, bins, points)

    source_arr = source.numpy()
    reference_arr = reference.numpy()

    # Resample mask to source image space if needed
    if mask.shape != source.shape:
        mask = ants.resample_image_to_target(mask, source, interp_type='nearestNeighbor')

    mask_arr = mask.numpy().astype(bool)

    # Check if mask has any voxels
    if not np.any(mask_arr):
        log.info('[HISTOGRAM_MASKED] Warning: Empty mask, using full-image histogram matching')
        return ants.histogram_match_image(source, reference, bins, points)

    # Extract masked regions
    source_masked = source_arr[mask_arr]
    reference_masked = reference_arr[mask_arr]

    # Compute quantiles for matching
    quantile_levels = np.linspace(0, 100, points)
    s_quantiles = np.percentile(source_masked, quantile_levels)
    r_quantiles = np.percentile(reference_masked, quantile_levels)

    # Map source intensities to reference distribution
    # Use interpolation for smooth mapping
    matched_arr = np.interp(source_arr.flatten(), s_quantiles, r_quantiles)
    matched_arr = matched_arr.reshape(source_arr.shape)

    # Create new ANTs image
    return ants.from_numpy(matched_arr, origin=source.origin, spacing=source.spacing, direction=source.direction)


def get_ants_image(row, prefer_coregistered=False, template_bets=None, force_preprocess=False):
    """
    Load an ants image from the given file_path and apply the pre-processing steps:
      1. Reorient to LPI.
      2. Resample to target spacing (0.5mm isotropic).
      3. Denoise using Rician noise model.
      4. Bias field correction (N4).
      5. Percentile-based intensity standardization.
      6. Brain-masked histogram matching to template of same modality (if template_bets provided).

    Args:
        row: DataFrame row containing image metadata
        prefer_coregistered: If True, prefer coregistered_image over raw image_path
        template_bets: Dictionary of template images (keys: 't1', 't2', 't1_bet', 't2_bet')
                       If provided, histogram matching will be applied to same-modality template
        force_preprocess: If True, apply preprocessing even to coregistered images
                         (used when re-preprocessing with updated methods)

    Returns:
        tuple: (image, segmentation) - ANTs images or (None, None) if TEST_MODE

    Raises:
        ValueError: If no valid image path found
    """
    # Determine which image path to use
    img_path = None
    is_coregistered = False

    # Check if this image needs re-preprocessing based on version
    preprocessing_version = row.get('preprocessing_version', None)
    needs_reprocess = (preprocessing_version != PREPROCESSING_VERSION) if preprocessing_version else False

    if prefer_coregistered and row.get('coregistered_image') is not None and pd.notna(row.get('coregistered_image')):
        img_path = row.get('coregistered_image')
        is_coregistered = True
    elif row.get('image path') is not None and pd.notna(row.get('image path')):
        img_path = row.get('image path')
    
    if not img_path:
        raise ValueError("Row must contain either 'coregistered_image' or 'image path' entry.")
    
    img_path = str(img_path)
    img_filename = os.path.basename(img_path)
    img_type = "coregistered" if is_coregistered else "raw"
    log.info(f"[INPUT] Processing {img_type} image: {img_filename}")
    log.debug(f"Full path: {img_path}")
    
    if TEST_MODE:
        return None, None
    
    # Validate file exists
    if not Path(img_path).is_file():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # Load image
    try:
        image = ants.image_read(img_path)
        # Fix zero spacing issues that cause ITK errors
        image = fix_zero_spacing(image, img_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {img_path}: {str(e)}")

    # Determine if we should apply preprocessing
    # Apply preprocessing if:
    # 1. Image is not coregistered (raw image)
    # 2. force_preprocess flag is True (explicit re-preprocessing request)
    # 3. Image is coregistered but outdated (needs_reprocess) and INCREMENTAL_MODE allows it
    should_preprocess = (
        not is_coregistered or
        force_preprocess or
        (needs_reprocess and INCREMENTAL_MODE in ["smart", "force_reprocess"])
    )

    if should_preprocess:
        # If re-preprocessing a coregistered image, load from raw path instead
        if is_coregistered and (force_preprocess or needs_reprocess):
            raw_path = row.get('image path')
            if raw_path and pd.notna(raw_path):
                log.info(f"[REPROCESS] Loading raw image for re-preprocessing (version: {preprocessing_version} -> {PREPROCESSING_VERSION})")
                try:
                    image = ants.image_read(str(raw_path))
                    image = fix_zero_spacing(image, str(raw_path))
                except Exception as e:
                    log.info(f"[REPROCESS] Warning: Could not load raw image for re-preprocessing, using coregistered version")
                    # Continue with already-loaded coregistered image
        image = image.reorient_image2(orientation='LPI')
        
        # Resample to target spacing (0.5mm isotropic)
        log.info(f"Resampling from {image.spacing} to {TARGET_SPACING}")
        image = ants.resample_image(image, TARGET_SPACING, use_voxels=False, interp_type=4)  # interp_type=4 is bspline
        image = ants.denoise_image(image, noise_model='Rician')
        image = ants.abp_n4(image)

        # Determine modality early for all preprocessing steps
        study_type = row.get('Study Type', '').strip().lower()
        if study_type.startswith('t1'):
            modality = 't1'
        elif study_type.startswith('t2'):
            modality = 't2'
        else:
            modality = 't2'  # Default fallback for unknown types
            log.info(f"[MODALITY] Unknown study type '{study_type}', defaulting to T2")

        log.info(f"[MODALITY] Detected modality: {modality.upper()} (from study type: '{row.get('Study Type', '')}')")

        # Step 1: Percentile-based intensity standardization (replaces TruncateIntensity + Normalize)
        log.info(f"[INTENSITY] Applying percentile-based standardization (1-99th percentile → 0-1)")
        image = standardize_intensity_percentiles(image, lower_percentile=1, upper_percentile=99,
                                                   target_min=0, target_max=1)

        # Step 2: Brain-masked histogram matching to template of same modality
        if template_bets is not None and ENABLE_HISTOGRAM_MATCHING:
            template_key = modality  # Use detected modality (t1 or t2)

            if template_key in template_bets:
                try:
                    template_image = template_bets[template_key]

                    # Get brain mask for masked histogram matching
                    try:
                        log.info(f"[HISTOGRAM] Extracting {modality.upper()} brain mask for masked histogram matching")
                        brain_mask = get_bet_image(image, row, modality=modality)
                        log.info(f"[HISTOGRAM] Brain-masked histogram matching to {template_key.upper()} template")
                        image = histogram_match_masked(
                            source=image,
                            reference=template_image,
                            mask=brain_mask,
                            bins=HISTOGRAM_MATCHING_BINS,
                            points=HISTOGRAM_MATCHING_POINTS
                        )
                        del brain_mask
                        gc.collect()
                        log.info(f"[HISTOGRAM] Successfully matched to {template_key.upper()} template (brain-masked)")
                    except Exception as mask_error:
                        log.info(f"[HISTOGRAM] Failed to get brain mask ({mask_error}), using full-image matching")
                        image = ants.histogram_match_image(
                            source_image=image,
                            reference_image=template_image,
                            number_of_histogram_bins=HISTOGRAM_MATCHING_BINS,
                            number_of_match_points=HISTOGRAM_MATCHING_POINTS
                        )
                        log.info(f"[HISTOGRAM] Successfully matched to {template_key.upper()} template (full-image)")
                except Exception as e:
                    log.info(f"[HISTOGRAM] Failed to match histogram to template: {e}. Using original image.")
            else:
                log.info(f"[HISTOGRAM] Template '{template_key}' not found in template_bets")

        # Apply head mask to remove background (after normalization and histogram matching)
        if ENABLE_HEAD_MASKING:
            # Use the same modality detected earlier for consistency
            log.info(f"[HEAD_MASK] Using {modality.upper()} modality for head mask extraction")
            head_mask = get_head_mask(image, row=row, modality=modality)
            if head_mask is not None:
                image = apply_head_mask(image, head_mask)
                del head_mask
                gc.collect()
                log.info(f"[HEAD_MASK] Applied head mask, background removed")
            else:
                log.info(f"[HEAD_MASK] Warning: get_head_mask returned None, background not removed")
        else:
            log.info(f"[HEAD_MASK] Skipped - ENABLE_HEAD_MASKING is False (background preserved)")

    # Segmentation: load if path exists (prefer coregistered)
    segmentation = None
    seg_path = None
    is_seg_coregistered = False
    
    if prefer_coregistered and row.get('coregistered_segmentation'):
        seg_path = row.get('coregistered_segmentation')
        is_seg_coregistered = True
    elif row.get('segmentation path'):
        seg_path = row.get('segmentation path')
        is_seg_coregistered = False
    
    if seg_path:
        seg_path = Path(str(seg_path)).expanduser()
        if seg_path.is_file():
            try:
                segmentation = ants.image_read(str(seg_path))
                # Fix zero spacing issues
                segmentation = fix_zero_spacing(segmentation, str(seg_path))
                # Only reorient and resample if segmentation is not coregistered
                if not is_seg_coregistered:
                    segmentation = segmentation.reorient_image2('LPI')
                    # Resample segmentation using nearest neighbor interpolation to preserve labels
                    log.info(f"Resampling segmentation from {segmentation.spacing} to {TARGET_SPACING}")
                    segmentation = ants.resample_image(segmentation, TARGET_SPACING, use_voxels=False, interp_type=1)  # interp_type=1 is nearestNeighbor
            except Exception as e:
                log.warning(f"Failed to load segmentation from {seg_path}: {str(e)}")
        else:
            log.warning(f"Segmentation file not found: {seg_path}")

    return image, segmentation


def save_coregistered(row, transformed_image, transformed_segmentation, output_dir=COREGISTERED_OUTPUT_DIR):
    """
    Save transformed_image and transformed_segmentation under output directory
    preserving the relative path with respect to `cwd` when available.
    Returns a dict with the output paths.
    In TEST_MODE, returns paths without writing to disk.
    """
    if TEST_MODE:
        log.info("[TEST_MODE] Skipping file write for coregistered images")
        return {
            'image_coregistered_path': 'TEST_MODE_coregistered_image.nii.gz',
            'seg_coregistered_path': 'TEST_MODE_coregistered_segmentation.nii.gz',
        }
    
    img_path = row.get('image path')
    if not img_path:
        raise ValueError("Row must contain an 'image path' entry.")

    # Helper to split out .nii.gz specially
    def split_ext(p):
        if p.lower().endswith('.nii.gz'):
            return p[:-7], '.nii.gz'
        else:
            base, ext = os.path.splitext(p)
            return base, ext

    # Determine a relative path for the image (strip cwd if available)
    cwd = os.getcwd()
    try:
        rel_img = os.path.relpath(img_path, start=cwd)
    except Exception:
        rel_img = os.path.basename(img_path)

    # Preserve subfolders under coregistered_images
    sbase, sext = split_ext(rel_img)
    out_img_rel = f"{sbase}_coregistered{sext}"
    out_img_path = os.path.normpath(os.path.join(output_dir, out_img_rel))

    parent = os.path.dirname(out_img_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Note: Head masking already applied during preprocessing using precomputed masks on raw images
    # Masking is preserved through transformation, no need to re-mask
    ants.image_write(transformed_image, out_img_path)

    out_seg_path = None
    if transformed_segmentation is not None and row.get('segmentation path'):
        seg_path = row['segmentation path']
        try:
            rel_seg = os.path.relpath(seg_path, start=cwd)
        except Exception:
            rel_seg = os.path.basename(seg_path)

        ssbase, ssext = split_ext(rel_seg)
        out_seg_rel = f"{ssbase}_coregistered{ssext}"
        out_seg_path = os.path.normpath(os.path.join(output_dir, out_seg_rel))

        seg_parent = os.path.dirname(out_seg_path)
        if seg_parent:
            os.makedirs(seg_parent, exist_ok=True)

        ants.image_write(transformed_segmentation, out_seg_path)

    return {
        'image_coregistered_path': out_img_path,
        'seg_coregistered_path': out_seg_path,
    }


# ============================================================================
# REGISTRATION HELPERS
# ============================================================================

def get_aff_metric(fixed_study_type, moving_study_type):
    """
    Determine the appropriate affine metric based on study types.

    Args:
        fixed_study_type: Study type of the fixed image (e.g., 'T1', 'T2', 'T2_FLAIR')
        moving_study_type: Study type of the moving image (e.g., 'T1', 'T2', 'T2_FLAIR')

    Returns:
        str: 'GC' (Gradient Correlation) if modalities match, 'mattes' (Mutual Information) otherwise

    Note:
        - GC is better for same-modality registration (same contrast)
        - Mattes MI is better for cross-modality registration (different contrasts)
        - Modality is determined by T1/T2 prefix only
    """
    # Normalize study types for comparison (case-insensitive, strip whitespace)
    fixed_normalized = (fixed_study_type or '').strip().lower()
    moving_normalized = (moving_study_type or '').strip().lower()

    # If either is empty/unknown, default to mattes (safer choice)
    if not fixed_normalized or not moving_normalized:
        log.info(f"[AFF_METRIC] Unknown study type(s): fixed='{fixed_study_type}', moving='{moving_study_type}' -> using 'mattes'")
        return 'mattes'

    # Extract modality prefix (t1 or t2)
    fixed_modality = fixed_normalized[:2] if fixed_normalized.startswith(('t1', 't2')) else fixed_normalized
    moving_modality = moving_normalized[:2] if moving_normalized.startswith(('t1', 't2')) else moving_normalized

    # Check if modalities are the same
    if fixed_modality == moving_modality:
        log.info(f"[AFF_METRIC] Same modality: '{fixed_study_type}' ({fixed_modality}) == '{moving_study_type}' ({moving_modality}) -> using 'GC'")
        return 'GC'
    else:
        log.info(f"[AFF_METRIC] Different modalities: '{fixed_study_type}' ({fixed_modality}) vs '{moving_study_type}' ({moving_modality}) -> using 'mattes'")
        return 'mattes'


def apply_histogram_matching(fixed, moving, fixed_study_type=None, moving_study_type=None,
                            fixed_mask=None, moving_mask=None):
    """
    Apply brain-masked histogram matching to normalize the moving image's intensities to match the fixed image.
    Only applied when histogram matching is enabled and modalities are the same.

    Args:
        fixed: ANTs image (reference for histogram)
        moving: ANTs image (to be matched)
        fixed_study_type: Study type of fixed image
        moving_study_type: Study type of moving image
        fixed_mask: Optional brain mask for fixed image (for masked histogram matching)
        moving_mask: Optional brain mask for moving image (for masked histogram matching)

    Returns:
        ANTs image: Histogram-matched moving image, or original if matching not applied

    Note:
        Modality is determined by T1/T2 prefix only
        Uses brain-masked histogram matching when masks are available
    """
    if not ENABLE_HISTOGRAM_MATCHING:
        return moving

    # Only apply histogram matching for same-modality registration
    fixed_normalized = (fixed_study_type or '').strip().lower()
    moving_normalized = (moving_study_type or '').strip().lower()

    # Skip if either study type is unknown
    if not fixed_normalized or not moving_normalized:
        log.info(f"[HISTOGRAM] Skipping - unknown modality")
        return moving

    # Extract modality prefix (t1 or t2)
    fixed_modality = fixed_normalized[:2] if fixed_normalized.startswith(('t1', 't2')) else fixed_normalized
    moving_modality = moving_normalized[:2] if moving_normalized.startswith(('t1', 't2')) else moving_normalized

    # Skip if modalities are different (cross-modality)
    if fixed_modality != moving_modality:
        log.info(f"[HISTOGRAM] Skipping - different modalities: '{fixed_study_type}' ({fixed_modality}) vs '{moving_study_type}' ({moving_modality})")
        return moving

    try:
        # Use brain-masked histogram matching if both masks are provided
        if fixed_mask is not None and moving_mask is not None:
            log.info(f"[HISTOGRAM] Brain-masked histogram matching (modality: {fixed_modality})")

            # Create combined mask (intersection of fixed and moving brain regions)
            # Resample moving mask to fixed space for consistent masking
            moving_mask_resampled = ants.resample_image_to_target(moving_mask, moving,
                                                                   interp_type='nearestNeighbor')

            matched_moving = histogram_match_masked(
                source=moving,
                reference=fixed,
                mask=moving_mask_resampled,
                bins=HISTOGRAM_MATCHING_BINS,
                points=HISTOGRAM_MATCHING_POINTS
            )
            log.info(f"[HISTOGRAM] Successfully matched histogram (brain-masked)")
        else:
            # Fallback to standard full-image histogram matching
            log.info(f"[HISTOGRAM] Full-image histogram matching (modality: {fixed_modality})")
            matched_moving = ants.histogram_match_image(
                source_image=moving,
                reference_image=fixed,
                number_of_histogram_bins=HISTOGRAM_MATCHING_BINS,
                number_of_match_points=HISTOGRAM_MATCHING_POINTS
            )
            log.info(f"[HISTOGRAM] Successfully matched histogram (full-image)")

        return matched_moving

    except Exception as e:
        log.info(f"[HISTOGRAM] Failed to match histogram: {e}. Using original image.")
        return moving


def register_images(level, fixed, moving, fixed_mask=None, moving_mask=None, fixed_study_type=None, moving_study_type=None):
    """
    Unified registration function for all levels of the hierarchical registration pipeline.

    Handles three registration levels with level-specific behavior:
    - 'pt_anchor_to_template': Two-step (initializer + main), no masks in initializer,
      mask_all_stages=True in main registration.
    - 'tp_anchor_to_pt_anchor': Two-step (initializer + main), masks passed to initializer,
      no mask_all_stages in main registration.
    - 'study_to_tp_anchor': Single-step registration with masks, Identity initial transform
      when center alignment is disabled.

    Args:
        level: Registration level key, one of 'pt_anchor_to_template',
               'tp_anchor_to_pt_anchor', or 'study_to_tp_anchor'.
        fixed: Fixed ANTs image (the image being registered to).
        moving: Moving ANTs image (the reference/target space).
        fixed_mask: Optional brain extraction mask for fixed image.
        moving_mask: Optional brain extraction mask for moving image.
        fixed_study_type: Study type of fixed image (for metric selection).
        moving_study_type: Study type of moving image (for metric selection).

    Returns:
        list: List of inverse transform file paths for apply_transforms.

    Raises:
        ValueError: If post-registration validation fails.
    """
    # Level display names for logging
    level_labels = {
        'pt_anchor_to_template': 'Patient anchor to template',
        'tp_anchor_to_pt_anchor': 'Timepoint anchor to patient anchor',
        'study_to_tp_anchor': 'Study to timepoint anchor',
    }
    label = level_labels.get(level, level)

    if TEST_MODE:
        log.info(f"[TEST_MODE] Skipping register_images ({level})")
        return []

    # Validate images before registration to catch NaN/Inf spacing issues
    fixed = validate_image_for_registration(fixed, f"{level}: fixed")
    moving = validate_image_for_registration(moving, f"{level}: moving")

    # Apply histogram matching for same-modality registration
    moving = apply_histogram_matching(fixed, moving, fixed_study_type, moving_study_type)

    # Calculate MI before registration
    try:
        mi_before = ants.image_mutual_information(fixed, moving)
    except Exception as e:
        log.warning(f"Failed to calculate MI before registration: {e}")
        mi_before = 0.0
    log.info(f"[MI BEFORE] {label}: {mi_before:.6f}")

    # Determine affine metric based on study types
    aff_metric = get_aff_metric(fixed_study_type, moving_study_type)

    # Determine if this is a two-step (initializer + main) or single-step registration
    config = REGISTRATION_CONFIG[level]
    is_two_step = 'initializer' in config

    if is_two_step:
        # --- Two-step registration (pt_anchor_to_template, tp_anchor_to_pt_anchor) ---
        init_method = config['initializer']
        main_method = config['main']
        init_params = INITIALIZER_PARAMS.get(init_method, INITIALIZER_PARAMS['DenseRigid'])
        main_params = REGISTRATION_PARAMS.get(main_method, REGISTRATION_PARAMS['Similarity'])

        # Center alignment
        if CENTER_ALIGNMENT_CONFIG.get(level, True):
            center_transform = align_image_centers(fixed, moving, fixed_mask=fixed_mask, moving_mask=moving_mask,
                                                    fixed_study_type=fixed_study_type, moving_study_type=moving_study_type)
            if center_transform is None:
                log.warning("Center alignment failed, attempting registration without initial transform")
        else:
            center_transform = None
            log.info(f"Center alignment disabled for {level} level")

        # Determine whether to pass masks in the initializer registration.
        # pt_anchor_to_template does NOT pass masks; tp_anchor_to_pt_anchor DOES.
        use_masks_in_init = (level != 'pt_anchor_to_template')

        # Initializer registration
        log.info(f"Applying {init_method} initialization with aff_metric='{aff_metric}'")
        init_kwargs = dict(
            fixed=fixed,
            moving=moving,
            type_of_transform=init_method,
            aff_metric=aff_metric,
            aff_iterations=init_params.get('aff_iterations', (200, 100)),
            aff_shrink_factors=init_params.get('aff_shrink_factors', (2, 1)),
            aff_smoothing_sigmas=init_params.get('aff_smoothing_sigmas', (1, 0)),
            aff_sampling=init_params.get('aff_sampling', 256),
            aff_random_sampling_rate=init_params.get('aff_random_sampling_rate', 0.9),
            mask_all_stages=True,
            grad_step=init_params.get('grad_step', 0.1),
            initial_transform=center_transform if center_transform else None,
            random_seed=SEED,
        )
        if use_masks_in_init:
            init_kwargs['mask'] = fixed_mask
            init_kwargs['moving_mask'] = moving_mask

        try:
            initializer = ants.registration(**init_kwargs)
        except Exception as init_error:
            log.error(f"Initialization registration failed: {init_error}")
            log.info("Retrying initialization without center transform and masks")
            retry_kwargs = dict(
                fixed=fixed,
                moving=moving,
                type_of_transform=init_method,
                aff_metric=aff_metric,
                aff_iterations=init_params.get('aff_iterations', (200, 100)),
                aff_shrink_factors=init_params.get('aff_shrink_factors', (2, 1)),
                aff_smoothing_sigmas=init_params.get('aff_smoothing_sigmas', (1, 0)),
                aff_sampling=init_params.get('aff_sampling', 256),
                aff_random_sampling_rate=init_params.get('aff_random_sampling_rate', 0.9),
                mask_all_stages=False,
                grad_step=init_params.get('grad_step', 0.1),
                random_seed=SEED,
            )
            initializer = ants.registration(**retry_kwargs)

        # Main registration
        # pt_anchor_to_template uses mask_all_stages=True; tp_anchor_to_pt_anchor does not
        log.info(f"Applying {main_method} registration with aff_metric='{aff_metric}'")
        main_kwargs = dict(
            fixed=fixed,
            moving=moving,
            type_of_transform=main_method,
            aff_metric=aff_metric,
            aff_iterations=main_params.get('aff_iterations', (2000, 1000)),
            aff_shrink_factors=main_params.get('aff_shrink_factors', (2, 1)),
            aff_smoothing_sigmas=main_params.get('aff_smoothing_sigmas', (1, 0)),
            aff_sampling=main_params.get('aff_sampling', 256),
            aff_random_sampling_rate=main_params.get('aff_random_sampling_rate', 0.7),
            initial_transform=initializer['fwdtransforms'],
            single_precision=False,
            grad_step=main_params.get('grad_step', 0.1),
            random_seed=SEED,
        )
        if level == 'pt_anchor_to_template':
            main_kwargs['mask_all_stages'] = True

        reg_result = ants.registration(**main_kwargs)

    else:
        # --- Single-step registration (study_to_tp_anchor) ---
        reg_method = config['method']
        reg_params = REGISTRATION_PARAMS.get(reg_method, REGISTRATION_PARAMS['DenseRigid'])

        # Center alignment (usually unnecessary for study_to_tp_anchor)
        if CENTER_ALIGNMENT_CONFIG.get(level, False):
            center_transform = align_image_centers(fixed, moving, fixed_mask=fixed_mask, moving_mask=moving_mask,
                                                    fixed_study_type=fixed_study_type, moving_study_type=moving_study_type)
            if center_transform is None:
                log.warning("Center alignment failed, using Identity initial transform")
                center_transform = 'Identity'
        else:
            center_transform = 'Identity'

        log.info(f"Applying {reg_method} registration with aff_metric='{aff_metric}'")
        reg_result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=reg_method,
            mask=fixed_mask,
            moving_mask=moving_mask,
            aff_metric=aff_metric,
            aff_iterations=reg_params.get('aff_iterations', (20000, 10000)),
            aff_shrink_factors=reg_params.get('aff_shrink_factors', (2, 1)),
            aff_smoothing_sigmas=reg_params.get('aff_smoothing_sigmas', (1, 0)),
            aff_sampling=reg_params.get('aff_sampling', 256),
            aff_random_sampling_rate=reg_params.get('aff_random_sampling_rate', 0.9),
            grad_step=reg_params.get('grad_step', 0.1),
            initial_transform=center_transform,
            single_precision=False,
            mask_all_stages=True,
            random_seed=SEED
        )

    # Extract transforms before cleanup
    inv_transforms = reg_result['invtransforms']

    # Cleanup registration result dict (contains large warped image)
    if is_two_step:
        del initializer
    del reg_result
    gc.collect()

    # Calculate MI after registration and validate
    try:
        warped_check = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=inv_transforms,
            interpolator='lanczosWindowedSinc',
            whichtoinvert=[True] * len(inv_transforms)
        )

        if not validate_warped_image(warped_check, f"{level}: warped_check"):
            raise ValueError("Warped image has invalid spacing or dimensions after apply_transforms")

        mi_after = ants.image_mutual_information(fixed, warped_check)
        log.info(f"[MI AFTER]  {label}: {mi_after:.6f} (change: {mi_after - mi_before:+.6f})")
    except Exception as e:
        log.warning(f"Failed to calculate MI after registration: {e}")
        raise ValueError(f"Post-registration validation failed: {str(e)}")
    finally:
        # Cleanup validation image
        if 'warped_check' in dir():
            del warped_check
            gc.collect()

    return inv_transforms


# ============================================================================
# UTILITY HELPERS
# ============================================================================

def select_anchor(patient_df, modality=None, prefer_coregistered=False):
    """
    Select the anchor image based on the configured metric from the remaining df.

    The metric used for ranking is controlled by ANCHOR_SELECTION_METRIC config:
        - 'column': which DataFrame column to sort by (e.g., 'file_size_mb', 'entropy')
        - 'ascending': sort order (False = prefer highest, True = prefer lowest)

    Args:
        patient_df: DataFrame containing patient image metadata
        modality: Optional string to filter by Study Type (e.g., 'T1', 'T2')
        prefer_coregistered: If True, prioritize coregistered studies over raw studies

    Returns:
        Series: The selected anchor row

    Raises:
        ValueError: If patient_df is empty or has invalid structure
    """
    # Read anchor selection metric from config
    metric_col = ANCHOR_SELECTION_METRIC['column']
    metric_ascending = ANCHOR_SELECTION_METRIC['ascending']

    # Safety check: validate input DataFrame
    if patient_df is None or patient_df.empty:
        raise ValueError("patient_df cannot be None or empty")

    required_cols = [metric_col, 'segmentation path']
    missing_cols = [col for col in required_cols if col not in patient_df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    # Filter by modality if specified
    search_df = patient_df.copy()  # Preserve original for fallback
    if modality is not None:
        if 'Study Type' not in patient_df.columns:
            raise ValueError("'Study Type' column not found for modality filtering")
        filtered_by_modality = patient_df[patient_df['Study Type'].str.contains(modality, na=False)]
        if filtered_by_modality.empty:
            log.warning(f'No images found with modality "{modality}". Using all available images.')
            search_df = patient_df.copy()  # Use original, not the empty filtered result
        else:
            search_df = filtered_by_modality.copy()

    # Prefer coregistered studies if requested (check for non-null values)
    original_search_df = search_df.copy()  # Keep for fallback
    if prefer_coregistered:
        coregistered_df = search_df[search_df['coregistered_image'].notna()]
        if not coregistered_df.empty:
            search_df = coregistered_df.copy()
            log.info(f'Preferring coregistered studies for anchor selection ({len(coregistered_df)} available)')
        else:
            log.info(f'No coregistered studies available, using raw studies for anchor selection')

    # Sort by configured metric and select best candidate
    ordered_df = search_df.sort_values(by=metric_col, ascending=metric_ascending)

    if ordered_df.empty:
        raise ValueError("No images available for anchor selection")

    anchor_row = ordered_df.iloc[0]
    coreg_status = "coregistered" if (prefer_coregistered and 'coregistered_image' in anchor_row.index and pd.notna(anchor_row['coregistered_image'])) else "raw"
    has_seg = "with segmentation" if pd.notna(anchor_row.get('segmentation path')) else "without segmentation"
    sort_direction = "lowest" if metric_ascending else "highest"
    metric_value = anchor_row[metric_col]
    log.info(f'Selected {coreg_status} anchor {has_seg} by {sort_direction} {metric_col}: {metric_value}')

    return anchor_row


def select_and_load_anchor_with_fallback(patient_df, modality=None, prefer_coregistered=False, 
                                          bet_modality='t2', max_attempts=5, template_bets=None):
    """
    Select and load an anchor image with fallback logic for invalid images.
    
    If the selected anchor fails to load (e.g., invalid spacing, corrupted file),
    this function will try the next best anchor until success or no more candidates.
    
    Args:
        patient_df: DataFrame containing patient image metadata
        modality: Optional string to filter by Study Type (e.g., 'T1', 'T2')
        prefer_coregistered: If True, prioritize coregistered studies over raw studies
        bet_modality: Modality for brain extraction ('t1' or 't2')
        max_attempts: Maximum number of anchors to try before giving up
        template_bets: Dictionary of template images for histogram matching
    
    Returns:
        tuple: (anchor_row, anchor_image, anchor_seg, anchor_bet, failed_indices)
            - anchor_row: The selected anchor row (Series)
            - anchor_image: Loaded and preprocessed ANTs image
            - anchor_seg: Loaded segmentation (or None)
            - anchor_bet: Brain extraction mask (or None)
            - failed_indices: List of row indices that failed (for error logging)
    
    Raises:
        ValueError: If no valid anchor could be loaded after all attempts
    """
    candidates_df = patient_df.copy()
    failed_indices = []
    attempt = 0
    
    while attempt < max_attempts and not candidates_df.empty:
        attempt += 1
        
        try:
            # Select the best anchor from remaining candidates
            anchor_row = select_anchor(candidates_df, modality=modality, prefer_coregistered=prefer_coregistered)
            anchor_filename = os.path.basename(anchor_row.get("image path", "N/A"))
            
            log.info(f'[ANCHOR ATTEMPT {attempt}/{max_attempts}] Trying: {anchor_filename}')
            
            # Try to load the anchor image
            anchor_image, anchor_seg = get_ants_image(anchor_row, prefer_coregistered=prefer_coregistered, template_bets=template_bets)
            
            # Validate the loaded image (this will raise if invalid)
            if anchor_image is not None:
                anchor_image = validate_image_for_registration(anchor_image, f"anchor: {anchor_filename}")
            
            # Get brain extraction mask
            anchor_bet = get_bet_image(anchor_image, anchor_row, modality=bet_modality)
            
            log.info(f'[ANCHOR SUCCESS] Successfully loaded anchor: {anchor_filename}')
            
            # Return successful anchor
            return anchor_row, anchor_image, anchor_seg, anchor_bet, failed_indices
            
        except Exception as e:
            # Log the failure
            anchor_filename = anchor_row.get("image path", "N/A") if 'anchor_row' in dir() else "unknown"
            log.info(f'[ANCHOR FAILED] {os.path.basename(str(anchor_filename))}: {type(e).__name__}: {str(e)[:100]}')
            
            # Track failed index for error logging
            if 'anchor_row' in dir() and anchor_row is not None:
                failed_indices.append((anchor_row.name, str(e)))
                # Remove failed anchor from candidates
                candidates_df = candidates_df[candidates_df.index != anchor_row.name]
            
            if candidates_df.empty:
                log.info(f'[ANCHOR EXHAUSTED] No more candidates available after {attempt} attempts')
                break
            else:
                remaining = len(candidates_df)
                log.info(f'[ANCHOR RETRY] {remaining} candidates remaining, trying next best...')
    
    # All attempts failed
    error_msg = f"Failed to load any valid anchor after {attempt} attempts. Failed indices: {[idx for idx, _ in failed_indices]}"
    raise ValueError(error_msg)


def select_and_register_patient_anchor_with_retry(patient_df, patient_template_image, patient_template_bet,
                                                   template_study_type, modality='t2_thin',
                                                   prefer_coregistered=False, bet_modality='t2',
                                                   max_attempts=5, template_bets=None):
    """
    Select a patient anchor and register to template with retry on registration failure.

    If the registration fails (not just loading), this function will try selecting a
    different anchor and retry until success or no more candidates.

    If all candidates of the preferred modality fail but retry attempts remain,
    the function will expand to other modalities.

    Args:
        patient_df: DataFrame containing patient image metadata
        patient_template_image: Template image to register to
        patient_template_bet: Template brain extraction mask
        template_study_type: Study type of the template (for metric selection)
        template_bets: Dictionary of template images for histogram matching
        modality: Optional string to filter by Study Type (will expand to all if exhausted)
        prefer_coregistered: If True, prioritize coregistered studies
        bet_modality: Modality for brain extraction ('t1' or 't2')
        max_attempts: Maximum number of anchors to try before giving up

    Returns:
        tuple: (anchor_row, anchor_image, anchor_seg, anchor_bet, anchor_study_type,
                tx_to_template, failed_indices)

    Raises:
        ValueError: If no valid anchor could be registered after all attempts
    """
    all_candidates_df = patient_df.copy()
    candidates_df = patient_df.copy()
    failed_indices = []
    attempt = 0
    current_modality = modality
    modality_expanded = False

    while attempt < max_attempts:
        attempt += 1
        anchor_row = None
        anchor_image = None
        anchor_seg = None
        anchor_bet = None
        
        # Check if we need to expand to other modalities
        if candidates_df.empty and not modality_expanded and attempt < max_attempts:
            # Remove already-failed indices from all_candidates_df
            remaining_candidates = all_candidates_df[~all_candidates_df.index.isin([idx for idx, _ in failed_indices])]
            if not remaining_candidates.empty:
                log.info(f'[PATIENT ANCHOR EXPAND] Preferred modality "{current_modality}" exhausted. Expanding to all modalities.')
                candidates_df = remaining_candidates
                current_modality = None  # No modality filter
                modality_expanded = True
            else:
                log.info(f'[PATIENT ANCHOR EXHAUSTED] All candidates exhausted after {attempt} attempts')
                break
        elif candidates_df.empty:
            log.info(f'[PATIENT ANCHOR EXHAUSTED] No more candidates available after {attempt} attempts')
            break

        try:
            # Select the best anchor from remaining candidates
            anchor_row = select_anchor(candidates_df, modality=current_modality, prefer_coregistered=prefer_coregistered)
            anchor_filename = os.path.basename(anchor_row.get("image path", "N/A"))

            modality_msg = f" (expanded to all modalities)" if modality_expanded else f" (modality: {current_modality})"
            log.info(f'[PATIENT ANCHOR ATTEMPT {attempt}/{max_attempts}] Trying: {anchor_filename}{modality_msg}')
            
            # Try to load the anchor image
            anchor_image, anchor_seg = get_ants_image(anchor_row, prefer_coregistered=prefer_coregistered, template_bets=template_bets)
            
            # Validate the loaded image
            if anchor_image is not None:
                anchor_image = validate_image_for_registration(anchor_image, f"patient anchor: {anchor_filename}")
            
            # Get brain extraction mask
            anchor_bet = get_bet_image(anchor_image, anchor_row, modality=bet_modality)
            
            log.info(f'[PATIENT ANCHOR LOADED] {anchor_filename}')
            
            # Now attempt registration
            anchor_study_type = anchor_row.get('Study Type', '')
            
            log.info(f'[PATIENT ANCHOR REGISTERING] Attempting registration to template...')
            tx_to_template = register_images('pt_anchor_to_template',
                fixed=anchor_image,
                moving=patient_template_image,
                fixed_mask=anchor_bet,
                moving_mask=patient_template_bet,
                fixed_study_type=anchor_study_type,
                moving_study_type=template_study_type
            )
            
            log.info(f'[PATIENT ANCHOR SUCCESS] Registration completed for: {anchor_filename}')
            
            # Return successful anchor and registration
            return anchor_row, anchor_image, anchor_seg, anchor_bet, anchor_study_type, tx_to_template, failed_indices
            
        except Exception as e:
            # Log the failure
            anchor_filename = anchor_row.get("image path", "N/A") if anchor_row is not None else "unknown"
            log.info(f'[PATIENT ANCHOR FAILED] {os.path.basename(str(anchor_filename))}: {type(e).__name__}: {str(e)[:200]}')
            
            # Track failed index for error logging
            if anchor_row is not None:
                failed_indices.append((anchor_row.name, f"Registration failed: {str(e)}"))
                # Remove failed anchor from candidates
                candidates_df = candidates_df[candidates_df.index != anchor_row.name]
            
            # Cleanup
            if anchor_image is not None:
                del anchor_image
            if anchor_seg is not None:
                del anchor_seg
            if anchor_bet is not None:
                del anchor_bet
            gc.collect()

    # All attempts failed
    error_msg = f"Failed to register any valid patient anchor after {attempt} attempts. Failed indices: {[idx for idx, _ in failed_indices]}"
    raise ValueError(error_msg)


def select_and_register_tp_anchor_with_retry(timepoint_df, patient_anchor_image, patient_anchor_seg,
                                              patient_anchor_study_type, patient_anchor_row_name,
                                              modality='t2_thin', prefer_coregistered=False,
                                              bet_modality='t2', max_attempts=5, template_bets=None):
    """
    Select a timepoint anchor and register to patient anchor with retry on registration failure.

    If the registration fails (not just loading), this function will try selecting a
    different anchor and retry until success or no more candidates.

    If all candidates of the preferred modality fail but retry attempts remain,
    the function will expand to other modalities.

    Args:
        timepoint_df: DataFrame containing timepoint image metadata
        patient_anchor_image: Patient anchor image to register to
        patient_anchor_seg: Patient anchor segmentation mask
        patient_anchor_study_type: Study type of the patient anchor (for metric selection)
        patient_anchor_row_name: Index of patient anchor row (to skip if same)
        modality: Optional string to filter by Study Type (will expand to all if exhausted)
        template_bets: Dictionary of template images for histogram matching
        prefer_coregistered: If True, prioritize coregistered studies
        bet_modality: Modality for brain extraction ('t1' or 't2')
        max_attempts: Maximum number of anchors to try before giving up

    Returns:
        tuple: (anchor_row, anchor_image, anchor_seg, anchor_bet, anchor_study_type,
                tx_to_pt_anchor, failed_indices, is_same_as_patient_anchor)

    Raises:
        ValueError: If no valid anchor could be registered after all attempts
    """
    all_candidates_df = timepoint_df.copy()
    candidates_df = timepoint_df.copy()
    failed_indices = []
    attempt = 0
    current_modality = modality
    modality_expanded = False

    while attempt < max_attempts:
        attempt += 1
        anchor_row = None
        anchor_image = None
        anchor_seg = None
        anchor_bet = None
        
        # Check if we need to expand to other modalities
        if candidates_df.empty and not modality_expanded and attempt < max_attempts:
            # Remove already-failed indices from all_candidates_df
            remaining_candidates = all_candidates_df[~all_candidates_df.index.isin([idx for idx, _ in failed_indices])]
            if not remaining_candidates.empty:
                log.info(f'[TP ANCHOR EXPAND] Preferred modality "{current_modality}" exhausted. Expanding to all modalities.')
                candidates_df = remaining_candidates
                current_modality = None  # No modality filter
                modality_expanded = True
            else:
                log.info(f'[TP ANCHOR EXHAUSTED] All candidates exhausted after {attempt} attempts')
                break
        elif candidates_df.empty:
            log.info(f'[TP ANCHOR EXHAUSTED] No more candidates available after {attempt} attempts')
            break

        try:
            # Select the best anchor from remaining candidates
            anchor_row = select_anchor(candidates_df, modality=current_modality, prefer_coregistered=prefer_coregistered)
            anchor_filename = os.path.basename(anchor_row.get("image path", "N/A"))

            modality_msg = f" (expanded to all modalities)" if modality_expanded else f" (modality: {current_modality})"
            log.info(f'[TP ANCHOR ATTEMPT {attempt}/{max_attempts}] Trying: {anchor_filename}{modality_msg}')

            # Check if this is the same as the patient anchor
            if anchor_row.name == patient_anchor_row_name:
                log.info(f'[TP ANCHOR] Same as patient anchor, skipping registration')
                # Load the anchor anyway for processing other studies
                anchor_image, anchor_seg = get_ants_image(anchor_row, prefer_coregistered=prefer_coregistered, template_bets=template_bets)
                if anchor_image is not None:
                    anchor_image = validate_image_for_registration(anchor_image, f"tp anchor: {anchor_filename}")
                anchor_bet = get_bet_image(anchor_image, anchor_row, modality=bet_modality)
                anchor_study_type = anchor_row.get('Study Type', '')
                return anchor_row, anchor_image, anchor_seg, anchor_bet, anchor_study_type, [], failed_indices, True
            
            # Try to load the anchor image
            anchor_image, anchor_seg = get_ants_image(anchor_row, prefer_coregistered=prefer_coregistered, template_bets=template_bets)
            
            # Validate the loaded image
            if anchor_image is not None:
                anchor_image = validate_image_for_registration(anchor_image, f"tp anchor: {anchor_filename}")
            
            # Get brain extraction mask
            anchor_bet = get_bet_image(anchor_image, anchor_row, modality=bet_modality)
            
            log.info(f'[TP ANCHOR LOADED] {anchor_filename}')
            
            # Now attempt registration
            anchor_study_type = anchor_row.get('Study Type', '')
            
            log.info(f'[TP ANCHOR REGISTERING] Attempting registration to patient anchor...')
            tx_to_pt_anchor = register_images('tp_anchor_to_pt_anchor',
                fixed=anchor_image,
                moving=patient_anchor_image,
                fixed_mask=anchor_seg,
                moving_mask=patient_anchor_seg,
                fixed_study_type=anchor_study_type,
                moving_study_type=patient_anchor_study_type
            )
            
            log.info(f'[TP ANCHOR SUCCESS] Registration completed for: {anchor_filename}')
            
            # Return successful anchor and registration
            return anchor_row, anchor_image, anchor_seg, anchor_bet, anchor_study_type, tx_to_pt_anchor, failed_indices, False
            
        except Exception as e:
            # Log the failure
            anchor_filename = anchor_row.get("image path", "N/A") if anchor_row is not None else "unknown"
            log.info(f'[TP ANCHOR FAILED] {os.path.basename(str(anchor_filename))}: {type(e).__name__}: {str(e)[:200]}')
            
            # Track failed index for error logging
            if anchor_row is not None:
                failed_indices.append((anchor_row.name, f"Registration failed: {str(e)}"))
                # Remove failed anchor from candidates
                candidates_df = candidates_df[candidates_df.index != anchor_row.name]
            
            # Cleanup
            if anchor_image is not None:
                del anchor_image
            if anchor_seg is not None:
                del anchor_seg
            if anchor_bet is not None:
                del anchor_bet
            gc.collect()

    # All attempts failed
    error_msg = f"Failed to register any valid timepoint anchor after {attempt} attempts. Failed indices: {[idx for idx, _ in failed_indices]}"
    raise ValueError(error_msg)


def get_bet_image(image, row, modality='t2'):
    """
    Extract brain mask from image or load pre-computed mask.
    If mask covers <10% of voxels, replace with an all-ones mask.
    
    Args:
        image: ANTs image object
        row: DataFrame row containing image metadata
        modality: Modality for brain extraction ('t1' or 't2')
    
    Returns:
        ANTs image: Brain extraction mask, or None if TEST_MODE
    
    Raises:
        RuntimeError: If mask has zero voxels
    """    
    # Try to load pre-computed mask from hd_output
    img_path = row.get('image path', '')
    base = os.path.basename(img_path)
    
    # Validate that base is a proper .nii.gz filename before constructing mask path
    if base and base.endswith('.nii.gz'):
        bet_mask_path = os.path.join(HD_OUTPUT_DIR, base.replace('.nii.gz', '_bet.nii.gz'))
    else:
        bet_mask_path = None  # Invalid filename, skip pre-computed mask lookup

    if TEST_MODE or image is None:
        return None

    # Check if bet_mask_path exists and is a file (not a directory)
    if bet_mask_path and os.path.isfile(bet_mask_path):
        log.info(f'Loading pre-computed BET mask: {bet_mask_path}')
        try:
            mask = ants.image_read(bet_mask_path)
            # Fix zero spacing issues that cause ITK errors
            mask = fix_zero_spacing(mask, bet_mask_path)
            # Resample pre-computed mask to match target image space (image may have been reoriented/resampled)
            mask = ants.resample_image_to_target(mask, image, interp_type='nearestNeighbor')
        except Exception as e:
            log.warning(f'Failed to read pre-computed mask {bet_mask_path}: {e}')
            mask = _cached_brain_extraction(image, modality=modality, cache_path=bet_mask_path)
    elif bet_mask_path and os.path.exists(bet_mask_path):
        log.warning(f'bet_mask_path is not a file: {bet_mask_path}')
        mask = _cached_brain_extraction(image, modality=modality)
    else:
        mask = _cached_brain_extraction(image, modality=modality, cache_path=bet_mask_path)

    # Validate and fix mask if needed
    mask_np = mask.numpy()
    nonzero = (mask_np > 0).sum()
    total = mask_np.size
    
    if total == 0:
        raise RuntimeError("Mask has zero voxels")
    
    coverage_pct = (nonzero / total) * 100
    log.info(f'Mask coverage: {coverage_pct:.2f}%')

    if coverage_pct < 10:
        log.warning(f'Mask coverage < 10%. Using all-ones mask instead.')
        ones_np = np.ones_like(mask_np, dtype=mask_np.dtype)
        mask = ants.from_numpy(ones_np, origin=mask.origin, spacing=mask.spacing, direction=mask.direction)

    del mask_np
    return mask


def get_head_mask(image, row=None, bet_mask=None, modality='t2', force=False):
    """
    Get a head mask (including skull) for background removal.

    Strategy:
    1. If bet_mask provided, dilate it to include skull
    2. If no bet_mask but row provided, try to load/compute brain mask then dilate
    3. Fallback to Otsu intensity thresholding via ants.get_mask()

    Args:
        image: ANTs image object
        row: Optional DataFrame row for loading pre-computed masks
        bet_mask: Optional pre-computed brain extraction mask
        modality: Modality for brain extraction if needed ('t1' or 't2')
        force: If True, compute mask even if ENABLE_HEAD_MASKING is False
               (useful for other purposes like QC visualization)

    Returns:
        ANTs image: Binary head mask (including skull region), or None if disabled/TEST_MODE

    Note:
        Respects the ENABLE_HEAD_MASKING configuration flag unless force=True.
        Returns None early if head masking is disabled to prevent unnecessary computation.
    """
    # Defense-in-depth: check the master flag first (unless forced)
    if not force and not ENABLE_HEAD_MASKING:
        log.info('[HEAD_MASK] Skipped - ENABLE_HEAD_MASKING is False')
        return None

    if TEST_MODE or image is None:
        return None
    
    # Calculate dilation radius in voxels from mm
    # Use minimum spacing dimension to be conservative
    min_spacing = min(image.spacing)
    dilation_radius_voxels = int(np.ceil(HEAD_MASK_DILATION_MM / min_spacing))
    log.info(f'[HEAD_MASK] Dilation: {HEAD_MASK_DILATION_MM}mm = {dilation_radius_voxels} voxels (spacing: {min_spacing:.2f}mm)')
    
    dilated_mask = None
    
    # Strategy 1: Use provided bet_mask and dilate
    if bet_mask is not None:
        log.info('[HEAD_MASK] Using provided brain mask, dilating to head region')
        try:
            # Resample mask to match target image space
            bet_mask_resampled = ants.resample_image_to_target(bet_mask, image, interp_type='nearestNeighbor')
            dilated_mask = ants.morphology(bet_mask_resampled, operation='dilate', radius=dilation_radius_voxels)
            del bet_mask_resampled
        except Exception as e:
            log.info(f'[HEAD_MASK] Failed to dilate provided mask: {e}')
    
    # Strategy 2: Try to load/compute brain mask from row, then dilate
    if dilated_mask is None and row is not None:
        try:
            # Try to load pre-computed mask
            img_path = row.get('image path', '')
            base = os.path.basename(img_path)
            
            # Validate that base is a proper .nii.gz filename before constructing mask path
            if base and base.endswith('.nii.gz'):
                bet_mask_path = os.path.join(HD_OUTPUT_DIR, base.replace('.nii.gz', '_bet.nii.gz'))
            else:
                bet_mask_path = None  # Invalid filename, skip pre-computed mask lookup

            # Check if bet_mask_path exists and is a file (not a directory)
            if bet_mask_path and os.path.isfile(bet_mask_path):
                log.info(f'[HEAD_MASK] Loading pre-computed brain mask: {bet_mask_path}')
                try:
                    brain_mask = ants.image_read(bet_mask_path)
                    # Fix zero spacing issues that cause ITK errors
                    brain_mask = fix_zero_spacing(brain_mask, bet_mask_path)
                    # Resample pre-computed mask to match target image space (image may have been reoriented/resampled)
                    brain_mask = ants.resample_image_to_target(brain_mask, image, interp_type='nearestNeighbor')
                except Exception as e:
                    log.info(f'[HEAD_MASK] Failed to read pre-computed mask: {e}, computing new mask')
                    brain_mask = _cached_brain_extraction(image, modality=modality, cache_path=bet_mask_path)
            elif bet_mask_path and os.path.exists(bet_mask_path):
                log.info(f'[HEAD_MASK] bet_mask_path is not a file: {bet_mask_path}, computing new mask')
                brain_mask = _cached_brain_extraction(image, modality=modality)
            else:
                log.info(f'[HEAD_MASK] Computing brain mask for dilation (modality: {modality})')
                brain_mask = _cached_brain_extraction(image, modality=modality, cache_path=bet_mask_path)

            # Dilate to get head mask
            dilated_mask = ants.morphology(brain_mask, operation='dilate', radius=dilation_radius_voxels)
            del brain_mask
            gc.collect()
            log.info('[HEAD_MASK] Successfully created head mask from brain mask')
        except Exception as e:
            log.info(f'[HEAD_MASK] Failed to create mask from brain extraction: {e}')
    
    # Strategy 3: Fallback to Otsu thresholding
    if dilated_mask is None:
        log.info('[HEAD_MASK] Falling back to Otsu intensity thresholding')
        try:
            # ants.get_mask uses Otsu thresholding with morphological cleanup
            # cleanup=2 applies opening then closing to remove small islands
            dilated_mask = ants.get_mask(image, low_thresh=None, high_thresh=None, cleanup=2)
            log.info('[HEAD_MASK] Successfully created head mask via Otsu thresholding')
        except Exception as e:
            log.info(f'[HEAD_MASK] Otsu thresholding failed: {e}')
            # Ultimate fallback: return all-ones mask (no masking)
            log.info('[HEAD_MASK] WARNING: All methods failed, returning all-ones mask')
            ones_np = np.ones(image.shape, dtype=np.float32)
            dilated_mask = ants.from_numpy(ones_np, origin=image.origin, spacing=image.spacing, direction=image.direction)
    
    # Validate mask coverage
    mask_np = dilated_mask.numpy()
    nonzero = (mask_np > 0).sum()
    total = mask_np.size
    coverage_pct = (nonzero / total) * 100
    log.info(f'[HEAD_MASK] Final mask coverage: {coverage_pct:.2f}%')

    if coverage_pct < HEAD_MASK_MIN_COVERAGE:
        log.info(f'[HEAD_MASK] WARNING: Coverage < {HEAD_MASK_MIN_COVERAGE}%, using all-ones mask')
        ones_np = np.ones(image.shape, dtype=np.float32)
        dilated_mask = ants.from_numpy(ones_np, origin=image.origin, spacing=image.spacing, direction=image.direction)
    
    # Ensure binary mask (threshold at 0.5)
    mask_np = dilated_mask.numpy()
    mask_np = (mask_np > 0.5).astype(np.float32)
    dilated_mask = ants.from_numpy(mask_np, origin=dilated_mask.origin, spacing=dilated_mask.spacing, direction=dilated_mask.direction)
    
    return dilated_mask


def apply_head_mask(image, head_mask, force=False):
    """
    Zero out voxels outside the head mask.

    Args:
        image: ANTs image to mask
        head_mask: Binary head mask (1 inside head, 0 outside)
        force: If True, apply mask even if ENABLE_HEAD_MASKING is False

    Returns:
        ANTs image with background set to 0, or original image if masking disabled/skipped

    Note:
        Respects the ENABLE_HEAD_MASKING configuration flag unless force=True.
        This is a defense-in-depth check - callers should also check the flag.
    """
    # Defense-in-depth: check the master flag (unless forced)
    if not force and not ENABLE_HEAD_MASKING:
        return image

    if image is None or head_mask is None:
        return image

    # Multiply image by mask to zero out background
    masked_image = image * head_mask
    return masked_image


def apply_transitive_transforms(transformlist, fixed, moving, seg=None):
    """
    Apply a list of transforms to image and segmentation.
    In TEST_MODE, returns None values without processing.
    """
    if TEST_MODE:
        log.info("[TEST_MODE] Skipping apply_transitive_transforms")
        return None, None
    
    # Warp study
    warped_anchor = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transformlist,
        interpolator='lanczosWindowedSinc',
        whichtoinvert=[True] * len(transformlist)
    )

    warped_anchor_seg = None
    if seg is not None:
        warped_anchor_seg = ants.apply_transforms(
            fixed=fixed,
            moving=seg,
            transformlist=transformlist,
            interpolator='genericLabel',
            whichtoinvert=[True] * len(transformlist)
        )
    return warped_anchor, warped_anchor_seg


def generate_overlay_safety_check(warped_anchor, warped_curr, anchor_filename, curr_filename, 
                                  mi_value, step_title="", output_dir=OVERLAY_OUTPUT_DIR):
    """
    Generate overlay plot for quality control showing all three orientations in one image.
    In TEST_MODE, returns a placeholder path without writing to disk.
    """
    if TEST_MODE:
        log.info(f"[TEST_MODE] Skipping overlay plot file write for {curr_filename}")
        return os.path.join(output_dir, f'{curr_filename}_overlay_testmode.png')
    
    # Save overlay comparison plot with all three orientations
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f'{curr_filename}_overlay.png'
    plot_filepath = os.path.join(output_dir, plot_filename)
    
    try:
        import tempfile
        from PIL import Image
        
        # Define orientations: axis=0 (sagittal), axis=1 (coronal), axis=2 (axial)
        orientations = [
            ('Axial', 2),
            ('Sagittal', 0),
            ('Coronal', 1)
        ]
        
        temp_files = []
        
        # Generate a plot for each orientation (reduced slices for smaller output)
        for orientation_name, axis_num in orientations:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_files.append(temp_file.name)
            temp_file.close()
            
            ants.plot(
                warped_anchor,
                overlay=warped_curr,
                overlay_alpha=0.225,
                axis=axis_num,
                nslices=12,  # Reduced from 21 for smaller plots
                title=f'{orientation_name}',
                filename=temp_files[-1]
            )
            plt.close('all')
        
        # Load and stitch images together horizontally
        images = [Image.open(f) for f in temp_files]
        
        # Scale down images to reduce final size
        scale_factor = 0.5  # Reduce to 50% of original size
        scaled_images = []
        for img in images:
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            scaled_images.append(scaled_img)
        
        # Calculate combined dimensions using scaled images
        widths = [img.width for img in scaled_images]
        heights = [img.height for img in scaled_images]
        total_width = sum(widths)
        max_height = max(heights)
        
        # Add space for title at top (scaled down)
        title_height = 40
        combined_height = max_height + title_height
        
        # Create combined image with white background
        combined = Image.new('RGB', (total_width, combined_height), color='white')
        
        # Paste each scaled image
        x_offset = 0
        for img in scaled_images:
            combined.paste(img, (x_offset, title_height))
            x_offset += img.width
        
        # Add title text using matplotlib (smaller figure)
        fig, ax = plt.subplots(figsize=(total_width/150, combined_height/150), dpi=100)
        ax.imshow(combined)
        ax.axis('off')
        ax.set_title(f'{step_title}\nBase: {anchor_filename} | Overlay: {curr_filename} | MI: {mi_value:.4f}', 
                     fontsize=8, pad=5)
        plt.tight_layout()
        plt.savefig(plot_filepath, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close('all')
        
        # Clean up temp files and PIL images
        for img in images:
            img.close()
        for img in scaled_images:
            img.close()
        for f in temp_files:
            try:
                os.remove(f)
            except OSError:
                pass
                
    except Exception as e:
        log.warning(f"Failed to generate overlay plot: {e}")
        plot_filepath = None
    finally:
        plt.close('all')
    
    return plot_filepath


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

# Global cache for template images in worker processes
# ANTs images cannot be pickled, so each worker must load them separately
_worker_template_bets = None


def _init_worker():
    """Initialize worker process with ITK thread settings and random seeds."""
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(THREADS)

    # Set random seeds for reproducibility in worker processes
    import random
    import numpy as np
    import warnings
    random.seed(SEED)
    np.random.seed(SEED)

    # TensorFlow seed for antspynet
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except ImportError:
        pass

    # Suppress matplotlib warnings in worker processes
    warnings.filterwarnings('ignore')


def _get_worker_templates(template_folder):
    """
    Get template images for the current worker process.
    
    Templates are loaded once per worker and cached in a global variable.
    This is necessary because ANTs images cannot be pickled/serialized
    for transfer between processes.
    
    Args:
        template_folder: Path to folder containing template NIfTI files
        
    Returns:
        dict: Dictionary of template images (same as load_templates output)
    """
    global _worker_template_bets
    
    if _worker_template_bets is None:
        log.info(f"[WORKER] Loading templates from {template_folder}")
        _worker_template_bets = load_templates(template_folder)
    
    return _worker_template_bets


def _process_single_patient_worker(args):
    """
    Worker function to process a single patient. Designed to be called in parallel.
    
    This function receives all necessary data serialized and returns updates to be
    applied to the main DataFrame.
    
    Args:
        args: Tuple of (patient_id, patient_df, template_folder, patient_idx, total_patients)
              Note: template_folder is a path string, not template_bets dict, because
              ANTs images cannot be pickled for multiprocessing.
        
    Returns:
        Dictionary with:
            - patient_id: The patient identifier
            - updates: List of (row_index, column, value) tuples
            - success: Boolean
            - error_message: Error message if failed
            - duration: Processing time in seconds
    """
    patient, patient_df, template_folder, patient_idx, total_patients = args

    # Save original patient indices for fallback error logging
    original_patient_indices = list(patient_df.index)

    # Load templates in worker process (cached after first load)
    template_bets = _get_worker_templates(template_folder)

    # Set ITK threads for this worker
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(THREADS)

    updates = []  # List of (row_index, column, value) tuples
    patient_start_time = perf_counter()
    
    def add_update(idx, column, value):
        """Helper to record a DataFrame update."""
        updates.append((idx, column, value))
    
    def log_error(row_index, error_details, fallback_indices=None):
        """
        Helper to log errors as updates.

        Args:
            row_index: Primary row index to log error to
            error_details: Error message/details
            fallback_indices: List of row indices to log to if row_index is None
                             (useful for logging errors to all studies in a failed timepoint)
        """
        timestamp = datetime.now().isoformat()
        if row_index is not None:
            add_update(row_index, 'error_details', error_details)
            add_update(row_index, 'error_timestamp', timestamp)
        elif fallback_indices is not None:
            # Log error to all fallback rows (e.g., all studies in a failed timepoint)
            for idx in fallback_indices:
                add_update(idx, 'error_details', error_details)
                add_update(idx, 'error_timestamp', timestamp)
            log.error(f"Error recorded for {len(fallback_indices)} studies in failed timepoint")
    
    patient_template_row = None
    patient_anchor_row = None
    
    try:
        num_studies = len(patient_df)
        num_timepoints = patient_df['Timepoint'].nunique()
        
        log.info(f"\n{'#'*80}")
        log.info(f"# PATIENT {patient_idx + 1}/{total_patients}: {patient}")
        log.info(f"# Studies: {num_studies} | Timepoints: {num_timepoints}")
        log.info(f"# Started at: {datetime.now().strftime('%H:%M:%S')}")
        log.info(f"{'#'*80}")
        
        # Incremental processing logic based on INCREMENTAL_MODE
        all_coregistered = patient_df['coregistered_image'].notna().all()
        already_coregistered = patient_df['coregistered_image'].notna().sum()
        remaining = num_studies - already_coregistered

        # Check preprocessing versions for existing coregistered images
        needs_reprocessing = patient_df[
            (patient_df['coregistered_image'].notna()) &
            (patient_df['preprocessing_version'] != PREPROCESSING_VERSION)
        ]
        outdated_count = len(needs_reprocessing)

        log.info(f'[STATUS] Already coregistered: {already_coregistered}/{num_studies} | Remaining: {remaining}')
        if outdated_count > 0:
            log.info(f'[STATUS] Outdated preprocessing version: {outdated_count}/{num_studies} (need re-preprocessing)')

        # Apply incremental mode logic
        if INCREMENTAL_MODE == "skip_all":
            # Original behavior: skip if ANY study is coregistered
            if all_coregistered:
                log.info(f'[SKIP] Patient {patient}: All {num_studies} images already coregistered (mode: skip_all)')
                return {
                    'patient_id': patient,
                    'updates': updates,
                    'success': True,
                    'skipped': True,
                    'error_message': None,
                    'duration': perf_counter() - patient_start_time
                }

        elif INCREMENTAL_MODE == "process_new":
            # Only process new studies (those without coregistered_image)
            if all_coregistered:
                log.info(f'[SKIP] Patient {patient}: All {num_studies} images already coregistered (mode: process_new)')
                return {
                    'patient_id': patient,
                    'updates': updates,
                    'success': True,
                    'skipped': True,
                    'error_message': None,
                    'duration': perf_counter() - patient_start_time
                }
            # Filter to only new studies
            log.info(f'[INCREMENTAL] Processing only {remaining} new studies (mode: process_new)')
            patient_df = patient_df[patient_df['coregistered_image'].isna()].copy()

        elif INCREMENTAL_MODE == "smart":
            # Process new studies + re-preprocess outdated ones
            if all_coregistered and outdated_count == 0:
                log.info(f'[SKIP] Patient {patient}: All {num_studies} images up-to-date (mode: smart)')
                return {
                    'patient_id': patient,
                    'updates': updates,
                    'success': True,
                    'skipped': True,
                    'error_message': None,
                    'duration': perf_counter() - patient_start_time
                }
            # Mark outdated studies for reprocessing
            studies_to_process = patient_df[
                (patient_df['coregistered_image'].isna()) |
                (patient_df['preprocessing_version'] != PREPROCESSING_VERSION)
            ].copy()
            total_to_process = len(studies_to_process)
            log.info(f'[INCREMENTAL] Processing {remaining} new + {outdated_count} outdated = {total_to_process} studies (mode: smart)')
            if REUSE_TRANSFORMATIONS and outdated_count > 0:
                log.info(f'[INCREMENTAL] Will reuse existing transformations for {outdated_count} outdated studies')

        elif INCREMENTAL_MODE == "force_reprocess":
            # Reprocess everything
            log.info(f'[INCREMENTAL] Force re-processing all {num_studies} studies (mode: force_reprocess)')
            if REUSE_TRANSFORMATIONS:
                log.info(f'[INCREMENTAL] Will reuse existing transformations where available')

        else:
            raise ValueError(f"Unknown INCREMENTAL_MODE: {INCREMENTAL_MODE}. Must be one of: skip_all, process_new, smart, force_reprocess")
        
        # Check if there is already a coregistered study in this patient
        coregistered_seg_exists = not patient_df['coregistered_segmentation'].isna().all()

        # Set these variables to None automatically
        patient_template_image = None
        patient_template_seg = None
        patient_template_bet = None
        patient_anchor_image = None
        patient_anchor_seg = None
        patient_anchor_bet = None

        if coregistered_seg_exists:
            try:
                log.info(f'Coregistered segmentation exists for patient {patient}')
                coregistered_segmentations = patient_df[patient_df['coregistered_segmentation'].notna()]
                patient_template_row = select_anchor(coregistered_segmentations, 't2_thin', prefer_coregistered=PREFER_COREGISTERED)
                log.info(f'[TEMPLATE] Using coregistered study as template: {os.path.basename(patient_template_row.get("coregistered_image", "N/A"))}')
                patient_template_image, patient_template_seg = get_ants_image(patient_template_row, prefer_coregistered=PREFER_COREGISTERED, template_bets=template_bets)
                patient_df = patient_df[patient_df['coregistered_segmentation'].isna()]
                
                # Check if there are any remaining studies to process after filtering
                if patient_df.empty:
                    log.info(f'[SKIP] Patient {patient}: All studies already have coregistered segmentations.')
                    return {
                        'patient_id': patient,
                        'updates': updates,
                        'success': True,
                        'skipped': True,
                        'error_message': None,
                        'duration': perf_counter() - patient_start_time
                    }
            except Exception as e:
                error_details = f"[load_patient_template] Failed to load patient template.\n\n{traceback.format_exc()}"
                if patient_template_row is not None:
                    log_error(patient_template_row.name, error_details)
                log.error(f'Failed to load patient template: {e}')
                raise
        else:
            log.info(f'[TEMPLATE] Using MNI T2 template (no prior coregistered studies)')
            patient_template_image = template_bets['t2']
            patient_template_bet = template_bets['t2_bet']
        
        # Template study type: use 'T2' for MNI template, or get from template row if using coregistered
        template_study_type = patient_template_row.get('Study Type', 'T2') if coregistered_seg_exists else 'T2'
        
        # 1) T(Patient_Anchor -> Template) with retry on registration failure
        step1_start = perf_counter()
        log.info("\n" + "="*80)
        log.info(f"[STEP 1] Selecting and Registering Patient Anchor to Template")
        log.info("="*80)
        
        try:
            # Select patient anchor and register with retry on failure
            patient_anchor_row, patient_anchor_image, patient_anchor_seg, patient_anchor_bet, \
            patient_anchor_study_type, tx_pt_anchor_to_template, failed_anchor_indices = \
                select_and_register_patient_anchor_with_retry(
                    patient_df=patient_df,
                    patient_template_image=patient_template_image,
                    patient_template_bet=patient_template_bet,
                    template_study_type=template_study_type,
                    modality='t2_thin',
                    prefer_coregistered=PREFER_COREGISTERED,
                    bet_modality='t2',
                    max_attempts=5,
                    template_bets=template_bets
                )
            
            # Log errors for failed anchor attempts
            for failed_idx, failed_reason in failed_anchor_indices:
                error_details = f"[patient_anchor_registration_retry] Anchor candidate failed, trying next. Reason: {failed_reason}"
                log_error(failed_idx, error_details)
            
            patient_anchor_filename = os.path.basename(patient_anchor_row.get("image path", "N/A"))
            log.info(f'[ANCHOR] Selected patient anchor: {patient_anchor_filename}')
            log.info(f'         File size: {patient_anchor_row.get("file_size_mb", 0):.2f} MB | Timepoint: {patient_anchor_row.get("Timepoint", "N/A")}')
            add_update(patient_anchor_row.name, 'anchor', 'patient_anchor')
            log.info(f'[SUCCESS] Patient anchor registration completed in {perf_counter() - step1_start:.1f}s')
            
        except Exception as e:
            row_index = patient_anchor_row.name if patient_anchor_row is not None else None
            error_details = f"[select_and_register_patient_anchor] Failed after all retry attempts. Error: {type(e).__name__}: {str(e)}. Traceback: {traceback.format_exc()}"
            log_error(row_index, error_details)
            log.error(f'Failed to select and register patient anchor: {e}')
            raise

        # Apply transformation and save
        try:
            warped_patient_anchor, warped_patient_anchor_seg = apply_transitive_transforms(
                transformlist=tx_pt_anchor_to_template,
                fixed=patient_template_image,
                moving=patient_anchor_image,
                seg=patient_anchor_seg
            )

            output_paths = save_coregistered(
                patient_anchor_row,
                transformed_image=warped_patient_anchor,
                transformed_segmentation=warped_patient_anchor_seg
            )
            
            add_update(patient_anchor_row.name, 'coregistered_image', output_paths['image_coregistered_path'])
            add_update(patient_anchor_row.name, 'coregistered_segmentation', output_paths['seg_coregistered_path'])
            add_update(patient_anchor_row.name, 'preprocessing_version', PREPROCESSING_VERSION)
            
            log.info(f'[SAVED] {os.path.basename(output_paths["image_coregistered_path"])}')
            if output_paths['seg_coregistered_path']:
                log.info(f'[SAVED] {os.path.basename(output_paths["seg_coregistered_path"])}')

            # Safety check for patient anchor
            if not TEST_MODE and warped_patient_anchor is not None and patient_template_image is not None:
                try:
                    if validate_warped_image(warped_patient_anchor, "final patient anchor"):
                        mi_after_pt_anchor = ants.image_mutual_information(warped_patient_anchor, patient_template_image)
                        log.info(f'[FINAL MI] Patient anchor in template space: {mi_after_pt_anchor:.6f}')
                        add_update(patient_anchor_row.name, 'mi_after_registration', mi_after_pt_anchor)
                    else:
                        log.warning(f'Skipping MI calculation for patient anchor - invalid warped image')
                        mi_after_pt_anchor = 0.0
                except Exception as e:
                    log.warning(f'Failed to calculate final MI for patient anchor: {e}')
                    mi_after_pt_anchor = 0.0
            else:
                mi_after_pt_anchor = 0.0
            
            template_anchor_filename = os.path.basename(patient_template_row['image path']) if coregistered_seg_exists else 'MNI_template'
            
            overlay_path = generate_overlay_safety_check(
                warped_anchor=patient_template_image,
                warped_curr=warped_patient_anchor,
                anchor_filename=template_anchor_filename,
                curr_filename=os.path.basename(output_paths['image_coregistered_path']),
                mi_value=mi_after_pt_anchor,
                step_title="1) T(Patient_Anchor -> Template)"
            )

            add_update(patient_anchor_row.name, 'coregistered_overlay_plot', str(overlay_path))
            log.info(f'[STEP 1 COMPLETE] Duration: {perf_counter() - step1_start:.1f}s')

            # Cleanup warped patient anchor images (no longer needed after save and overlay)
            del warped_patient_anchor, warped_patient_anchor_seg, output_paths
            gc.collect()
        except Exception as e:
            row_index = patient_anchor_row.name
            error_details = f"[save_patient_anchor] Failed to save coregistered patient anchor. Timepoint: {patient_anchor_row.get('Timepoint')}. Error: {type(e).__name__}: {str(e)}. Traceback: {traceback.format_exc()}"
            log_error(row_index, error_details)
            log.error(f'Failed to save patient anchor: {e}')
            raise

        # 2) T(TP_anchor -> Patient_anchor) for each timepoint
        timepoints = patient_df['Timepoint'].unique()
        num_timepoints_to_process = len(timepoints)
        
        log.info(f'\nProcessing {num_timepoints_to_process} timepoints...')
        
        for tp_idx, timepoint in enumerate(timepoints):
            tp_start_time = perf_counter()
            timepoint_df = patient_df[patient_df['Timepoint'] == timepoint]
            timepoint_anchor_row = None
            num_studies_in_tp = len(timepoint_df)
            
            log.info(f'\n{"~"*60}')
            log.info(f'TIMEPOINT {tp_idx + 1}/{num_timepoints_to_process}: {timepoint} ({num_studies_in_tp} studies)')
            log.info(f'{"~"*60}')
            
            # Initialize variables that may or may not be assigned depending on control flow
            timepoint_anchor_image = None
            timepoint_anchor_seg = None
            timepoint_anchor_bet = None
            timepoint_anchor_study_type = None
            tx_tp_anchor_to_pt_anchor = None
            composed_transforms_tp = None
            warped_tp_anchor = None
            warped_tp_anchor_seg = None
            tp_output_paths = None
            warped_tp_anchor_check = None
            other_studies_df = None
            is_same_as_patient_anchor = False
            
            try:
                # Select and register the timepoint anchor with retry on failure
                timepoint_anchor_row, timepoint_anchor_image, timepoint_anchor_seg, timepoint_anchor_bet, \
                timepoint_anchor_study_type, tx_tp_anchor_to_pt_anchor, failed_tp_anchor_indices, is_same_as_patient_anchor = \
                    select_and_register_tp_anchor_with_retry(
                        timepoint_df=timepoint_df,
                        patient_anchor_image=patient_anchor_image,
                        patient_anchor_seg=patient_anchor_seg,
                        patient_anchor_study_type=patient_anchor_study_type,
                        patient_anchor_row_name=patient_anchor_row.name,
                        modality='t2_thin',
                        prefer_coregistered=PREFER_COREGISTERED,
                        bet_modality='t2',
                        max_attempts=min(5, len(timepoint_df)),
                        template_bets=template_bets
                    )
                
                # Log errors for failed anchor attempts
                for failed_idx, failed_reason in failed_tp_anchor_indices:
                    error_details = f"[tp_anchor_registration_retry] TP anchor candidate failed, trying next. Reason: {failed_reason}"
                    log_error(failed_idx, error_details)
                
                timepoint_anchor_filename = os.path.basename(timepoint_anchor_row.get("image path", "N/A"))
                
                # Handle case where timepoint anchor is same as patient anchor
                if is_same_as_patient_anchor:
                    log.info(f'Timepoint anchor ({timepoint_anchor_filename}) same as patient anchor.')
                    log.info(f'Skipping TP->PT registration, but processing other studies in this timepoint.')
                    
                    # Get other studies in this timepoint (excluding the patient anchor)
                    other_studies_df = timepoint_df[timepoint_df.index != patient_anchor_row.name]
                    num_other_studies = len(other_studies_df)
                    
                    if num_other_studies == 0:
                        log.info(f'No other studies in this timepoint. Timepoint {timepoint} completed in {perf_counter() - tp_start_time:.1f}s')
                        del timepoint_df
                        gc.collect()
                        continue
                    
                    log.info(f'Processing {num_other_studies} other studies directly to patient anchor...')
                    
                    # Process each other study - register directly to patient anchor
                    for study_idx, (idx, other_study_row) in enumerate(other_studies_df.iterrows()):
                        study_start_time = perf_counter()
                        other_study_filename = os.path.basename(other_study_row["image path"])
                        try:
                            log.info("\n" + "-"*80)
                            log.info(f'[STEP 3-DIRECT] Study {study_idx + 1}/{num_other_studies}: {other_study_filename}')
                            log.info(f'                Type: {other_study_row.get("Study Type", "N/A")} | Size: {other_study_row.get("file_size_mb", 0):.2f} MB')
                            log.info(f'                Registering directly to patient anchor (same timepoint)')
                            log.info("-"*80)
                            
                            other_study_image, other_study_seg = get_ants_image(other_study_row, prefer_coregistered=PREFER_COREGISTERED, template_bets=template_bets)
                            other_study_bet_modality = 't1' if 't1' in other_study_row.get('Study Type', '').lower() else 't2'
                            other_study_bet = get_bet_image(other_study_image, other_study_row, modality=other_study_bet_modality)
                            
                            # Get study types for adaptive metric selection
                            other_study_type = other_study_row.get('Study Type', '')
                            
                            # Registration: Study -> Patient Anchor (directly, no TP anchor intermediate)
                            tx_study_to_pt_anchor = register_images('study_to_tp_anchor',
                                fixed=other_study_image,
                                moving=patient_anchor_image,
                                fixed_mask=other_study_bet,
                                moving_mask=patient_anchor_bet,
                                fixed_study_type=other_study_type,
                                moving_study_type=patient_anchor_study_type
                            )
                            
                            # Safety check
                            warped_study_check = None
                            mi_local_study = 0.0

                            if not TEST_MODE:
                                try:
                                    warped_study_check = ants.apply_transforms(
                                        fixed=patient_anchor_image,
                                        moving=other_study_image,
                                        transformlist=tx_study_to_pt_anchor,
                                        whichtoinvert=[True] * len(tx_study_to_pt_anchor)
                                    )
                                    if validate_warped_image(warped_study_check, "study aligned to patient anchor"):
                                        mi_local_study = ants.image_mutual_information(warped_study_check, patient_anchor_image)
                                        log.info(f'[LOCAL MI] Study aligned to patient anchor: {mi_local_study:.6f}')
                                    else:
                                        log.warning(f'Skipping MI calculation - invalid warped study image')
                                except Exception as e:
                                    log.warning(f'Failed to calculate local MI for study: {e}')
                            
                            # Compose transforms: Study -> PT Anchor -> Template
                            composed_transforms_study = tx_pt_anchor_to_template + tx_study_to_pt_anchor
                            
                            # Apply and save
                            warped_study, warped_study_seg = apply_transitive_transforms(
                                transformlist=composed_transforms_study,
                                fixed=patient_template_image,
                                moving=other_study_image,
                                seg=other_study_seg
                            )
                            
                            study_output_paths = save_coregistered(
                                other_study_row,
                                transformed_image=warped_study,
                                transformed_segmentation=warped_study_seg
                            )
                            
                            add_update(idx, 'coregistered_image', study_output_paths['image_coregistered_path'])
                            add_update(idx, 'coregistered_segmentation', study_output_paths['seg_coregistered_path'])
                            add_update(idx, 'preprocessing_version', PREPROCESSING_VERSION)

                            study_output_image_filename = os.path.basename(study_output_paths["image_coregistered_path"])
                            study_output_seg_filename = os.path.basename(study_output_paths["seg_coregistered_path"]) if study_output_paths['seg_coregistered_path'] else None
                            
                            log.info(f'[SAVED] {study_output_image_filename}')
                            if study_output_seg_filename:
                                log.info(f'[SAVED] {study_output_seg_filename}')

                            if not TEST_MODE and warped_study is not None and patient_template_image is not None:
                                try:
                                    if validate_warped_image(warped_study, "final study in template space"):
                                        mi_final_study = ants.image_mutual_information(warped_study, patient_template_image)
                                        log.info(f'[FINAL MI] Study in template space: {mi_final_study:.6f}')
                                        add_update(idx, 'mi_after_registration', mi_final_study)
                                    else:
                                        log.warning(f'Skipping final MI calculation - invalid warped study')
                                except Exception as e:
                                    log.warning(f'Failed to calculate final MI for study: {e}')

                            study_overlay_path = generate_overlay_safety_check(
                                warped_anchor=patient_anchor_image,
                                warped_curr=warped_study_check if not TEST_MODE else warped_study,
                                anchor_filename=os.path.basename(patient_anchor_row['image path']),
                                curr_filename=os.path.basename(study_output_paths['image_coregistered_path']),
                                mi_value=mi_local_study,
                                step_title="3-DIRECT) T(Study -> Patient_Anchor)"
                            )

                            add_update(idx, 'coregistered_overlay_plot', str(study_overlay_path))

                            log.info(f'[SUCCESS] {other_study_filename} processed in {perf_counter() - study_start_time:.1f}s')
                            
                            # Memory cleanup
                            del other_study_image, other_study_seg, other_study_bet, tx_study_to_pt_anchor
                            del warped_study, warped_study_seg, study_output_paths
                            if warped_study_check is not None:
                                del warped_study_check
                            gc.collect()

                        except Exception as e:
                            reg_method = REGISTRATION_CONFIG['study_to_tp_anchor']['method']
                            error_details = f"[register_study_to_patient_anchor_direct] Failed to register study directly to patient anchor using {reg_method}. Timepoint: {timepoint}. Study type: {other_study_row.get('Study Type')}. File size: {other_study_row.get('file_size_mb', 0):.2f} MB.\n\n{traceback.format_exc()}"
                            log_error(idx, error_details)
                            log.error(f"Failed to process {other_study_filename}: {e}")
                            log.error(f"Details logged to output CSV")
                            gc.collect()
                            continue

                    # Timepoint complete
                    log.info(f'\n[TIMEPOINT COMPLETE] {timepoint}: {num_studies_in_tp} studies processed in {perf_counter() - tp_start_time:.1f}s')
                    del timepoint_df, other_studies_df
                    gc.collect()
                    continue
                
                # Non-same-as-patient-anchor case: registration already done in select_and_register_tp_anchor_with_retry
                log.info(f'[TP ANCHOR] {timepoint_anchor_filename}')
                log.info(f'            Size: {timepoint_anchor_row.get("file_size_mb", 0):.2f} MB')
                
                add_update(timepoint_anchor_row.name, 'anchor', 'timepoint_anchor')
                # Note: timepoint_anchor_image, seg, bet, and tx_tp_anchor_to_pt_anchor already set by select_and_register_tp_anchor_with_retry
                
                step2_start = perf_counter()
                
                # Compose transforms
                composed_transforms_tp = tx_pt_anchor_to_template + tx_tp_anchor_to_pt_anchor
                
                # Apply transformation and save
                warped_tp_anchor, warped_tp_anchor_seg = apply_transitive_transforms(
                    transformlist=composed_transforms_tp,
                    fixed=patient_template_image,
                    moving=timepoint_anchor_image,
                    seg=timepoint_anchor_seg
                )
                
                tp_output_paths = save_coregistered(
                    timepoint_anchor_row,
                    transformed_image=warped_tp_anchor,
                    transformed_segmentation=warped_tp_anchor_seg
                )
                
                add_update(timepoint_anchor_row.name, 'coregistered_image', tp_output_paths['image_coregistered_path'])
                add_update(timepoint_anchor_row.name, 'coregistered_segmentation', tp_output_paths['seg_coregistered_path'])
                add_update(timepoint_anchor_row.name, 'preprocessing_version', PREPROCESSING_VERSION)

                tp_output_image_filename = os.path.basename(tp_output_paths["image_coregistered_path"])
                tp_output_seg_filename = os.path.basename(tp_output_paths["seg_coregistered_path"]) if tp_output_paths['seg_coregistered_path'] else None
                
                log.info(f'[SAVED] {tp_output_image_filename}')
                if tp_output_seg_filename:
                    log.info(f'[SAVED] {tp_output_seg_filename}')
                
                # Safety checks
                mi_local_tp = 0.0

                if not TEST_MODE and timepoint_anchor_image is not None and patient_anchor_image is not None:
                    try:
                        warped_tp_anchor_check = ants.apply_transforms(
                            fixed=patient_anchor_image,
                            moving=timepoint_anchor_image,
                            transformlist=tx_tp_anchor_to_pt_anchor,
                            interpolator='lanczosWindowedSinc',
                            whichtoinvert=[True] * len(tx_tp_anchor_to_pt_anchor)
                        )
                        if validate_warped_image(warped_tp_anchor_check, "timepoint anchor aligned to patient anchor"):
                            mi_local_tp = ants.image_mutual_information(warped_tp_anchor_check, patient_anchor_image)
                            log.info(f'[LOCAL MI] Timepoint anchor aligned to patient anchor: {mi_local_tp:.6f}')
                        else:
                            log.warning(f'Skipping local MI calculation - invalid warped timepoint anchor')
                            warped_tp_anchor_check = warped_tp_anchor
                    except Exception as e:
                        log.warning(f'Failed to calculate local MI for timepoint anchor: {e}')
                        warped_tp_anchor_check = warped_tp_anchor

                    if warped_tp_anchor is not None and patient_template_image is not None:
                        try:
                            if validate_warped_image(warped_tp_anchor, "final timepoint anchor in template space"):
                                mi_final_tp = ants.image_mutual_information(warped_tp_anchor, patient_template_image)
                                log.info(f'[FINAL MI] Timepoint anchor in template space: {mi_final_tp:.6f}')
                                add_update(timepoint_anchor_row.name, 'mi_after_registration', mi_final_tp)
                            else:
                                log.warning(f'Skipping final MI calculation - invalid warped timepoint anchor')
                        except Exception as e:
                            log.warning(f'Failed to calculate final MI for timepoint anchor: {e}')
                else:
                    warped_tp_anchor_check = warped_tp_anchor
                
                tp_overlay_path = generate_overlay_safety_check(
                    warped_anchor=patient_anchor_image,
                    warped_curr=warped_tp_anchor_check if not TEST_MODE else warped_tp_anchor,
                    anchor_filename=os.path.basename(patient_anchor_row['image path']),
                    curr_filename=os.path.basename(tp_output_paths['image_coregistered_path']),
                    mi_value=mi_local_tp,
                    step_title="2) T(TP_anchor -> Patient_anchor)"
                )
                
                add_update(timepoint_anchor_row.name, 'coregistered_overlay_plot', str(tp_overlay_path))
                log.info(f'[STEP 2 COMPLETE] TP anchor processed in {perf_counter() - step2_start:.1f}s')

                # Cleanup warped TP anchor images (no longer needed after save and overlay)
                if warped_tp_anchor_check is not None:
                    del warped_tp_anchor_check
                    warped_tp_anchor_check = None
                del warped_tp_anchor, warped_tp_anchor_seg, tp_output_paths
                warped_tp_anchor = None
                warped_tp_anchor_seg = None
                tp_output_paths = None
                gc.collect()

            except Exception as e:
                row_index = timepoint_anchor_row.name if timepoint_anchor_row is not None else None
                error_details = f"[select_and_register_tp_anchor] Failed after all retry attempts. Timepoint: {timepoint}. Error: {type(e).__name__}: {str(e)}. Traceback: {traceback.format_exc()}"
                # Get all study indices in this timepoint as fallback for error logging
                timepoint_study_indices = list(timepoint_df.index)
                log_error(row_index, error_details, fallback_indices=timepoint_study_indices if row_index is None else None)
                log.error(f"Failed to select and register timepoint anchor for timepoint {timepoint}: {e}")
                log.error(f"Details logged to output CSV for {len(timepoint_study_indices)} studies in timepoint")
                gc.collect()
                continue
            
            # 3) T(Study -> TP_anchor) for other studies
            other_studies_df = timepoint_df[timepoint_df.index != timepoint_anchor_row.name]
            num_other_studies = len(other_studies_df)
            
            if num_other_studies > 0:
                log.info(f'\nProcessing {num_other_studies} other studies in timepoint {timepoint}...')
            
            for study_idx, (idx, other_study_row) in enumerate(other_studies_df.iterrows()):
                study_start_time = perf_counter()
                other_study_filename = os.path.basename(other_study_row["image path"])
                try:
                    log.info("\n" + "-"*80)
                    log.info(f'[STEP 3] Study {study_idx + 1}/{num_other_studies}: {other_study_filename}')
                    log.info(f'         Type: {other_study_row.get("Study Type", "N/A")} | Size: {other_study_row.get("file_size_mb", 0):.2f} MB')
                    log.info("-"*80)
                    
                    other_study_image, other_study_seg = get_ants_image(other_study_row, prefer_coregistered=PREFER_COREGISTERED, template_bets=template_bets)
                    other_study_bet_modality = 't1' if 't1' in other_study_row.get('Study Type', '').lower() else 't2'
                    other_study_bet = get_bet_image(other_study_image, other_study_row, modality=other_study_bet_modality)
                    # timepoint_anchor_bet already loaded by select_and_load_anchor_with_fallback
                    
                    # Get study types for adaptive metric selection
                    other_study_type = other_study_row.get('Study Type', '')
                    
                    # Registration
                    tx_study_to_tp_anchor = register_images('study_to_tp_anchor',
                        fixed=other_study_image,
                        moving=timepoint_anchor_image,
                        fixed_mask=other_study_bet,
                        moving_mask=timepoint_anchor_bet,
                        fixed_study_type=other_study_type,
                        moving_study_type=timepoint_anchor_study_type
                    )
                    
                    # Safety check
                    warped_study_check = None
                    mi_local_study = 0.0

                    if not TEST_MODE:
                        try:
                            warped_study_check = ants.apply_transforms(
                                fixed=timepoint_anchor_image,
                                moving=other_study_image,
                                transformlist=tx_study_to_tp_anchor,
                                whichtoinvert=[True] * len(tx_study_to_tp_anchor)
                            )
                            if validate_warped_image(warped_study_check, "study aligned to TP anchor"):
                                mi_local_study = ants.image_mutual_information(warped_study_check, timepoint_anchor_image)
                                log.info(f'[LOCAL MI] Study aligned to TP anchor: {mi_local_study:.6f}')
                            else:
                                log.warning(f'Skipping MI calculation - invalid warped study image')
                        except Exception as e:
                            log.warning(f'Failed to calculate local MI for study: {e}')
                    
                    # Compose transforms
                    composed_transforms_study = tx_pt_anchor_to_template + tx_tp_anchor_to_pt_anchor + tx_study_to_tp_anchor
                    
                    # Apply and save
                    warped_study, warped_study_seg = apply_transitive_transforms(
                        transformlist=composed_transforms_study,
                        fixed=patient_template_image,
                        moving=other_study_image,
                        seg=other_study_seg
                    )
                    
                    study_output_paths = save_coregistered(
                        other_study_row,
                        transformed_image=warped_study,
                        transformed_segmentation=warped_study_seg
                    )
                    
                    add_update(idx, 'coregistered_image', study_output_paths['image_coregistered_path'])
                    add_update(idx, 'coregistered_segmentation', study_output_paths['seg_coregistered_path'])
                    add_update(idx, 'preprocessing_version', PREPROCESSING_VERSION)

                    study_output_image_filename = os.path.basename(study_output_paths["image_coregistered_path"])
                    study_output_seg_filename = os.path.basename(study_output_paths["seg_coregistered_path"]) if study_output_paths['seg_coregistered_path'] else None
                    
                    log.info(f'[SAVED] {study_output_image_filename}')
                    if study_output_seg_filename:
                        log.info(f'[SAVED] {study_output_seg_filename}')

                    if not TEST_MODE and warped_study is not None and patient_template_image is not None:
                        try:
                            if validate_warped_image(warped_study, "final study in template space"):
                                mi_final_study = ants.image_mutual_information(warped_study, patient_template_image)
                                log.info(f'[FINAL MI] Study in template space: {mi_final_study:.6f}')
                                add_update(idx, 'mi_after_registration', mi_final_study)
                            else:
                                log.warning(f'Skipping final MI calculation - invalid warped study')
                        except Exception as e:
                            log.warning(f'Failed to calculate final MI for study: {e}')

                    study_overlay_path = generate_overlay_safety_check(
                        warped_anchor=timepoint_anchor_image,
                        warped_curr=warped_study_check if not TEST_MODE else warped_study,
                        anchor_filename=os.path.basename(timepoint_anchor_row['image path']),
                        curr_filename=os.path.basename(study_output_paths['image_coregistered_path']),
                        mi_value=mi_local_study,
                        step_title="3) T(Study -> TP_anchor)"
                    )
                    
                    add_update(idx, 'coregistered_overlay_plot', str(study_overlay_path))
                    
                    log.info(f'[SUCCESS] {other_study_filename} processed in {perf_counter() - study_start_time:.1f}s')

                    # Memory cleanup
                    del other_study_image, other_study_seg, other_study_bet, tx_study_to_tp_anchor
                    del warped_study, warped_study_seg, study_output_paths
                    if warped_study_check is not None:
                        del warped_study_check
                    gc.collect()

                except Exception as e:
                    reg_method = REGISTRATION_CONFIG['study_to_tp_anchor']['method']
                    error_details = f"[register_study_to_timepoint] Failed to register study to timepoint anchor using {reg_method}. Timepoint: {timepoint}. Study type: {other_study_row.get('Study Type')}. File size: {other_study_row.get('file_size_mb', 0):.2f} MB.\n\n{traceback.format_exc()}"
                    log_error(idx, error_details)
                    log.error(f"Failed to process {other_study_filename}: {e}")
                    log.error(f"Details logged to output CSV")
                    gc.collect()
                    continue
            
            # Print timepoint summary
            tp_duration = perf_counter() - tp_start_time
            log.info(f'\n[TIMEPOINT COMPLETE] {timepoint}: {num_studies_in_tp} studies processed in {tp_duration:.1f}s')
            
            # Memory cleanup for timepoint - only delete variables that were assigned
            if timepoint_anchor_image is not None:
                del timepoint_anchor_image
            if timepoint_anchor_seg is not None:
                del timepoint_anchor_seg
            if timepoint_anchor_bet is not None:
                del timepoint_anchor_bet
            if tx_tp_anchor_to_pt_anchor is not None:
                del tx_tp_anchor_to_pt_anchor
            if composed_transforms_tp is not None:
                del composed_transforms_tp
            if warped_tp_anchor is not None:
                del warped_tp_anchor
            if warped_tp_anchor_seg is not None:
                del warped_tp_anchor_seg
            if tp_output_paths is not None:
                del tp_output_paths
            if timepoint_df is not None:
                del timepoint_df
            if other_studies_df is not None:
                del other_studies_df
            if warped_tp_anchor_check is not None:
                del warped_tp_anchor_check
            gc.collect()
        
        # Print patient summary
        patient_duration = perf_counter() - patient_start_time
        log.info(f'\n{"#"*80}')
        log.info(f'# PATIENT COMPLETE: {patient}')
        log.info(f'# Duration: {patient_duration:.1f}s ({patient_duration/60:.1f} min)')
        log.info(f'# Completed at: {datetime.now().strftime("%H:%M:%S")}')
        log.info(f'{"#"*80}')
        
        # Memory cleanup for patient
        del patient_anchor_image, patient_anchor_seg, patient_anchor_bet
        del tx_pt_anchor_to_template, patient_df
        if coregistered_seg_exists and patient_template_image is not template_bets['t2']:
            del patient_template_image, patient_template_seg
        gc.collect()

        return {
            'patient_id': patient,
            'updates': updates,
            'success': True,
            'skipped': False,
            'error_message': None,
            'duration': perf_counter() - patient_start_time
        }
        
    except Exception as e:
        row_index = patient_anchor_row.name if patient_anchor_row is not None else None
        error_details = f"[patient_level_error] Unhandled error during patient processing.\n\n{traceback.format_exc()}"
        # Use original_patient_indices as fallback to log error to all studies when no specific row
        log_error(row_index, error_details, fallback_indices=original_patient_indices if row_index is None else None)
        log.error(f"\nFailed to process patient {patient}: {e}")
        if row_index is None:
            log.error(f"Details logged to all {len(original_patient_indices)} studies for patient {patient}")
        gc.collect()

        return {
            'patient_id': patient,
            'updates': updates,
            'success': False,
            'skipped': False,
            'error_message': str(e),
            'duration': perf_counter() - patient_start_time
        }


def process_patients(meta_df, template_folder, max_patients=None):
    """
    Main processing loop for patient coregistration with optional parallelization.
    
    Args:
        meta_df: DataFrame containing patient metadata
        template_folder: Path to folder containing MNI template images
                        (passed as path string because ANTs images cannot be pickled)
        max_patients: Maximum number of patients to process (None for all)
        
    Returns:
        DataFrame: Updated metadata with coregistration information
    """
    patient_ids = meta_df['Reworked Patient ID'].unique()
    total_patients = len(patient_ids) if max_patients is None else min(len(patient_ids), max_patients)
    patients_to_process = patient_ids[:total_patients]

    # Check for existing checkpoints and skip already-processed patients
    checkpoint_dir = os.path.join(COREGISTERED_OUTPUT_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    already_processed = set()
    for pid in patients_to_process:
        ckpt_path = os.path.join(checkpoint_dir, f'meta_df_checkpoint_{pid}.csv')
        if os.path.exists(ckpt_path):
            already_processed.add(pid)
    if already_processed:
        log.info(f"Found {len(already_processed)} existing checkpoints, loading and skipping those patients")
        # Load the latest checkpoint (largest file or last written) to restore meta_df state
        for pid in already_processed:
            ckpt_path = os.path.join(checkpoint_dir, f'meta_df_checkpoint_{pid}.csv')
            ckpt_df = pd.read_csv(ckpt_path)
            # Update meta_df rows for this patient from checkpoint
            patient_mask = meta_df['Reworked Patient ID'] == pid
            for col in ckpt_df.columns:
                if col in meta_df.columns:
                    meta_df.loc[patient_mask, col] = ckpt_df[col].values
            log.info(f"Restored checkpoint for patient {pid}")
        patients_to_process = [p for p in patients_to_process if p not in already_processed]
        total_patients = len(patients_to_process)
        log.info(f"Remaining patients to process: {total_patients}")

    log.info(f"\nStarting processing of {total_patients} patients at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Parallel workers: {PARALLEL_WORKERS} | Threads per worker: {THREADS}")
    
    # Prepare work items - pass template_folder path (not template_bets objects)
    # because ANTs images cannot be pickled for multiprocessing
    work_items = []
    for patient_idx, patient in enumerate(patients_to_process):
        patient_df = meta_df[meta_df['Reworked Patient ID'] == patient].copy()
        work_items.append((patient, patient_df, template_folder, patient_idx, total_patients))
    
    # Process patients
    results = []
    success_count = 0
    skip_count = 0
    error_count = 0
    
    if PARALLEL_WORKERS <= 1:
        # Sequential processing (original behavior)
        log.info("Running in sequential mode (PARALLEL_WORKERS=1)")
        for work_item in tqdm(work_items, desc="Processing patients", unit="patient"):
            result = _process_single_patient_worker(work_item)
            results.append(result)
    else:
        # Parallel processing
        log.info(f"Running in parallel mode with {PARALLEL_WORKERS} workers")
        with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS, initializer=_init_worker) as executor:
            # Submit all work
            futures = {executor.submit(_process_single_patient_worker, item): item[0] for item in work_items}
            
            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing patients", unit="patient"):
                patient_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    log.error(f"Worker failed for patient {patient_id}: {e}")
                    results.append({
                        'patient_id': patient_id,
                        'updates': [],
                        'success': False,
                        'skipped': False,
                        'error_message': str(e),
                        'duration': 0
                    })
    
    # Apply all updates to the DataFrame
    log.info("\nApplying updates to DataFrame...")
    for result in results:
        if result['skipped']:
            skip_count += 1
        elif result['success']:
            success_count += 1
        else:
            error_count += 1
            
        for row_idx, column, value in result['updates']:
            if row_idx is not None and row_idx in meta_df.index:
                meta_df.loc[row_idx, column] = value

        # Save checkpoint after each patient's updates are applied
        patient_id = result['patient_id']
        patient_mask = meta_df['Reworked Patient ID'] == patient_id
        patient_checkpoint = meta_df.loc[patient_mask]
        ckpt_path = os.path.join(checkpoint_dir, f'meta_df_checkpoint_{patient_id}.csv')
        patient_checkpoint.to_csv(ckpt_path, index=False)
        log.info(f"Saved checkpoint for patient {patient_id} to {ckpt_path}")

    log.info(f'\n{"="*80}')
    log.info(f'[PROCESSING COMPLETE]')
    log.info(f'Processed: {success_count} patients | Skipped: {skip_count} | Errors: {error_count}')
    log.info(f'Completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log.info(f'{"="*80}')

    return meta_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    log.info("=" * 80)
    log.info("Longitudinal MRI Preprocessing and Coregistration")
    log.info("=" * 80)

    # Print configuration summary
    log.info("\n[CONFIG] Current settings:")
    log.info(f"  - Preprocessing version: {PREPROCESSING_VERSION}")
    log.info(f"  - Incremental mode: {INCREMENTAL_MODE}")
    log.info(f"  - Head masking (background removal): {'ENABLED' if ENABLE_HEAD_MASKING else 'DISABLED'}")
    if ENABLE_HEAD_MASKING:
        log.info(f"    - Dilation radius: {HEAD_MASK_DILATION_MM}mm")
        log.info(f"    - Min coverage threshold: {HEAD_MASK_MIN_COVERAGE}%")
    log.info(f"  - Histogram matching: {'ENABLED' if ENABLE_HISTOGRAM_MATCHING else 'DISABLED'}")
    log.info(f"  - Center alignment:")
    for level, enabled in CENTER_ALIGNMENT_CONFIG.items():
        log.info(f"    - {level}: {'ENABLED' if enabled else 'DISABLED'}")
    log.info(f"  - Target spacing: {TARGET_SPACING}mm isotropic")
    log.info(f"  - Parallel workers: {PARALLEL_WORKERS} x {THREADS} threads")
    log.info(f"  - Test mode: {'ENABLED' if TEST_MODE else 'DISABLED'}")

    # Setup template folder path
    # Note: Templates are loaded within worker processes because ANTs images cannot be pickled
    log.info("\n[STEP 1] Verifying templates...")
    cwd = os.getcwd()
    template_folder = os.path.join(cwd, TEMPLATE_FOLDER)
    
    # Verify templates exist by loading them once (also caches for sequential mode)
    template_bets = load_templates(template_folder)
    log.info(f"Template folder: {template_folder}")
    log.info(f"Available templates: {list(template_bets.keys())}")
    
    # For sequential mode, pre-cache templates in the global variable
    global _worker_template_bets
    _worker_template_bets = template_bets
    
    # Load data
    log.info("\n[STEP 2] Loading data...")
    meta_df = pd.read_csv(INPUT_CSV)
    
    # Filter by allowed study types if configured
    if ALLOWED_STUDY_TYPES is not None:
        meta_df = meta_df[meta_df['Study Type'].isin(ALLOWED_STUDY_TYPES)]
        log.info(f"Filtered to study types in: {ALLOWED_STUDY_TYPES}")
        log.info(f"Remaining rows after filtering: {len(meta_df)}")

    # Add derived columns
    meta_df['Reworked Patient ID'] = meta_df['Patient_MRI_Days Tracker'].astype(str).str.split('_', n=1).str[0]
    meta_df['Timepoint'] = meta_df['Patient_MRI_Days Tracker'].astype(str).str.split('_', n=1).str[1]
    
    # Move the two new columns to the front
    _front = ['Reworked Patient ID', 'Timepoint']
    _cols = [c for c in _front if c in meta_df.columns] + [c for c in meta_df.columns if c not in _front]
    meta_df = meta_df[_cols]
    
    # Initialize anchor tracking and coregistered image columns
    meta_df['anchor'] = None
    
    # Initialize output columns as object type to handle strings and avoid float64 incompatibility errors
    output_cols = ['coregistered_image', 'coregistered_segmentation', 'coregistered_overlay_plot',
                   'preprocessing_version', 'error_details', 'error_timestamp']
    for col in output_cols:
        if col not in meta_df.columns:
            meta_df[col] = None
        meta_df[col] = meta_df[col].astype('object')
    
    log.info(f"Total rows: {len(meta_df)}")
    
    # Process patients - pass template_folder (not template_bets) because ANTs images cannot be pickled
    log.info("\n[STEP 3] Processing patients...")
    meta_df = process_patients(meta_df, template_folder, max_patients=MAX_PATIENTS)
    
    # Save results
    log.info("\n[STEP 4] Saving results...")
    meta_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f'Saved updated metadata to {OUTPUT_CSV}')
    
    # Display summary statistics
    log.info('\n' + "=" * 80)
    log.info('[SUMMARY] Coregistration Results:')
    log.info("=" * 80)
    log.info(f'Total rows: {len(meta_df)}')
    log.info(f'Patient anchors: {(meta_df["anchor"] == "patient_anchor").sum()}')
    log.info(f'Timepoint anchors: {(meta_df["anchor"] == "timepoint_anchor").sum()}')
    log.info(f'Coregistered images: {meta_df["coregistered_image"].notna().sum()}')
    log.info(f'Coregistered segmentations: {meta_df["coregistered_segmentation"].notna().sum()}')
    log.info(f'Errors logged: {meta_df["error_details"].notna().sum()}')
    log.info("=" * 80)


def _apply_config(config):
    """Override module-level config variables from a dict (loaded from YAML)."""
    g = globals()
    config_map = {
        'seed': 'SEED',
        'threads': 'THREADS',
        'parallel_workers': 'PARALLEL_WORKERS',
        'test_mode': 'TEST_MODE',
        'prefer_coregistered': 'PREFER_COREGISTERED',
        'preprocessing_version': 'PREPROCESSING_VERSION',
        'incremental_mode': 'INCREMENTAL_MODE',
        'reuse_transformations': 'REUSE_TRANSFORMATIONS',
        'enable_histogram_matching': 'ENABLE_HISTOGRAM_MATCHING',
        'histogram_matching_bins': 'HISTOGRAM_MATCHING_BINS',
        'histogram_matching_points': 'HISTOGRAM_MATCHING_POINTS',
        'enable_head_masking': 'ENABLE_HEAD_MASKING',
        'head_mask_dilation_mm': 'HEAD_MASK_DILATION_MM',
        'head_mask_min_coverage': 'HEAD_MASK_MIN_COVERAGE',
        'target_spacing': 'TARGET_SPACING',
        'max_patients': 'MAX_PATIENTS',
        'allowed_study_types': 'ALLOWED_STUDY_TYPES',
        'template_z_crop': 'TEMPLATE_Z_CROP',
        'input_csv': 'INPUT_CSV',
        'template_dir': 'TEMPLATE_FOLDER',
        'output_base_dir': 'BASE_DIR',
    }
    for yaml_key, global_name in config_map.items():
        if yaml_key in config and config[yaml_key] is not None:
            g[global_name] = config[yaml_key]

    # Re-derive dependent variables after overrides
    if 'preprocessing_version' in config:
        from datetime import date
        g['EXPERIMENT_SUFFIX'] = f"_{date.today().strftime('%m_%d_%Y')}_V{g['PREPROCESSING_VERSION']}"

    if 'threads' in config:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(g['THREADS'])

    # Handle nested config dicts
    if 'registration_config' in config and config['registration_config'] is not None:
        g['REGISTRATION_CONFIG'] = config['registration_config']
    if 'registration_params' in config and config['registration_params'] is not None:
        g['REGISTRATION_PARAMS'] = config['registration_params']
    if 'center_alignment_config' in config and config['center_alignment_config'] is not None:
        g['CENTER_ALIGNMENT_CONFIG'] = config['center_alignment_config']
    if 'anchor_selection_metric' in config and config['anchor_selection_metric'] is not None:
        g['ANCHOR_SELECTION_METRIC'] = config['anchor_selection_metric']

    # Re-derive output paths if base dir or version changed
    if any(k in config for k in ['output_base_dir', 'preprocessing_version']):
        base = g.get('BASE_DIR', '')
        suffix = g.get('EXPERIMENT_SUFFIX', '')
        g['HD_OUTPUT_DIR'] = os.path.join(base, 'hd_output')
        g['OUTPUT_CSV'] = f'df_coregistered{suffix}.csv'
        g['COREGISTERED_OUTPUT_DIR'] = os.path.join(base, f'df_coregistered{suffix}')
        g['OVERLAY_OUTPUT_DIR'] = os.path.join(g['COREGISTERED_OUTPUT_DIR'], 'overlay_plots')

    # Re-seed if seed changed
    if 'seed' in config:
        import random
        random.seed(g['SEED'])
        np.random.seed(g['SEED'])
        try:
            import tensorflow as tf
            tf.random.set_seed(g['SEED'])
        except ImportError:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MRI preprocessing and coregistration")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides inline defaults)")
    args = parser.parse_args()

    if args.config:
        import sys
        sys.path.insert(0, str(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
        from shared.config_loader import load_config
        from shared.run_logger import init_run

        config = load_config(args.config)
        _apply_config(config)

        run_dir = init_run(
            base_dir=config.get("output_dir", "."),
            project="mri_registration",
            experiment="preprocessing",
            config=config,
        )
        log.info(f"Run directory: {run_dir}")

    main()