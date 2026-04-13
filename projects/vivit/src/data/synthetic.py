"""
Synthetic 3D Medical Imaging Dataset Generator

This module provides functionality for creating synthetic 3D medical imaging datasets
with time-series progression. It generates ellipsoidal structures that can grow over time
according to different growth patterns (decreasing, fat-tailed, steady, or stable). The
synthetic data includes both images and corresponding segmentation labels, with support
for rotations and noise injection to simulate realistic medical imaging scenarios.

The primary use case is for testing and validating medical image analysis pipelines,
particularly those dealing with longitudinal studies of growing anatomical structures.
"""

import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from typing import Literal, Optional
from monai.transforms.utils import rescale_array


def _coerce_rng(random_state: Optional[object]) -> np.random.Generator:
    """Return a Generator for deterministic local sampling."""
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        seed = int(random_state.randint(0, 2 ** 32 - 1))
        return np.random.default_rng(seed)
    raise TypeError("`random_state` must be None, np.random.Generator, or np.random.RandomState.")

def create_synthetic_dataset(
        num_patients: int = 30,
        lam: float = 3.5,
        off_axis_growth_p: float = 0.25,
        geometry_mode: Literal["ellipsoid", "lollipop"] = "ellipsoid",
        canal_axis: Literal["a", "b", "c"] = "c",
        seed: Optional[int] = None,
        random_state: Optional[object] = None,
) -> list[dict]:
    """
    Create a synthetic dataset of 3D medical images with temporal progression for multiple patients.
    
    Each patient in the dataset contains a series of 3D images showing an ellipsoidal structure
    that evolves over time according to a randomly selected growth pattern. The structure is
    rotated and positioned randomly within the image volume.
    
    Args:
        num_patients : int : Number of synthetic patients to generate in the dataset
        lam : float : Lambda parameter for Poisson distribution used to determine the number 
            of time points per patient
        off_axis_growth_p : float : Probability (0.0-1.0) that growth will occur in axes 
            perpendicular to the primary growth direction
        geometry_mode : Literal["ellipsoid", "lollipop"] : Geometry generation mode.
            "ellipsoid" preserves legacy behavior.
        canal_axis : Literal["a", "b", "c"] : Internal canal axis used by "lollipop" mode.
    
    Returns:
        data_dicts : list[dict] : List of patient data dictionaries, where each dictionary contains:
            - 'images': list of np.ndarray with shape (1, H, W, D) for each time point
            - 'labels': list of np.ndarray with shape (1, H, W, D) for each time point
            - 'patient_id': str identifier for the patient
            - 'dates': list[int] of time points in days
            - 'modality': str imaging modality (always 'T2')
            - 'growth': str growth pattern type
    """
    # Data
    data_dicts = []

    # Growth types
    growth_types = ["decreasing", "fat-tailed", "steady", "stable"]

    # Growth directions
    growth_directions = ["a", "b", "c"]
    
    rs = _coerce_rng(np.random.default_rng(seed) if random_state is None and seed is not None else random_state)

    for i in tqdm(range(num_patients)):
        # Number of samples
        n_samples = int(rs.poisson(lam=lam))
        n_samples = n_samples if n_samples > 1 else 2

        # Dates for samples
        dates = [0]
        for _ in range(n_samples - 1):
            dates.append(int(rs.integers(50, 400)) + dates[-1])

        # Get rotation degrees
        rotation_degrees = [int(rs.integers(0, 180)) for _ in range(3)]

        # Select growth type
        growth = str(rs.choice(growth_types))

        # Get direction of growth
        growth_direction = str(rs.choice(growth_directions))

        # Get the images and labels
        images, labels = create_synthetic_time_3d(
                height=394,
                width=466,
                depth=378,
                dates=dates,
                rotation_degrees=rotation_degrees,
                rad_max=50,
                rad_min=5,
                channel_dim=0,
                growth=growth,
                growth_direction=growth_direction,
                off_axis_growth_p=off_axis_growth_p,
                geometry_mode=geometry_mode,
                canal_axis=canal_axis,
                random_state=rs,
            )
        
        # Store the data
        sample_data = {
            "images": images,
            "labels": labels,
            "patient_id": f"patient_{i:03d}",
            "dates": dates,
            "modality": "T2",
            "growth": growth
        }
        data_dicts.append(sample_data)
    
    return data_dicts

def create_synthetic_time_3d(
        height: int,
        width: int,
        depth: int,
        dates: list[int],
        rotation_degrees: list[int],
        rad_max: int = 30,
        rad_min: int = 5,
        noise_max: float = 0.1,
        num_seg_classes: int = 5,
        channel_dim: Optional[int] = None,
        growth: Literal["decreasing", "fat-tailed", "steady", "stable"] = "decreasing",
        growth_direction: Literal["a", "b", "c"] = "a",
        off_axis_growth_p: float = 0.5,
        geometry_mode: Literal["ellipsoid", "lollipop"] = "ellipsoid",
        canal_axis: Literal["a", "b", "c"] = "c",
        # ── lollipop explicit geometry parameters (all in voxel units) ────────
        # When provided these override the rad_min-based defaults.  Callers
        # should convert from mm:  param_vox = param_mm / voxel_spacing_mm.
        canal_base_radius_init: Optional[float] = None,   # porus/opening radius at t=0
        canal_apex_radius_init: Optional[float] = None,   # fundus/tip radius at t=0
        canal_length_init: Optional[float] = None,        # IAC canal length at t=0
        bulb_radius_init: Optional[float] = None,         # CPA bulb radius at t=0 (0 = absent)
        canal_base_radius_max: Optional[float] = None,    # max porus radius (final timepoint)
        canal_apex_radius_max: Optional[float] = None,    # max fundus radius (final timepoint)
        canal_length_max_override: Optional[float] = None,  # max canal length
        bulb_radius_max: Optional[float] = None,          # max CPA bulb radius
        # ── placement ─────────────────────────────────────────────────────────
        centered: bool = False,   # True → fix centroid at cube centre (avoids boundary clipping)
        random_state: Optional[object] = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Create a time series of synthetic 3D images containing a growing ellipsoidal structure.
    
    Generates multiple 3D images at different time points showing an ellipsoid that grows
    according to the specified growth pattern. The ellipsoid is randomly positioned, rotated,
    and can grow anisotropically along a primary axis. Noise is added to simulate realistic
    imaging conditions.
    
    Args:
        height : int : Height dimension of the 3D image volume in pixels
        width : int : Width dimension of the 3D image volume in pixels
        depth : int : Depth dimension of the 3D image volume in pixels
        dates : list[int] : List of time points (in days) at which to generate images
        rotation_degrees : list[int] : Rotation angles in degrees for [z, y, x] axes used 
            to orient the ellipsoid
        rad_max : int : Maximum allowed radius for any axis of the ellipsoid in pixels
        rad_min : int : Minimum initial radius for the ellipsoid in pixels
        noise_max : float : Maximum noise level as fraction of num_seg_classes for background noise
        num_seg_classes : int : Number of segmentation classes used for noise scaling
        channel_dim : Optional[int] : Dimension to add channel axis (0 for first, -1 or 3 for last, 
            None for no channel dimension)
        growth : Literal["decreasing", "fat-tailed", "steady", "stable"] : Growth pattern type:
            - "decreasing": Growth rate decreases logarithmically over time
            - "fat-tailed": Sporadic growth bursts with exponential distribution
            - "steady": Constant linear growth over time
            - "stable": No growth, size remains constant
        growth_direction : Literal["a", "b", "c"] : Primary axis along which growth occurs 
            (a, b, or c correspond to the three ellipsoid semi-axes)
        off_axis_growth_p : float : Probability (0.0-1.0) that perpendicular axes also grow 
            when the primary axis grows
        geometry_mode : Literal["ellipsoid", "lollipop"] : Geometry generation mode.
            "ellipsoid" preserves legacy behavior.
        canal_axis : Literal["a", "b", "c"] : Internal canal axis used by "lollipop" mode.
        random_state : Optional[object] : Deterministic RNG source. Accepts
            np.random.Generator, np.random.RandomState, or None.
    
    Returns:
        images : list[np.ndarray] : List of noisy synthetic images, each with shape 
            (height, width, depth) if channel_dim is None, or (1, height, width, depth) 
            if channel_dim=0, or (height, width, depth, 1) if channel_dim=-1 or 3
        labels : list[np.ndarray] : List of binary segmentation masks corresponding to each 
            image, with same shape as images containing 0 for background and 1 for ellipsoid
    """
    # Check if the values are within range
    if 4 * rad_max <= rad_min:
        raise ValueError(f"`rad_min` {rad_min} should be less than 4 * `rad_max` {rad_max}.")
    if rad_min < 1:
        raise ValueError(f"`rad_min` {rad_min} should be no less than 1.")
    min_size = min(height, width, depth)
    if min_size <= 2 * rad_max:
        raise ValueError(f"the minimal size {min_size} of the image should be larger than `2 * rad_max` 2 * {rad_max}.")
    if geometry_mode not in ("ellipsoid", "lollipop"):
        raise ValueError(f"`geometry_mode` {geometry_mode} should be one of `ellipsoid`, `lollipop`.")
    if canal_axis not in ("a", "b", "c"):
        raise ValueError(f"`canal_axis` {canal_axis} should be one of `a`, `b`, `c`.")
    
    # Create the list of images and labels
    images = []
    labels = []

    # Set random state
    rs = _coerce_rng(random_state)
    
    # Create the centroid
    if centered:
        x, y, z = height // 2, width // 2, depth // 2
    else:
        x = int(rs.integers(rad_max, height - rad_max))
        y = int(rs.integers(rad_max, width - rad_max))
        z = int(rs.integers(rad_max, depth - rad_max))
    # Create the rotation object
    rotation = Rotation.from_euler('zyx', rotation_degrees, degrees=True)

    # Set initial random state (lollipop-specific vars typed Optional for clarity)
    rad = None
    cyl_length: Optional[float] = None
    base_radius: Optional[float] = None   # porus / opening radius
    apex_radius: Optional[float] = None   # fundus / tip radius
    cpa_radius: Optional[float] = None    # extrameatal CPA bulb radius
    canal_length_max: Optional[float] = None
    base_radius_max: Optional[float] = None
    apex_radius_max: Optional[float] = None
    cpa_radius_max: Optional[float] = None
    cpa_lob_amp: Optional[float] = None
    cpa_lob_phase_1: Optional[float] = None
    cpa_lob_phase_2: Optional[float] = None
    cpa_bias_1: Optional[float] = None
    cpa_bias_2: Optional[float] = None
    cpa_tex_phase_1: Optional[float] = None
    cpa_tex_phase_2: Optional[float] = None
    cpa_tex_phase_3: Optional[float] = None
    canal_tex_phase_1: Optional[float] = None
    canal_tex_phase_2: Optional[float] = None
    canal_tex_phase_3: Optional[float] = None

    for i, date in enumerate(dates):
        # Create the current image
        image = np.zeros((height, width, depth), dtype=np.float32)

        # Adjust the radius based on the growth type
        if rad is not None:
            if growth == "decreasing":
                # The growth decreases via the logarithm between the ratio of the current date and the max
                sampled_growth = np.ceil(-np.log(date / max(dates)))
            elif growth == "fat-tailed":
                # Draw n = 1 + (current_date - last_date) // 30 fat-tailed exponential samples
                n_samples = 1 + (date - dates[i - 1]) // 30
                sampled_growth = sum([max(0, int(rs.exponential(4.0) - 4.0)) for _ in range(n_samples)])
            elif growth == "steady":
                # Steadly grow each month by 1
                n_samples = (date - dates[i - 1]) // 30
                sampled_growth = n_samples
            elif growth == "stable":
                sampled_growth = 0
            else:
                raise ValueError(f"`growth` {growth} should be one of `decreasing`, `fat-tailed`, `steady`, `stable`.")
            
            # Update the radii by the growth direction, where the direction receives full growth
            if growth_direction == "a":
                a = a + sampled_growth if a + sampled_growth < rad_max else a
                if rs.uniform() < off_axis_growth_p:
                    b = b + sampled_growth if b + sampled_growth < rad_max else b 
                    c = c + sampled_growth if c + sampled_growth < rad_max else c
            elif growth_direction == "b":
                b = b + sampled_growth if b + sampled_growth < rad_max else b
                if rs.uniform() < off_axis_growth_p:
                    a = a + sampled_growth if a + sampled_growth < rad_max else a 
                    c = c + sampled_growth if c + sampled_growth < rad_max else c
            elif growth_direction == "c":
                c = c + sampled_growth if c + sampled_growth < rad_max else c
                if rs.uniform() < off_axis_growth_p:
                    a = a + sampled_growth if a + sampled_growth < rad_max else a
                    b = b + sampled_growth if b + sampled_growth < rad_max else b 
            else:
                raise ValueError(f"`growth_direction` {growth_direction} should be one of `a`, `b`, `c`.")

            # Lollipop-only blended growth routing.
            if geometry_mode == "lollipop":
                sg = float(sampled_growth) * 1.75
                # Canal always grows toward max (caps naturally).
                cyl_length  = min(cyl_length  + sg,        canal_length_max)
                base_radius = min(base_radius + sg * 0.35, base_radius_max)
                apex_radius = min(apex_radius + sg * 0.25, apex_radius_max)
                # CPA growth accelerates over time so late timepoints form a clear bulb.
                t_idx = i
                n_times = len(dates)
                time_frac = t_idx / max(1, n_times - 1)
                cpa_weight = 0.5 + 3.0 * time_frac
                cpa_radius  = min(cpa_radius + sg * cpa_weight, cpa_radius_max)
        else:
            # Set the initial radius to somewhere between the min and 1/3 the maximum
            rad = int(rs.integers(rad_min, rad_max // 2))

            # Set the initial radii
            a, b, c = rad, rad, rad

            # Initialize lollipop phase state for the first timepoint.
            if geometry_mode == "lollipop":
                # Max dimensions — caller-supplied (mm-derived) take priority over rad_min defaults.
                canal_length_max = canal_length_max_override if canal_length_max_override is not None else float(rad_min)
                base_radius_max  = canal_base_radius_max     if canal_base_radius_max     is not None else max(1.0, float(rad_min) * 0.5)
                apex_radius_max  = canal_apex_radius_max     if canal_apex_radius_max     is not None else max(1.0, float(rad_min) * 0.35)
                cpa_radius_max   = bulb_radius_max           if bulb_radius_max           is not None else float(rad_max)
                # Initial dimensions — tiny tube at t=0 (¼ of max, minimum 1 voxel).
                cyl_length  = canal_length_init      if canal_length_init      is not None else 2.0
                base_radius = canal_base_radius_init if canal_base_radius_init is not None else max(1.0, base_radius_max * 0.25)
                apex_radius = canal_apex_radius_init if canal_apex_radius_init is not None else max(1.0, apex_radius_max * 0.25)
                cpa_radius  = bulb_radius_init       if bulb_radius_init       is not None else max(base_radius * 0.4, 2.0)
                # Mild deterministic realism terms are sampled once per synthetic series.
                cpa_lob_amp = rs.uniform(0.06, 0.12)
                cpa_lob_phase_1 = rs.uniform(0.0, 2.0 * np.pi)
                cpa_lob_phase_2 = rs.uniform(0.0, 2.0 * np.pi)
                cpa_bias_1 = rs.uniform(-0.16, 0.16)
                cpa_bias_2 = rs.uniform(-0.12, 0.12)
                cpa_tex_phase_1 = rs.uniform(0.0, 2.0 * np.pi)
                cpa_tex_phase_2 = rs.uniform(0.0, 2.0 * np.pi)
                cpa_tex_phase_3 = rs.uniform(0.0, 2.0 * np.pi)
                canal_tex_phase_1 = rs.uniform(0.0, 2.0 * np.pi)
                canal_tex_phase_2 = rs.uniform(0.0, 2.0 * np.pi)
                canal_tex_phase_3 = rs.uniform(0.0, 2.0 * np.pi)
        
        # Create the matrix and rotate it
        spy, spx, spz = np.ogrid[-x : height - x, -y : width - y, -z : depth - z]

        rot_matrix_inv = rotation.as_matrix().T

        spy_rotate = rot_matrix_inv[0, 0] * spy + rot_matrix_inv[0, 1] * spx + rot_matrix_inv[0, 2] * spz
        spx_rotate = rot_matrix_inv[1, 0] * spy + rot_matrix_inv[1, 1] * spx + rot_matrix_inv[1, 2] * spz
        spz_rotate = rot_matrix_inv[2, 0] * spy + rot_matrix_inv[2, 1] * spx + rot_matrix_inv[2, 2] * spz

        if geometry_mode == "ellipsoid":
            # Legacy geometry path (unchanged): single rotated ellipsoid.
            geometry_mask = ((spx_rotate ** 2) / (a ** 2)  + (spy_rotate ** 2) / (b ** 2) + (spz_rotate ** 2) / (c ** 2)) <= 1
        else:
            # Anatomically-guided lollipop geometry for vestibular schwannoma.
            #
            # Coordinate orientation (canal_axis selects the IAC direction):
            #   x_rel > 0  →  intracanalicular (porus → fundus, lateral)
            #   x_rel ≤ 0  →  extrameatal / CPA side (medial)
            #
            # Four shape components (unioned):
            #   1. Porus hemisphere  — rounded opening at x_rel=0, radius = base_radius
            #   2. Tapered canal     — linearly tapers base_radius (porus) → apex_radius
            #                         (fundus) over x_rel ∈ [0, cyl_length]
            #   3. Fundus hemisphere — rounded cap at x_rel=cyl_length, radius = apex_radius
            #   4. CPA oblate ovoid  — centred at x_rel = -cpa_radius*0.55 (CPA space),
            #                         equatorial radius = cpa_radius > canal radius
            #
            # Physical dimensions are encoded via mm-to-voxel conversion in the caller.

            # Canal-axis selection in the rotated local frame.
            if canal_axis == "a":
                canal_coord, perp1_coord, perp2_coord = spx_rotate, spy_rotate, spz_rotate
            elif canal_axis == "b":
                canal_coord, perp1_coord, perp2_coord = spy_rotate, spx_rotate, spz_rotate
            else:
                canal_coord, perp1_coord, perp2_coord = spz_rotate, spx_rotate, spy_rotate
            x_rel = -canal_coord   # positive toward fundus/lateral
            rho = np.sqrt(perp1_coord ** 2 + perp2_coord ** 2)

            br = float(base_radius)   # porus (opening) radius in voxels
            ar = float(apex_radius)   # fundus (tip) radius in voxels
            cl = float(cyl_length)    # IAC canal length in voxels
            cr = float(cpa_radius)    # CPA bulb radius in voxels

            # 1. Porus hemisphere: smooth rounded opening at x_rel ∈ [-br, 0].
            r_porus = np.sqrt(np.maximum(0.0, br ** 2 - x_rel ** 2))
            porus_mask = (x_rel >= -br) & (x_rel <= 0.0) & (rho <= r_porus)

            # 2. Tapered canal body: convex taper from br (porus) to ar (fundus).
            t_canal = np.clip(x_rel / cl, 0.0, 1.0) if cl > 0.0 else np.zeros_like(x_rel)
            r_canal = ar + (br - ar) * (1.0 - t_canal) ** 1.5
            canal_mask = (x_rel >= 0.0) & (x_rel <= cl) & (rho <= r_canal)

            # 3. Fundus hemisphere: smooth rounded cap at x_rel ∈ (cl, cl+ar].
            r_fundus = np.sqrt(np.maximum(0.0, ar ** 2 - (x_rel - cl) ** 2))
            fundus_mask = (x_rel > cl) & (x_rel <= cl + ar) & (rho <= r_fundus)
            canal_like_mask = porus_mask | canal_mask | fundus_mask

            # 4. CPA extracanalicular bulb — oblate ovoid centred in CPA space.
            #    Centre is shifted beyond the porus and constrained to the CPA side.
            #    Equatorial radius dominates axial radius, producing an oblate head.
            if cr > 0.0:
                cpa_ctr  = max(br * 1.2, cr * 0.75)       # offset beyond porus into CPA space
                cpa_ax_r = min(br * 0.7, cr * 0.45)       # axial — oblate / disc-like
                cpa_eq_r = cr                              # equatorial — full head width
                # Smooth boundary realism: low-frequency angular lobulation plus a
                # subtle fixed off-axis bias so the bulb is not perfectly symmetric.
                theta = np.arctan2(perp2_coord, perp1_coord)
                lobulation = 1.0 + cpa_lob_amp * (
                    0.65 * np.sin(2.0 * theta + cpa_lob_phase_1) +
                    0.35 * np.sin(3.0 * theta + cpa_lob_phase_2)
                )
                cpa_eq_r_mod = np.maximum(cpa_eq_r * lobulation, cpa_eq_r * 0.82)
                cpa_perp1_ctr = cpa_bias_1 * cpa_eq_r
                cpa_perp2_ctr = cpa_bias_2 * cpa_eq_r
                cpa_rho = np.sqrt((perp1_coord - cpa_perp1_ctr) ** 2 + (perp2_coord - cpa_perp2_ctr) ** 2)
                cpa_dist = ((x_rel + cpa_ctr) / cpa_ax_r) ** 2 + (cpa_rho / cpa_eq_r_mod) ** 2
                cpa_mask = (cpa_dist <= 1.0) & (x_rel <= 0.0)
                geometry_mask = canal_like_mask | cpa_mask
            else:
                cpa_mask = np.zeros_like(canal_like_mask, dtype=bool)
                geometry_mask = canal_like_mask

            # Smooth compartment-aware intensity realism.
            if np.any(canal_like_mask):
                canal_tex = (
                    0.50 * np.sin(0.30 * x_rel + canal_tex_phase_1) +
                    0.30 * np.sin(0.22 * perp1_coord + canal_tex_phase_2) +
                    0.20 * np.sin(0.18 * (perp1_coord + perp2_coord) + canal_tex_phase_3)
                )
                canal_tex /= 1.0
                canal_depth = np.clip(t_canal, 0.0, 1.0)
                canal_signal = 0.56 + 0.05 * canal_tex - 0.03 * canal_depth
                image[canal_like_mask] = np.clip(canal_signal[canal_like_mask], 0.42, 0.72)

            if np.any(cpa_mask):
                cpa_tex = (
                    0.45 * np.sin(0.24 * (perp1_coord - cpa_perp1_ctr) + cpa_tex_phase_1) +
                    0.35 * np.sin(0.20 * (perp2_coord - cpa_perp2_ctr) + cpa_tex_phase_2) +
                    0.20 * np.sin(0.15 * (x_rel + cpa_ctr) + cpa_tex_phase_3)
                )
                cpa_tex /= 1.0
                cpa_core = np.clip(1.0 - cpa_dist, 0.0, 1.0)
                cpa_signal = 0.68 + 0.10 * cpa_tex + 0.08 * cpa_core
                image[cpa_mask] = np.clip(cpa_signal[cpa_mask], 0.50, 0.92)
    
        # Get the label mask
        label = np.ceil(image).astype(np.int32, copy=False)

        # Norm and create noisy image
        norm = rs.uniform(0, num_seg_classes * noise_max, size=image.shape).astype(np.float32)
        noisy_image = rescale_array(np.maximum(image, norm))

        # Apply channels
        if channel_dim is not None:
            if not (isinstance(channel_dim, int) and channel_dim in (-1, 0, 3)):
                raise AssertionError("invalid channel dim.")
            noisy_image, label = (
                (noisy_image[None], label[None]) if channel_dim == 0 else (noisy_image[..., None], label[..., None])
            )
        
        images.append(noisy_image)
        labels.append(label)
    
    return images, labels
