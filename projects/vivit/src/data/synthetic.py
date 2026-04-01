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
from typing import Literal
from monai.transforms.utils import rescale_array

def create_synthetic_dataset(
        num_patients: int = 30,
        lam: float = 3.5,
        off_axis_growth_p: float = 0.25,
        geometry_mode: Literal["ellipsoid", "lollipop"] = "ellipsoid",
        canal_axis: Literal["a", "b", "c"] = "c",
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
    
    for i in tqdm(range(num_patients)):
        # Number of samples
        n_samples = np.random.poisson(lam=lam)
        n_samples = n_samples if n_samples > 1 else 2

        # Dates for samples
        dates = [0]
        for _ in range(n_samples - 1):
            dates.append(np.random.randint(50, 400) + dates[-1])

        # Get rotation degrees
        rotation_degrees = [np.random.randint(0, 180) for _ in range(3)]

        # Select growth type
        growth = np.random.choice(growth_types)

        # Get direction of growth
        growth_direction = np.random.choice(growth_directions)

        # Get the images and labels
        images, labels = create_synthetic_time_3d(
                height=394,
                width=466,
                depth=378,
                dates=dates,
                rotation_degrees=rotation_degrees,
                rad_max=40,
                rad_min=5,
                channel_dim=0,
                growth=growth,
                growth_direction=growth_direction,
                off_axis_growth_p=off_axis_growth_p,
                geometry_mode=geometry_mode,
                canal_axis=canal_axis,
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
        channel_dim: int | None = None,
        growth: Literal["decreasing", "fat-tailed", "steady", "stable"] = "decreasing",
        growth_direction: Literal["a", "b", "c"] = "a",
        off_axis_growth_p: float = 0.5,
        geometry_mode: Literal["ellipsoid", "lollipop"] = "ellipsoid",
        canal_axis: Literal["a", "b", "c"] = "c",
        random_state: np.random.RandomState | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
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
        channel_dim : int | None : Dimension to add channel axis (0 for first, -1 or 3 for last, 
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
        random_state : np.random.RandomState | None : Random state for reproducible random 
            number generation, None uses global numpy random state
    
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
    rs: np.random.RandomState = np.random.random.__self__ if random_state is None else random_state
    
    # Create the centroid
    x = rs.randint(rad_max, height - rad_max)
    y = rs.randint(rad_max, width - rad_max)
    z = rs.randint(rad_max, depth - rad_max)
    centroid = (x, y , z)

    # Create the rotation object
    rotation = Rotation.from_euler('zyx', rotation_degrees, degrees=True)

    # Set initial random state
    rad = None

    for i, date in enumerate(dates):
        # Create the current image
        image = np.zeros((height, width, depth), dtype=np.float32)

        # Adjust the radius based on the growth type
        if rad:
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
        else:
            # Set the initial radius to somewhere between the min and 1/3 the maximum
            rad = rs.randint(rad_min, rad_max // 2)

            # Set the initial radii
            a, b, c = rad, rad, rad
        
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
            # 1) Canal-axis alignment in the rotated local frame.
            axis_lengths = {"a": a, "b": b, "c": c}
            if canal_axis == "a":
                canal_coord, perp1_coord, perp2_coord = spx_rotate, spy_rotate, spz_rotate
                perp1_len, perp2_len = b, c
            elif canal_axis == "b":
                canal_coord, perp1_coord, perp2_coord = spy_rotate, spx_rotate, spz_rotate
                perp1_len, perp2_len = a, c
            else:
                canal_coord, perp1_coord, perp2_coord = spz_rotate, spx_rotate, spy_rotate
                perp1_len, perp2_len = a, b
            canal_axis_len = axis_lengths[canal_axis]

            # 2) Origin / early bulb.
            canal_radius = max(1.0, 0.45 * min(perp1_len, perp2_len))
            origin_radius = max(1.0, 0.9 * canal_radius)
            origin_bulb = (perp1_coord ** 2 + perp2_coord ** 2 + canal_coord ** 2) <= (origin_radius ** 2)

            # 3) Intracanalicular cylinder.
            canal_length = max(canal_radius, 0.8 * canal_axis_len)
            intracanalicular_cylinder = (
                (canal_coord >= 0)
                & (canal_coord <= canal_length)
                & ((perp1_coord ** 2 + perp2_coord ** 2) <= (canal_radius ** 2))
            )

            # 4) Extracanalicular CPA mass.
            cpa_center = canal_length
            cpa_axis = max(canal_radius * 1.25, float(canal_axis_len))
            cpa_perp1 = max(canal_radius * 1.15, float(perp1_len))
            cpa_perp2 = max(canal_radius * 1.15, float(perp2_len))
            cpa_mass = (
                ((canal_coord - cpa_center) ** 2) / (cpa_axis ** 2)
                + (perp1_coord ** 2) / (cpa_perp1 ** 2)
                + (perp2_coord ** 2) / (cpa_perp2 ** 2)
            ) <= 1

            # 5) Final union of components.
            geometry_mask = origin_bulb | intracanalicular_cylinder | cpa_mass

        # Fill-in the circle
        image[geometry_mask] = rs.random() * 0.5 + 0.5
    
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
