"""
Grid Search with K-Fold Cross-Validation for Medical Image Segmentation

This module provides functionality for hyperparameter tuning of deep learning models
used in medical image segmentation tasks. It implements k-fold cross-validation
combined with grid search to systematically explore combinations of model parameters,
training parameters, and batch sizes.

The module leverages MONAI for medical imaging data handling and PyTorch for model
training. It supports distributed training through Hugging Face Accelerate and
includes custom data samplers and transforms for handling variable-length sequences
of medical images.

Key Features:
    - K-fold cross-validation with configurable number of splits
    - Grid search over model parameters, training parameters, and batch sizes
    - Automatic metric aggregation across folds (mean and standard deviation)
    - Memory management with garbage collection and CUDA cache clearing
    - Results export to CSV for analysis

Typical Usage:
    The main entry point is `grid_search()`, which orchestrates the hyperparameter
    search. It internally calls `kfold_cross_val()` for each parameter combination
    to evaluate model performance through cross-validation.
"""

import gc
import torch
import numpy as np
import pandas as pd
from accelerate import Accelerator
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from sklearn.model_selection import KFold
from typing import Any
from itertools import product
from src.data.transforms import build_train_transform, build_eval_transform
from src.data.samplers import LengthShuffledBucketBatchSampler
from src.data.utils import pad_sequence_collate_fn, TransformSequence
from src.train.train_ops import train

def kfold_cross_val(
        model: torch.nn.Module,
        dataset: list[dict[str, Any]],
        accelerator: Accelerator,
        batch_size: int,
        model_params: dict[str, Any],
        train_params: dict[str, Any],
        target_spacing: list[int, int, int],
        img_size: list[int, int, int],
        pos_neg: list[int, int],
        n_splits: int = 3,
        random_state: int = 42
) -> dict[str, float]:
    """
    Performs k-fold cross-validation on a medical image segmentation model.
    
    Splits the dataset into k folds, trains the model on k-1 folds, and validates
    on the remaining fold. This process is repeated k times with each fold serving
    as the validation set once. Results are aggregated across all folds to provide
    mean and standard deviation statistics for all metrics.
    
    Args:
        model : torch.nn.Module : The PyTorch model class (not instantiated) to be trained
        dataset : list[dict[str, Any]] : List of data dictionaries, each containing 'images', 
            'labels', and 'dates' keys. Length: [N_samples]
        accelerator : Accelerator : Hugging Face Accelerate object for distributed training
        batch_size : int : Number of samples per batch
        model_params : dict[str, Any] : Dictionary of parameters to instantiate the model
        train_params : dict[str, Any] : Dictionary of training hyperparameters (e.g., learning rate, 
            epochs, optimizer settings)
        target_spacing : list[int, int, int] : Target voxel spacing in mm for resampling medical 
            images [x, y, z]
        img_size : list[int, int, int] : Target spatial dimensions for cropping/padding images 
            [height, width, depth]
        pos_neg : list[int, int] : Ratio of positive to negative samples for sampling [pos, neg]
        n_splits : int : Number of folds for cross-validation (default: 3)
        random_state : int : Random seed for reproducibility (default: 42)
    
    Returns:
        kf_out : dict[str, str] : Dictionary containing aggregated metrics across folds. Each key
            is a metric name, and values are formatted strings with mean and standard deviation
            (e.g., "0.8534 +/- 0.0234"). Includes metrics: dice mean, rvd mean, hausdorff mean,
            surface dice mean, precision mean, recall mean, train_loss, and best_score.
    """
    # Reset the determinism
    set_determinism(random_state)

    # Create folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Create metrics
    kf_metrics = {}
    metric_names = []

    for fold, (train_idxs, val_idxs) in enumerate(kf.split(dataset)):
        print(f"fold {fold + 1}")
        print("----------------")

        # Define the datasets
        train_data = [dataset[i] for i in train_idxs]
        val_data = [dataset[i] for i in val_idxs]

        # Get the transforms
        train_transform = build_train_transform(
            target_spacing, roi_size=img_size, pos_neg=pos_neg
        )
        val_transform = build_eval_transform(
            target_spacing, img_size
        )
        
        # Build the dataset
        transform = TransformSequence(
            keys=["images", "labels", "dates"], spatial_transforms=train_transform
        )
        train_subset = Dataset(data=train_data, transform=transform)
        transform = TransformSequence(
            keys=["images", "labels", "dates"], spatial_transforms=val_transform
        )
        val_subset = Dataset(data=val_data, transform=transform)

        print(f"train subset size: {len(train_subset)}, validiation subset size: {len(val_subset)}")

        # Build the samplers    
        train_sampler = LengthShuffledBucketBatchSampler(
            train_subset,
            batch_size,
            seed=random_state
        )
        val_sampler = LengthShuffledBucketBatchSampler(
            val_subset,
            batch_size,
            seed=random_state
        )

        # Build the data loaders
        train_loader = DataLoader(
            train_subset,
            batch_sampler=train_sampler,
            collate_fn=pad_sequence_collate_fn
        )
        val_loader = DataLoader(
            val_subset,
            batch_sampler=val_sampler,
            collate_fn=pad_sequence_collate_fn
        )

        # Build the model
        k_model = model(**model_params)

        # Add the correct batch size
        train_params = train_params.copy()
        train_params["batch_size"] = batch_size

        # Train the model
        k_model, hist = train(
            k_model, 
            train_loader, 
            val_loader, 
            accelerator,
            **train_params
        )

        # Get the metrics
        kf_metrics[f"fold_{fold+1}"] = {
            "train_loss": hist["epoch_losses"][-1],
            "best_metrics": hist["best_metrics"],
            "best_score": hist["best_score"]
        }

        if not metric_names:
            metric_names = [ metric for metric in hist["best_metrics"].keys() ]

        # Clear out everything
        del k_model, train_loader, val_loader, train_data, train_subset, val_data, val_subset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Compute the averages and std across folds
    kf_out = {
        f"{metric}": f"{np.mean(
            [ kf_metrics[fold]["best_metrics"][metric] for fold in kf_metrics.keys() ]
        ):.4f} +/- {np.std(
                [ kf_metrics[fold]["best_metrics"][metric] for fold in kf_metrics.keys() ]
        ):.4f}"
        for metric in metric_names
    }

    # Compute the training loss
    kf_out["train_loss"] = f"{np.mean(
        [ kf_metrics[fold]["train_loss"] for fold in kf_metrics.keys() ]
    )} +/- {np.std(
        [ kf_metrics[fold]["train_loss"] for fold in kf_metrics.keys() ]
    ):.4f}"

    # Compute the best score
    kf_out["best_score"] = f"{np.mean(
        [ kf_metrics[fold]["best_score"] for fold in kf_metrics.keys() ]
    )} +/- {np.std(
        [ kf_metrics[fold]["best_score"] for fold in kf_metrics.keys() ]
    )}"

    return kf_out

def grid_search(
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        model_params: dict[str, list[Any]],
        train_params: dict[str, list[Any]],
        batch_sizes: list[int],
        n_splits: int,
        target_spacing: list[int, int, int],
        img_size: list[int, int, int],
        pos_neg: list[int, int],
        accelerator: Accelerator | None = None,
        file_path: str | None = None
) -> pd.DataFrame:
    """
    Performs exhaustive grid search over hyperparameters using k-fold cross-validation.
    
    Explores all possible combinations of model parameters, training parameters, and batch
    sizes by training and evaluating the model with each configuration. For each combination,
    k-fold cross-validation is performed to obtain robust performance estimates. Results
    are compiled into a pandas DataFrame and optionally saved to CSV.
    
    Args:
        model : torch.nn.Module : The PyTorch model class (not instantiated) to be trained
        dataset : torch.utils.data.Dataset : Complete dataset containing all training samples
        model_params : dict[str, list[Any]] : Dictionary where keys are model parameter names
            and values are lists of parameter values to search over
        train_params : dict[str, list[Any]] : Dictionary where keys are training parameter names
            (e.g., 'learning_rate', 'num_epochs') and values are lists of values to search over
        batch_sizes : list[int] : List of batch sizes to evaluate
        n_splits : int : Number of folds for cross-validation in each grid search iteration
        target_spacing : list[int, int, int] : Target voxel spacing in mm for resampling [x, y, z]
        img_size : list[int, int, int] : Target spatial dimensions for images [height, width, depth]
        pos_neg : list[int, int] : Ratio of positive to negative samples [pos, neg]
        accelerator : Accelerator | None : Hugging Face Accelerate object for distributed training
            (default: None)
        file_path : str | None : Path to save the results CSV file (default: None, no file saved)
    
    Returns:
        df : pd.DataFrame : DataFrame containing grid search results with shape 
            [N_combinations, N_columns] where N_combinations is the total number of parameter
            combinations tested and N_columns includes all parameter columns plus metrics.
            Columns include: model parameters, batch size, target spacing, img size, pos neg,
            training parameters, train loss, dice mean, rvd mean, hausdorff mean, 
            surface dice mean, precision mean, recall mean, and best score.
    """
    # Create the model dictionaries
    model_products = list(product(*list(model_params.values())))
    grid_model_params = [
        { name: prod[j] for j, name in enumerate(model_params.keys()) } 
        for prod in model_products
    ]

    # Create the train dictionaries
    train_products = list(product(*list(train_params.values())))
    grid_train_params = [
        { name: prod[j] for j, name in enumerate(train_params.keys()) }
        for prod in train_products
    ]

    # Create data frame for storage
    metrics = [
        "dice mean", 
        "rvd mean", 
        "hausdorff mean", 
        "surface dice mean", 
        "precision mean", 
        "recall mean"
    ]
    columns = (
        list(model_params.keys()) + 
        ["batch size", "target spacing", "img size", "pos neg"] + 
        list(train_params.keys()) + 
        ["train loss"] +
        metrics + 
        ["best score"]
    )
    
    df = pd.DataFrame(columns=columns)

    # Loop through the grid search
    print("Conducting grid search.")

    for i, model_param in enumerate(grid_model_params):
        for j, batch_size in enumerate(batch_sizes):
            for k, train_param in enumerate(grid_train_params):
                indx = (
                    i * len(batch_sizes) * len(grid_train_params) + 
                    j * len(grid_train_params) +
                    k + 1
                )
                
                print(f"Testing grid item {indx}/{len(grid_model_params) * len(batch_sizes) * len(grid_train_params)}")
                print("-------------------------------")

                # Cross validation
                kf_out = kfold_cross_val(
                    model=model,
                    dataset=dataset,
                    accelerator=accelerator,
                    batch_size=batch_size,
                    model_params=model_param,
                    train_params=train_param,
                    n_splits=n_splits,
                    target_spacing=target_spacing,
                    img_size=img_size,
                    pos_neg=pos_neg
                )

                # Store the results
                results = (
                    list(model_param.values()) + 
                    [batch_size, target_spacing, img_size, pos_neg] +
                    list(train_param.values()) + 
                    [kf_out["train_loss"]] + 
                    [ kf_out[metric] for metric in metrics ] + 
                    [kf_out["best_score"]]
                )

                df.loc[len(df)] = results
    
    # Save the df
    df.to_csv(file_path)

    return df