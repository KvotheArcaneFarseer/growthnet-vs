"""
train_ops.py

This module provides core training and validation operations for autoregressive 
segmentation models on variable-length sequences of 3D MRI brain scans.

The module implements an autoregressive training approach where the model learns to 
predict the next segmentation mask in a sequence given all previous images and masks. 
Training uses mixed precision via Accelerate and supports multiple loss functions 
including Dice, DiceCE, DiceFocal, Tversky, and TverskyCE.

Key Features:
    - Autoregressive training on variable-length sequences
    - Mixed precision training with Accelerate
    - Multiple loss function options
    - Comprehensive validation metrics (Dice, Hausdorff, Surface Dice, etc.)
    - Early stopping with patience
    - Learning rate scheduling (cosine, warmup, one-cycle)
    - Memory-efficient processing through gradient accumulation

Main Functions:
    - train: Main training loop with validation
    - train_step: Single training step with autoregressive forward pass
    - validation: Full validation loop over validation dataset
    - validation_step: Single validation batch with autoregressive inference
"""

import torch
import monai
import numpy as np
import gc
import logging
import os
from typing import Any, Literal, Optional
from accelerate import Accelerator
from monai.data.utils import decollate_batch
from monai.losses import (
    DiceLoss,
    DiceCELoss,
    DiceFocalLoss,
    TverskyLoss
)
from monai.metrics import (
    DiceMetric, 
    HausdorffDistanceMetric,
    SurfaceDiceMetric,
    ConfusionMatrixMetric
)
from src.train.utils import (
    get_device, 
    get_parameter_groups,
    TverskyCELoss,
    RelativeVolumdeDifferenceMetric,
    EarlyStopper,
    WeightedValidationMetrics
)
from src.inference.default_inference import inference
from src.data.utils import get_post_transform

def train(
        model: torch.nn.Module,
        train_loader: monai.data.DataLoader,
        val_loader: monai.data.DataLoader,
        accelerator: Optional[Accelerator] = None,
        max_epochs: int = 5,
        loss_function: Literal["dice", "dicece", "dicefocal", "tversky", "tverskyce"] = "dicece",
        initial_lr: float = 1e-4,
        scheduler: Literal["cos", "warmup", "onecycle"] = "onecycle",
        val_interval: int = 1,
        batch_size: int = 4,
        lam: float = 1.0,
        use_sliding_window: bool = False,
        include_background: bool = False,
        patience: int = 3,
        log_file: Optional[str] = None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Train a segmentation model using an autoregressive approach on variable-length 
    sequences of 3D MRI brain scans.
    
    This function implements the main training loop where the model learns to predict
    the next segmentation mask in a sequence given all previous images and masks. 
    Training uses mixed precision via Accelerate, validates at specified intervals, 
    and supports early stopping based on validation metrics.
    
    Args:
        model : torch.nn.Module : PyTorch segmentation model to train
        train_loader : monai.data.DataLoader : DataLoader containing training batches with 
            images, labels, sequence lengths, and dates
        val_loader : monai.data.DataLoader : DataLoader containing validation batches with 
            same structure as train_loader
        accelerator : Accelerator | None : Accelerate instance for distributed training and 
            mixed precision, creates new instance if None (default: None)
        max_epochs : int : Maximum number of training epochs (default: 5)
        loss_function : Literal["dice", "dicece", "dicefocal", "tversky", "tverskyce"] : 
            Loss function to use for training (default: "dicece")
        initial_lr : float : Initial learning rate for AdamW optimizer (default: 1e-4)
        scheduler : Literal["cos", "warmup", "onecycle"] : Learning rate scheduler type 
            (default: "onecycle")
        val_interval : int : Perform validation every N epochs (default: 1)
        batch_size : int : Batch size for learning rate scaling (default: 4)
        lam : float : Weight for CE/Focal component in combined loss functions (default: 1.0)
        use_sliding_window : bool : Whether to use sliding window inference during 
            validation (default: False)
        include_background : bool : Whether to include background class in loss and 
            metrics computation (default: False)
        patience : int : Number of epochs with no improvement before early stopping 
            (default: 3)
        log_file : Optional[str] : Path to save log file, console only if None (default: None)
    
    Returns:
        model : torch.nn.Module : Trained model unwrapped from accelerator
        output_dict : dict[str, Any] : Dictionary containing training results with keys:
            - "epoch_losses": List[float] of mean training losses per epoch
            - "validation_metrics": List[dict[str, float]] of validation metrics per interval
            - "best_metrics": dict[str, float] of best validation metrics achieved
            - "best_score": float representing the best composite validation score
    """
    # Setup logging
    logger = logging.getLogger('train_ops')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            logger.info(f"Created log directory: {log_dir}")
        
        # FileHandler will create the file if it doesn't exist
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    device_name = get_device()
    logger.info(f"Using device: {device_name}")
    logger.info(f"Training configuration - Epochs: {max_epochs}, Initial LR: {initial_lr}, "
                f"Val interval: {val_interval}")

    # Accelerator
    if not accelerator:
        accelerator = Accelerator()
        logger.info("Initialized Accelerator with standard precision")

    # Model and functions
    device = accelerator.device
    effective_batch_size = batch_size * accelerator.num_processes
    lr = initial_lr * (effective_batch_size / batch_size) ** 0.5

    # Set the loss functions
    if loss_function == "dice":
        loss_fn = DiceLoss(
            include_background=include_background,
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
            reduction="mean"
        )
    elif loss_function == "dicece":
        loss_fn = DiceCELoss(
            include_background=include_background,
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
            reduction="mean",
            weight=torch.tensor([5.0]).to(device),
            lambda_ce=lam
        )
    elif loss_function == "dicefocal":
        loss_fn = DiceFocalLoss(
            include_background=include_background,
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
            reduction="mean",
            gamma=2.0,
            lambda_focal=lam
        )
    elif loss_function == "tversky":
        loss_fn = TverskyLoss(
            include_background=include_background,
            smooth_nr=0,
            smooth_dr=1e-5,
            to_onehot_y=False,
            sigmoid=True,
            reduction="mean",
            alpha=0.3,
            beta=0.7
        )
    elif loss_function == "tverskyce":
        loss_fn = TverskyCELoss(
            include_background=include_background,
            smooth_nr=0,
            smooth_dr=1e-5,
            to_onehot_y=False,
            sigmoid=True,
            reduction="mean",
            alpha=0.3,
            beta=0.7,
            weight=torch.tensor([5.0]).to(device),
            lambda_ce=lam
        )
    else:
        raise ValueError(f"`loss_function` set to {loss_function} and should be one of `dice`, `dicece`, `dicefocal`, `tversky`, `tverskyce`")
    
    optim_fn = torch.optim.AdamW(
        get_parameter_groups(model), lr
    )
    
    # Set the lr scheduler based on the option
    if scheduler == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_fn,
            T_max=max_epochs * len(train_loader)
        )
    elif scheduler == "warmup":
        warmup_iters = int(0.1 * max_epochs * len(train_loader))
        lr_scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optim_fn,
            start_factor=0.01,
            total_iters=warmup_iters
        )
        lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_fn,
            T_max=(max_epochs * len(train_loader)) - warmup_iters
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optim_fn,
            [lr_scheduler1, lr_scheduler2],
            milestones=[warmup_iters]
        )
    elif scheduler == "onecycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim_fn,
            max_lr=lr * 10,
            total_steps=max_epochs * len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100
        )
    else:
        raise ValueError(f"`scheduler` set to {scheduler} and should be one of `cos`, `warmup`, `onecycle`")
    
    early_stopper = EarlyStopper(
        patience=patience,
        delta=0.005,
        alpha=0.8
    )
    
    logger.info(f"Initialized loss functions ({loss_function}), optimizer (AdamW), and scheduler ({scheduler})")

    # Move everything to device
    model, optim_fn, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model,
        optim_fn,
        train_loader,
        val_loader,
        lr_scheduler
    )

    # Metrics
    metrics = {
        "dice mean": DiceMetric(include_background=include_background, reduction="mean"),
        "rvd mean": RelativeVolumdeDifferenceMetric(),
        "hausdorff mean": HausdorffDistanceMetric(
            include_background=include_background,
            distance_metric="euclidean",
            percentile=95.0,
            directed=False,
            reduction="mean",
            get_not_nans=False
        ),
        "surface dice mean": SurfaceDiceMetric(
            class_thresholds=[2.0],
            include_background=include_background,
            distance_metric="euclidean",
            reduction="mean",
            get_not_nans=False,
            use_subvoxels=False
        ),
        "precision mean": ConfusionMatrixMetric(
            include_background=include_background,
            metric_name="precision",
            reduction="mean"
        ),
        "recall mean": ConfusionMatrixMetric(
            include_background=include_background,
            metric_name="sensitivity",
            reduction="mean"
        )
    }

    # Validation weighting (we do uniform weighting)
    metric_weights = {
        k: 1.0 / len(metrics)
        for k in metrics.keys()
    }
    val_scorer = WeightedValidationMetrics(
        metric_weights,
        ["rvd mean", "hausdorff mean"],
        recent_window=max_epochs
    )
    
    logger.info(f"Initialized {len(metrics)} validation metrics: {list(metrics.keys())}")

    # Initialize tracking lists for losses and metrics
    epoch_loss_values = []
    metric_values = []
    best_score = 1.0
    best_metrics = None

    for epoch in range(max_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_losses = []

        # Train on batch
        for i, batch in enumerate(train_loader):
            inputs, labels, seq_lengths, dates = (
                batch["images"], 
                batch["labels"],
                batch["sequence_lengths"],
                batch["dates"]
            )

            loss = train_step(
                model,
                inputs,
                labels,
                seq_lengths,
                dates,
                optim_fn,
                loss_fn,
                accelerator,
                device,
                logger,
            )

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Epoch {epoch + 1} - Batch {i + 1}/{len(train_loader)} - Train loss: {loss:.4f}")

            # Add losses
            epoch_losses.append(loss)

            # Clear the batch
            del inputs, labels, seq_lengths, dates, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            lr_scheduler.step()
        
        mean_epoch_loss = np.mean(epoch_losses)
        epoch_loss_values.append(mean_epoch_loss)
        logger.info(f"Epoch {epoch + 1}/{max_epochs} completed - Mean loss: {mean_epoch_loss:.4f}")

        # Validate
        if (epoch + 1) % val_interval == 0:
            logger.info(f"Running validation for epoch {epoch + 1}/{max_epochs}")
            results = validation(
                model,
                val_loader,
                metrics,
                logger,
                use_sliding_window
            )
            
            # Get val score
            val_losses = {}
            
            for k, metric in results.items():
                if k not in ["rvd mean", "hausdorff mean"]:
                    val_losses[k] = 1.0 - metric
                else:
                    val_losses[k] = metric

            val_score = val_scorer.score(val_losses)

            # Update early stopper
            early_stopper.check_early_stop(val_score)

            # Check whether the score is the best
            if val_score < best_score:
                best_score = val_score
                best_metrics = results

            # Store values
            metric_values.append(results)

            logger.info(f"Validation results for epoch {epoch + 1}/{max_epochs}:")
            for k, v in results.items():
                logger.info(f"  {k}: {v:.4f}")
            logger.info(f"composite score {val_score:.4f}, best score {best_score:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("Cleared CUDA cache and ran garbage collection")

        # End loop if early stop
        if early_stopper.stop_training:
            break
    
    logger.info("Training completed successfully")

    # Clear accelerator
    optim_fn, train_loader, val_loader, lr_scheduler = accelerator.clear(
        optim_fn,
        train_loader,
        val_loader,
        lr_scheduler
    )

    model = accelerator.unwrap_model(model)
    
    # Prepare output dictionary with training results
    output_dict = {
        "epoch_losses": epoch_loss_values,
        "validation_metrics": metric_values,
        "best_metrics": best_metrics if best_metrics else {},
        "best_score": best_score
    }
    
    if epoch_loss_values:
        logger.info(f"Training summary - Total epochs: {epoch} of {max_epochs}, Final loss: {epoch_loss_values[-1]:.4f}")
    if metric_values:
        logger.info("Best validation metrics:")
        for k, v in best_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
    
    return model, output_dict

def train_step(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        seq_lengths: torch.Tensor,
        dates: torch.Tensor,
        optim_fn: torch.optim.Adam,
        loss_fn: DiceLoss,
        accelerator: Accelerator,
        device: torch.device,
        logger: logging.Logger
) -> float:
    """
    Perform a single training step on a batch using autoregressive forward passes.
    
    This function processes a batch by iterating through each time step in the sequence. 
    For each time step i, the model uses all previous images (0 to i) and masks to 
    predict the segmentation at time step i+1. Gradients are accumulated across all 
    time steps before performing a single optimizer step with gradient clipping.
    
    Args:
        model : torch.nn.Module : Neural network model for segmentation
        inputs : torch.Tensor : Input images with shape (batch_size, sequence_length, 
            channels, height, width, depth)
        labels : torch.Tensor : Ground truth segmentation labels with shape 
            (batch_size, sequence_length, channels, height, width, depth)
        seq_lengths : torch.Tensor : Actual sequence length for each batch element with 
            shape (batch_size,)
        dates : torch.Tensor : Timestamps for each image in the sequences with shape 
            (batch_size, sequence_length)
        optim_fn : torch.optim.Adam : AdamW optimizer instance for updating model parameters
        loss_fn : DiceLoss : Loss function for computing segmentation loss (Dice, DiceCE, etc.)
        accelerator : Accelerator : Accelerate instance for mixed precision training and 
            distributed support
        device : torch.device : Device to move tensors to (CPU or CUDA)
        logger : logging.Logger : Logger instance for recording training progress
    
    Returns:
        loss : float : Average loss across all time steps in the sequence
    """

    T = inputs.shape[1]

    seq_losses = []

    # Zero the gradient
    optim_fn.zero_grad()

    for i in range(T-1):
        # Grab the relevant inputs
        x, y, ds = (
            inputs[:, :i+1].to(device), 
            labels[:, i+1].to(device), 
            dates[:, :i+2].to(device)
        )

        # Update sequence lengths
        adjusted_seq_lengths = torch.minimum(
            seq_lengths, torch.full_like(seq_lengths, i+1)
        ).clamp_min(1.0).to(device)

        # Forward pass
        y_hat = model(x, adjusted_seq_lengths, ds)

        # Get the combined loss
        loss = loss_fn(y_hat, y)

        # Scale by the length
        loss = loss / (T - 1)

        # Backpropagate
        accelerator.backward(loss)

        # Add the loss item
        seq_losses.append(loss.detach().item())

        logger.debug(f"{i} loss: {loss.detach().item():.4f}")

        # Delete intermediate tensors and free cache
        del x, y, ds, adjusted_seq_lengths, y_hat, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Clip the gradients
    accelerator.clip_grad_norm_(model.parameters(), 1.0)

    # Step
    optim_fn.step()

    # Get the average loss
    loss = np.sum(seq_losses)

    # Garbage clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return loss

def validation(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        metrics: dict[str, monai.metrics.Metric],
        logger: logging.Logger,
        use_sliding_window: bool
) -> dict[str, float]:
    """
    Perform validation on the entire validation dataset and compute segmentation metrics.
    
    This function runs the model in evaluation mode on all validation batches and 
    computes predictions using the same autoregressive approach as training. It 
    calculates various segmentation metrics including Dice coefficient, Hausdorff 
    distance, surface Dice, precision, and recall. Metrics are aggregated across 
    all validation samples.
    
    Args:
        model : torch.nn.Module : Trained neural network model to validate
        val_loader : torch.utils.data.DataLoader : DataLoader containing validation data 
            with batches of images, labels, sequence lengths, and dates
        metrics : dict[str, monai.metrics.Metric] : Dictionary mapping metric names to 
            MONAI metric objects to compute (e.g., DiceMetric, HausdorffDistanceMetric)
        logger : logging.Logger : Logger instance for recording validation progress
        use_sliding_window : bool : Whether to use sliding window inference for predictions
    
    Returns:
        results : dict[str, float] : Dictionary mapping metric names to their computed 
            scalar values aggregated across all validation samples
    """
    model.eval()
    
    logger.debug(f"Starting validation on {len(val_loader)} batches")

    post_trans = get_post_transform()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, labels, seq_lengths, dates = (
                batch["images"], 
                batch["labels"],
                batch["sequence_lengths"],
                batch["dates"]
            )

            # Perform validation step
            validation_step(
                model,
                inputs,
                labels,
                seq_lengths,
                dates,
                post_trans,
                metrics,
                logger,
                use_sliding_window
            )
            
            logger.debug(f"Processed validation batch {batch_idx + 1}/{len(val_loader)}")

        # Aggregate statistics
        results = {}
        for k, metric in metrics.items():
            agg_result = metric.aggregate()
            if isinstance(agg_result, list):
                results[k] = agg_result[0].item()
            else:
                results[k] = agg_result.item()

        # Reset the metrics
        for metric in metrics.values():
            metric.reset()
        
        logger.debug("Validation metrics aggregated successfully and reset")
    
    return results

def validation_step(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        seq_lengths: torch.Tensor,
        dates: torch.Tensor,
        post_trans: monai.transforms.Transform,
        metrics: dict[str, monai.metrics.Metric],
        logger: logging.Logger,
        use_sliding_window: bool
) -> None:
    """
    Process a single validation batch using autoregressive inference and update metrics.
    
    This function iterates through each time step in the validation sequences in an 
    autoregressive manner, similar to the training process but without gradient 
    computation. For each time step, it generates predictions using all previous 
    images and masks, applies post-processing transforms, filters predictions based 
    on valid sequence lengths, and updates metric accumulators.
    
    Args:
        model : torch.nn.Module : Neural network model in evaluation mode
        inputs : torch.Tensor : Input images for validation batch with shape 
            (batch_size, sequence_length, channels, height, width, depth)
        labels : torch.Tensor : Ground truth segmentation labels with shape 
            (batch_size, sequence_length, channels, height, width, depth)
        seq_lengths : torch.Tensor : Actual sequence length for each batch element 
            with shape (batch_size,)
        dates : torch.Tensor : Timestamps for each image with shape 
            (batch_size, sequence_length)
        post_trans : monai.transforms.Transform : MONAI transform pipeline for 
            post-processing predictions (e.g., thresholding, converting to discrete labels)
        metrics : dict[str, monai.metrics.Metric] : Dictionary of MONAI metric objects 
            to update with predictions and ground truth
        logger : logging.Logger : Logger instance for recording validation step progress
        use_sliding_window : bool : Whether to use sliding window inference for predictions
    
    Returns:
        None : Metrics are updated in-place through their internal accumulators
    """

    T = inputs.shape[1]

    for i in range(T - 1):
        # Grab the relevant inputs
        x, y, ds = (
            inputs[:, :i+1], 
            labels[:, i+1], 
            dates[:, :i+2]
        )

        # Update sequence lengths
        adjusted_seq_lengths = torch.minimum(
            seq_lengths, torch.full_like(seq_lengths, i+1)
        ).clamp_min(1.0)

        # Get valid mask
        valid_mask = (i + 1) < seq_lengths

        # Use inference
        outputs = inference(
            model,
            x,
            adjusted_seq_lengths,
            ds,
            use_sliding_window=use_sliding_window
        )

        # Decollate
        outputs = [post_trans(z) for z in decollate_batch(outputs)]

        # Filter with valid mask
        valid_indices = torch.where(valid_mask)[0]
        valid_outputs = [outputs[idx] for idx in valid_indices]
        valid_y = y[valid_mask]

        # Get metrics
        for metric in metrics.values():
            metric(y_pred=valid_outputs, y=valid_y)
        
        logger.debug(f"Validation step {i+1}/{T-1}: {len(valid_outputs)} valid predictions")