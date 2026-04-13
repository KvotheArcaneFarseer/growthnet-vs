"""
Pretraining Operations Module

This module provides functionality for pretraining neural network models using contrastive learning
and reconstruction objectives. It implements a combined training approach that uses contrastive loss
to learn meaningful representations and reconstruction loss to preserve image fidelity.

The module includes:
- Main pretraining loop with learning rate scheduling and validation
- Training step with contrastive and reconstruction loss computation
- Validation functions for model evaluation
- Support for distributed training via Hugging Face Accelerate
- Comprehensive logging of training metrics and progress

Key Features:
- Contrastive learning with temperature-scaled loss
- L1 reconstruction loss for image fidelity
- Warmup and cosine annealing learning rate scheduling
- Gradient clipping for training stability
- Periodic validation with best model tracking
- Memory management with CUDA cache clearing
"""

import torch
import monai
import numpy as np
import logging
import gc
import os
from accelerate import Accelerator
from monai.losses.contrastive import ContrastiveLoss
from torch.nn import L1Loss
from typing import Any, Optional
from src.train.utils import get_device, get_parameter_groups

def pretrain(
        model: torch.nn.Module,
        train_loader: monai.data.DataLoader,
        val_loader: monai.data.DataLoader,
        accelerator: Optional[Accelerator] = None,
        max_epochs: int = 5,
        initial_lr: float = 1e-4,
        val_interval: int = 1,
        batch_size: int = 4,
        log_file: Optional[str] = None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Pretrain a neural network model using contrastive learning and reconstruction objectives.
    
    This function implements a complete pretraining pipeline with warmup and cosine annealing
    learning rate scheduling, gradient clipping, and periodic validation. The training combines
    contrastive loss (for learning representations) with reconstruction loss (for preserving
    image quality). Training progress and validation metrics are logged both to console and
    optionally to a file.
    
    Args:
        model : torch.nn.Module : The neural network model to pretrain
        train_loader : monai.data.DataLoader : DataLoader providing training batches with keys
            'image', 'image_2', and 'gt_image'
        val_loader : monai.data.DataLoader : DataLoader providing validation batches with keys
            'image' and 'gt_image'
        accelerator : Accelerator | None : Hugging Face Accelerator for distributed training,
            creates new instance if None (default: None)
        max_epochs : int : Maximum number of training epochs (default: 5)
        initial_lr : float : Initial learning rate before scaling (default: 1e-4)
        val_interval : int : Number of epochs between validation runs (default: 1)
        batch_size : int : Batch size per device for loss computation (default: 4)
        log_file : Optional[str] : Path to log file, creates console-only logger if None
            (default: None)
    
    Returns:
        model : torch.nn.Module : The pretrained model unwrapped from accelerator
        output_dict : dict[str, Any] : Dictionary containing training results with keys:
            - 'epoch_losses': list of mean total losses per epoch
            - 'validation_metrics': list of validation result dictionaries
            - 'best_metrics': float of best validation reconstruction loss achieved
    """
    # Setup logging
    logger = logging.getLogger('pretrain_ops')
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

    contrastive_loss_fn = ContrastiveLoss(temperature=0.05, batch_size=batch_size)
    recon_loss_fn = L1Loss()

    optim_fn = torch.optim.AdamW(
        get_parameter_groups(model), lr
    )

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

    logger.info(f"Initialized loss functions (contrastive and L1), optimizer (AdamW), and scheduler (warmup)")

    # Move everything to device
    model, optim_fn, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model,
        optim_fn,
        train_loader,
        val_loader,
        lr_scheduler
    )

    epoch_total_losses = []
    validation_losses = []

    best_validation_loss = float("inf")

    for epoch in range(max_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{max_epochs}")
        model.train()
        contrastive_losses = []
        recon_losses = []
        total_losses = []

        for i, batch in enumerate(train_loader):
            inputs = {
                "image1": batch["image"],
                "image2": batch["image_2"],
                "gt_image": batch["gt_image"]
            }

            losses = train_step(
                model,
                inputs,
                optim_fn,
                contrastive_loss_fn,
                recon_loss_fn,
                accelerator,
                logger
            )

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Epoch {epoch + 1} - Batch {i + 1}/{len(train_loader)} - contrastive loss: {losses["con loss"]:.4f}; recon loss: {losses["recon loss"]:.4f}; total loss: {losses["total loss"]:.4f}")

            # Add losses
            contrastive_losses.append(losses["con loss"])
            recon_losses.append(losses["recon loss"])
            total_losses.append(losses["total loss"])

            # Clear the batch
            del inputs, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            lr_scheduler.step()
        
        # Get the averages
        mean_epoch_contrastive_loss = np.mean(contrastive_losses)
        mean_epoch_recon_loss = np.mean(recon_losses)
        mean_epoch_loss = np.mean(total_losses)

        epoch_total_losses.append(mean_epoch_loss)
        logger.info(f"Epoch {epoch + 1}/{max_epochs} completed - Mean loss: {mean_epoch_loss:.4f}; Mean contrastive loss: {mean_epoch_contrastive_loss:.4f}; Mean recon loss: {mean_epoch_recon_loss:.4f}")

        # Validate
        if (epoch + 1) % val_interval == 0:
            logger.info(f"Running validation for epoch {epoch + 1}/{max_epochs}")
            results = validation(
                model,
                val_loader,
                contrastive_loss_fn,
                recon_loss_fn,
                logger,
            )

            validation_losses.append(results)
            
            logger.info(f"Validation results for epoch {epoch + 1}/{max_epochs}:")
            for k, v in results.items():
                logger.info(f"  {k}: {v:.4f}")
            
            if results["recon mean"] < best_validation_loss:
                best_validation_loss = results["recon mean"]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("Cleared CUDA cache and ran garbage collection")

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
        "epoch_losses": epoch_total_losses,
        "validation_metrics": validation_losses,
        "best_metrics": best_validation_loss
    }
    
    if epoch_total_losses:
        logger.info(f"Training summary - Total epochs: {epoch} of {max_epochs}, Final loss: {epoch_total_losses[-1]:.4f}")
        logger.info(f"Best validation loss: {best_validation_loss:.4f}")
    
    return model, output_dict

def train_step(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        optim_fn: torch.optim.Adam,
        contrastive_loss_fn: ContrastiveLoss,
        recon_loss_fn: L1Loss,
        accelerator: Accelerator,
        logger: logging.Logger
) -> dict[str, float]:
    """
    Execute a single training step with forward pass, loss computation, and backpropagation.
    
    This function performs one complete training iteration including: forward passes through the
    model with two augmented views of the input, computation of contrastive loss between the
    views, computation of reconstruction loss against ground truth, combined loss calculation,
    backpropagation with gradient clipping, and optimizer step. Memory is cleared after the step.
    
    Args:
        model : torch.nn.Module : The neural network model in training mode
        inputs : dict : Dictionary containing input tensors with keys:
            - 'image1': torch.Tensor of shape (B, C, D, H, W) - first augmented view
            - 'image2': torch.Tensor of shape (B, C, D, H, W) - second augmented view
            - 'gt_image': torch.Tensor of shape (B, C, D, H, W) - ground truth image
            where B=batch, C=channels, D=depth, H=height, W=width
        optim_fn : torch.optim.Adam : AdamW optimizer for model parameters
        contrastive_loss_fn : ContrastiveLoss : MONAI contrastive loss function
        recon_loss_fn : L1Loss : L1 reconstruction loss function
        accelerator : Accelerator : Hugging Face Accelerator for distributed training
        logger : logging.Logger : Logger for debug messages
    
    Returns:
        train_step_losses : dict[str, float] : Dictionary containing scalar loss values:
            - 'con loss': contrastive loss value
            - 'recon loss': reconstruction loss value
            - 'total loss': combined total loss value (con_loss * recon_loss + recon_loss)
    """
    inputs1, inputs2, gt_input = (
        inputs["image1"],
        inputs["image2"],
        inputs["gt_image"]
    )

    # Zero the gradient
    optim_fn.zero_grad()
    
    # Forward passes
    outputs_v1, _ = model(inputs1)
    outputs_v2, _ = model(inputs2)

    # Flatten
    flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
    flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

    # Contrastive and Recon Loss
    con_loss = contrastive_loss_fn(flat_out_v1, flat_out_v2)
    recon_loss = recon_loss_fn(outputs_v1, gt_input)

    # Total loss
    loss = con_loss * recon_loss + recon_loss

    # Backpropagate
    accelerator.backward(loss)

    # Clip the gradients
    accelerator.clip_grad_norm_(model.parameters(), 1.0)

    # Step
    optim_fn.step()

    # Detach losses
    loss = loss.detach().item()
    con_loss = con_loss.detach().item()
    recon_loss = recon_loss.detach().item()

    logger.debug(f"con loss: {con_loss:.4f}\nrecon loss: {recon_loss:.4f}\ntotal loss: {loss:.4f}")

    # Store results
    train_step_losses = {
        "con loss": con_loss,
        "recon loss": recon_loss,
        "total loss": loss
    }

    # Garbage cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return train_step_losses

def validation(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        contrastive_loss_fn: ContrastiveLoss,
        recon_loss_fn: L1Loss,
        logger: logging.Logger
) -> dict[str, float]:
    """
    Perform validation by evaluating the model on the validation dataset.
    
    This function evaluates the model in eval mode with gradient computation disabled. It
    iterates through all validation batches, computes reconstruction losses, and aggregates
    the results to produce mean validation metrics. The model is automatically returned to
    its original state after validation.
    
    Args:
        model : torch.nn.Module : The neural network model to evaluate
        val_loader : torch.utils.data.DataLoader : DataLoader providing validation batches
            with keys 'image' and 'gt_image'
        contrastive_loss_fn : ContrastiveLoss : MONAI contrastive loss function (passed to
            validation_step but not used)
        recon_loss_fn : L1Loss : L1 reconstruction loss function for computing validation loss
        logger : logging.Logger : Logger for debug messages during validation
    
    Returns:
        results : dict[str, float] : Dictionary containing aggregated validation metrics:
            - 'recon mean': mean reconstruction loss across all validation batches
    """
    model.eval()
    
    logger.debug(f"Starting validation on {len(val_loader)} batches")

    recon_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs = {
                "image1": batch["image"],
                "gt_image": batch["gt_image"]
            }

            # Perform validation step
            losses = validation_step(
                model,
                inputs,
                contrastive_loss_fn,
                recon_loss_fn,
                logger
            )

            recon_losses.append(losses["recon loss"])
            
            logger.debug(f"Processed validation batch {batch_idx + 1}/{len(val_loader)}")

        # Aggregate statistics
        mean_recon_loss = np.mean(recon_losses)

        # Store results
        results = {
            "recon mean": mean_recon_loss,
        }
        
        logger.debug("Validation metrics aggregated successfully and reset")
    
    return results

def validation_step(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        contrastive_loss_fn: ContrastiveLoss,
        recon_loss_fn: L1Loss,
        logger: logging.Logger,
) -> dict[str, float]:
    """
    Execute a single validation step with forward pass and loss computation.
    
    This function performs one validation iteration without gradient computation. It conducts
    a forward pass through the model with a single input view, computes reconstruction loss
    against ground truth, and returns the detached loss value for aggregation.
    
    Args:
        model : torch.nn.Module : The neural network model in eval mode
        inputs : dict : Dictionary containing input tensors with keys:
            - 'image1': torch.Tensor of shape (B, C, D, H, W) - input image
            - 'gt_image': torch.Tensor of shape (B, C, D, H, W) - ground truth image
            where B=batch, C=channels, D=depth, H=height, W=width
        contrastive_loss_fn : ContrastiveLoss : MONAI contrastive loss function (not used in
            validation step)
        recon_loss_fn : L1Loss : L1 reconstruction loss function for computing validation loss
        logger : logging.Logger : Logger for debug messages
    
    Returns:
        results : dict[str, float] : Dictionary containing scalar loss value:
            - 'recon loss': reconstruction loss value for this validation step
    """
    inputs1, gt_input = (
        inputs["image1"],
        inputs["gt_image"]
    )
    
    # Forward passes
    outputs_v1, hidden_v1 = model(inputs1)

    # Contrastive and Recon Loss
    recon_loss = recon_loss_fn(outputs_v1, gt_input)

    recon_loss = recon_loss.detach().item()

    # Store results
    results = {
        "recon loss": recon_loss,
    }
        
    logger.debug(f"Validation step total loss: {recon_loss:.4f}")

    return results