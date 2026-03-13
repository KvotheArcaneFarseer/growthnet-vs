"""
Medical Image Segmentation Utilities Module

This module provides custom loss functions, metrics, and training utilities specifically
designed for medical image segmentation tasks using PyTorch and MONAI.

Contents:
    - TverskyCELoss: Combined Tversky and Binary Cross-Entropy loss for segmentation
    - RelativeVolumdeDifferenceMetric: Metric for computing relative volume differences
    - EarlyStopper: Early stopping implementation with exponential moving average
    - WeightedValidationMetrics: Composite metric scoring with adaptive normalization
    - get_parameter_groups: Utility for organizing model parameters with weight decay
    - get_device: Utility for device detection (CUDA/CPU)

Dependencies:
    - torch: PyTorch deep learning framework
    - monai: Medical Open Network for AI library
    - numpy: Numerical computing
    - scipy: Scientific computing utilities
"""

import torch
import monai
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.nn import BCEWithLogitsLoss
from scipy.stats import zscore
from scipy.special import expit
from typing import Any, Callable
from monai.losses.tversky import TverskyLoss
from monai.utils import LossReduction, look_up_option
from monai.utils.enums import StrEnum
from monai.metrics import CumulativeIterationMetric
from monai.metrics.utils import do_metric_reduction
from monai.networks.utils import one_hot

class TverksyCEReduction(StrEnum):
    MEAN = "mean"
    SUM = "sum"

class TverskyCELoss(_Loss):
    """
    Combined Tversky and Binary Cross-Entropy loss for image segmentation.
    
    This loss function combines the Tversky loss (which generalizes Dice and Focal losses)
    with Binary Cross-Entropy loss, allowing for flexible weighting between the two
    components. Useful for handling class imbalance in medical image segmentation.
    
    Args:
        include_background : bool : Whether to include background class in loss calculation (default: True)
        to_onehot_y : bool : Whether to convert target to one-hot encoding (default: False)
        sigmoid : bool : Whether to apply sigmoid activation to input (default: False)
        softmax : bool : Whether to apply softmax activation to input (default: False)
        other_act : Callable | None : Optional custom activation function to apply (default: None)
        alpha : float : Weight for false positives in Tversky loss (default: 0.5)
        beta : float : Weight for false negatives in Tversky loss (default: 0.5)
        reduction : LossReduction | str : Method for reducing loss ('mean' or 'sum') (default: LossReduction.MEAN)
        smooth_nr : float : Smoothing constant for numerator to avoid division by zero (default: 1e-5)
        smooth_dr : float : Smoothing constant for denominator to avoid division by zero (default: 1e-5)
        batch : bool : Whether to compute loss per batch or per sample (default: False)
        weight : float | None : Positive class weight for BCE loss (default: None)
        lambda_tversky : float : Weight coefficient for Tversky loss component (default: 1.0)
        lambda_ce : float : Weight coefficient for Cross-Entropy loss component (default: 1.0)
    """
    def __init__(
            self, 
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act: Callable | None = None,
            alpha: float = 0.5,
            beta: float = 0.5,
            reduction: LossReduction | str = LossReduction.MEAN,
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
            weight: float | None = None,
            lambda_tversky: float = 1.0,
            lambda_ce: float = 1.0
    ) -> None:
        super().__init__()

        reduction = look_up_option(reduction, TverksyCEReduction).value
        self.tversky = TverskyLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            alpha=alpha,
            beta=beta,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch
        )
        self.bce = BCEWithLogitsLoss(
            pos_weight=weight,
            reduction=reduction
        )
        if lambda_tversky < 0.0:
            raise ValueError(f"`lambda_tverksy` with value {lambda_tversky} should be no less 0.0")
        if lambda_ce < 0.0:
            raise ValueError(f"`lambda_ce` with value {lambda_ce} should be no less 0.0")
        
        self.lambda_tversky = lambda_tversky
        self.lambda_ce = lambda_ce
    
    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the combined Tversky and Cross-Entropy loss.
        
        Args:
            input : torch.Tensor : Model predictions with shape (B, C, H, W, [D]) where B is batch size,
                                   C is number of channels, H is height, W is width, D is depth (optional)
            target : torch.Tensor : Ground truth labels with shape (B, C, H, W, [D]) or (B, 1, H, W, [D])
                                    for non-one-hot encoded targets
        
        Returns:
            total_loss : torch.Tensor : Scalar tensor containing the weighted sum of Tversky and CE losses
        """
        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )
        
        tversky_loss = self.tversky(input, target)
        ce_loss = self.bce(input, target)
        total_loss = self.lambda_tversky * tversky_loss + self.lambda_ce * ce_loss

        return total_loss

class RelativeVolumdeDifferenceMetric(CumulativeIterationMetric):
    """
    Metric for computing relative volume difference between predictions and ground truth.
    
    Calculates the relative difference in volume (number of positive voxels) between
    predicted and target segmentation masks. Useful for evaluating over/under-segmentation
    in medical imaging tasks. Returns NaN for empty target volumes to avoid division by zero.
    
    Args:
        include_background : bool : Whether to include background class in metric calculation (default: False)
        reduction : str : Method for reducing metric across batch ('mean', 'sum', etc.) (default: 'mean')
    """
    def __init__(
            self,
            include_background=False,
            reduction="mean"
    ) -> None:
        super().__init__()

        self.include_background = include_background
        self.reduction = reduction
    
    def aggregate(
            self,
            reduction: monai.utils.enums.MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate the buffered metric values across all iterations.
        
        Args:
            reduction : monai.utils.enums.MetricReduction | str | None : Optional reduction method to override
                                                                          the default reduction (default: None)
        
        Returns:
            f : torch.Tensor | tuple[torch.Tensor, torch.Tensor] : Aggregated metric value(s), scalar or tuple
                                                                    depending on reduction method
        """
        data = self.get_buffer().unsqueeze(1)
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"the data aggregate must be PyTorch Tensor, got {type(data)}.")
        
        # perform metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)

        return f

    def _compute_tensor(
            self,
            y_pred: torch.Tensor,
            y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relative volume difference for a single batch.
        
        Args:
            y_pred : torch.Tensor : Predicted segmentation mask with shape (B, C, H, W, [D])
            y : torch.Tensor : Ground truth segmentation mask with shape (B, C, H, W, [D])
        
        Returns:
            f : torch.Tensor : Relative volume difference metric with shape (B,) after reduction,
                              contains NaN for samples with empty target volumes
        """
        # Volume calculations
        pred_volume = torch.sum((y_pred > 0).flatten(1), dim=1).float()
        target_volume = torch.sum((y > 0).flatten(1), dim=1).float()

        # Initialize with NaN
        rvd = torch.full_like(pred_volume, float('nan'))

        # Create masks to handle binary classification
        target_not_empty = target_volume > 0
        
        if target_not_empty.any():
            rvd[target_not_empty] = (
                torch.abs(pred_volume[target_not_empty].abs() - target_volume[target_not_empty].abs()) 
                / target_volume[target_not_empty].abs()
            )

        # Add channel
        rvd = rvd.unsqueeze(1)

        f, not_nans = do_metric_reduction(rvd, self.reduction)

        return f

class EarlyStopper:
    """
    Early stopping implementation with exponential moving average smoothing.
    
    Monitors validation loss with exponential smoothing and stops training when no
    improvement is observed for a specified number of epochs (patience). Uses a delta
    threshold to determine significant improvements.
    
    Args:
        patience : int : Number of epochs to wait for improvement before stopping
        delta : float : Minimum change in smoothed loss to qualify as improvement (default: 0.0)
        alpha : float : Smoothing factor for exponential moving average, range [0, 1] (default: 0.9)
        verbose : bool : Whether to print messages when stopping early (default: False)
    """
    def __init__(
            self,
            patience: int,
            delta: float = 0.0,
            alpha: float = 0.9,
            verbose: bool = False
    ) -> None:
        # Attributes
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_smoothed_loss = float("inf")
        self.alpha = alpha
        self.smoothed_loss = 0
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(
            self,
            val_loss: float
    ) -> None:
        """
        Check if training should stop based on validation loss.
        
        Updates the exponential moving average of validation loss and determines
        whether improvement has occurred. Sets the stop_training flag if patience
        threshold is exceeded.
        
        Args:
            val_loss : float : Current epoch's validation loss value
        
        Returns:
            None : Updates internal state attributes (stop_training, no_improvement_count)
        """
        # Update the smoothed loss
        self._compute_moving_average(val_loss)

        # Check for how it has changed
        if self.smoothed_loss < self.best_smoothed_loss - self.delta:
            self.best_smoothed_loss = self.smoothed_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count > self.patience:
                self.stop_training = True
                if self.verbose:
                    print("stopping early as no improvement has been observed")
    
    def _compute_moving_average(
            self,
            val_loss: float
    ) -> None:
        """
        Update the exponential moving average of validation loss.
        
        Args:
            val_loss : float : Current validation loss value to incorporate into moving average
        
        Returns:
            None : Updates the smoothed_loss attribute in-place
        """
        # Update the smoothed loss
        next_loss = self.alpha * val_loss + (1 - self.alpha) * self.smoothed_loss
        self.smoothed_loss = next_loss

class WeightedValidationMetrics:
    """
    Composite metric scoring with adaptive z-score normalization.
    
    Computes a weighted combination of multiple validation metrics, applying adaptive
    z-score normalization based on recent metric history. Useful for model selection
    when multiple metrics need to be balanced with different importance weights.
    
    Args:
        metric_weighting : dict[str, float] : Dictionary mapping metric names to their weights
        history_names : list[str] : List of metric names to track history for normalization
        recent_window : int : Number of recent epochs to use for z-score calculation (default: 10)
    """
    def __init__(
            self,
            metric_weighting: dict[str, float],
            history_names: list[str],
            recent_window: int = 10
    ) -> None:
        self.metric_weighting = metric_weighting
        self.metric_history = {
            name: [] for name in history_names
        }
        self.recent_window = recent_window
    
    def score(
            self,
            metrics: dict[str, float]
    ) -> float:
        """
        Compute weighted composite score from multiple metrics.
        
        Normalizes tracked metrics using adaptive z-scores and sigmoid transformation,
        then computes weighted sum according to the metric weighting scheme. Non-tracked
        metrics are used directly without normalization.
        
        Args:
            metrics : dict[str, float] : Dictionary mapping metric names to their current values
        
        Returns:
            composite_score : float : Weighted sum of normalized metrics
        """
        # Get the z score for the right metrics
        values = { k: metrics[k] for k in self.metric_history.keys() }
        normalized_values = self._normalize(values)

        # Add to normalized metrics
        normalized_metrics = {
            k: normalized_values[k] if k in set(self.metric_history.keys()) else metrics[k] 
            for k in metrics.keys()
        }

        return np.sum([
            self.metric_weighting[k] * normalized_metrics[k] 
            for k in self.metric_weighting.keys() 
        ])
    
    def _normalize(
            self,
            values: dict[str, float]
    ) -> dict[str, float]:
        """
        Normalize metrics using adaptive z-scores and sigmoid transformation.
        
        Computes z-scores based on recent metric history window and applies sigmoid
        function to map to [0, 1] range. Updates metric history with new values.
        
        Args:
            values : dict[str, float] : Dictionary of metric values to normalize
        
        Returns:
            normalized_values : dict[str, float] : Dictionary of sigmoid-transformed z-scores,
                                                    range [0, 1] for each metric
        """
        # Add the values
        for k, v in values.items():
            self.metric_history[k].append(v)
        
        # Calculate the z scores
        z_scores = [ zscore(v[-self.recent_window:])[-1] if len(v) > 1 else 100.0
                    for v in self.metric_history.values() 
        ]
        return {
            k: expit(z_scores[i]) 
            for i, k in enumerate(values.keys())
        }
    
def get_parameter_groups(
        model: torch.nn.Module
) -> list[dict[str, Any]]:
    """
    Organize model parameters into groups for differential weight decay.
    
    Separates model parameters into two groups: those that should have weight decay
    applied (weights) and those that should not (biases and normalization parameters).
    This is a common practice to improve training stability and performance.
    
    Args:
        model : torch.nn.Module : PyTorch model to extract parameters from
    
    Returns:
        optimizer_grouped_parameters : list[dict[str, Any]] : List of two parameter group dictionaries,
                                                               each containing 'params' and 'weight_decay' keys
    """
    no_decay = [
        "bias",
        "LayerNorm.weight",
        "LayerNorm.bias",
        "norm1",
        "norm2",
        "norm.weight",
        "norm.bias"
    ]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]

    return optimizer_grouped_parameters

def get_device() -> str:
    """
    Detect and return the available PyTorch device.
    
    Checks for CUDA availability and returns the appropriate device string
    for tensor operations. Defaults to CPU if CUDA is not available.
    
    Args:
        None
    
    Returns:
        device : str : Device string, either 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    return device