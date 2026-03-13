import torch
from typing import Literal, Tuple, Union, Optional
from monai.inferers import sliding_window_inference
from monai.data import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple

def temporal_sliding_window_inference(
        model: torch.nn.Module,
        input: torch.Tensor,
        seq_lengths: torch.Tensor,
        dates: torch.Tensor,
        roi_size: Union[Tuple[int, int, int], int] = (64, 64, 64),
        sw_batch_size: int = 1,
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[float, Tuple[float, ...]] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Optional[Union[torch.device, str]] = None,
        device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """
    Sliding window inference for TemporalUNETR model.
    
    This function applies sliding window inference on the spatial dimensions (H, W, D)
    while keeping the temporal dimension intact. Each spatial window processes the 
    full temporal sequence.
    
    Args:
        model: TemporalUNETR model
        input: Input tensor with shape (B, T, C, H, W, D)
        seq_lengths: Sequence lengths tensor with shape (B,)
        dates: Dates tensor with shape (B, T)
        roi_size: Size of the sliding window for spatial dimensions
        sw_batch_size: Number of windows to process in parallel
        overlap: Overlap ratio between windows (0-1)
        mode: Blending mode for overlapping windows
        sigma_scale: Sigma scale for Gaussian blending
        padding_mode: Padding mode for the input
        cval: Constant value for padding
        sw_device: Device for sliding window computation
        device: Device for final output
        
    Returns:
        Output tensor with shape (B, C, H, W, D)
    """
    
    # Get the shape
    B, T, C, H, W, D = input.shape
    
    # Process each batch item separately to handle different seq_lengths
    outputs = []
    
    for b in range(B):
        # Get the current batch item
        batch_input = input[b:b+1]  # Shape: (1, T, C, H, W, D)
        batch_seq_length = seq_lengths[b:b+1]  # Shape: (1,)
        batch_dates = dates[b:b+1] if dates is not None else None  # Shape: (1, T)
        
        # Extract the last valid timestep for spatial inference
        # This is what we'll use for the sliding window spatial dimensions
        last_idx = int(seq_lengths[b].item()) - 1
        spatial_reference = batch_input[:, last_idx, ...]  # Shape: (1, C, H, W, D)
        
        # Define a wrapper function that processes temporal sequences for each window
        def window_model_wrapper(window_input: torch.Tensor) -> torch.Tensor:
            """
            Wrapper that takes a spatial window and processes it through the temporal model.
            
            Args:
                window_input: Shape (1, C, h, w, d) where h, w, d are window dimensions
                
            Returns:
                Output with shape (1, out_C, h, w, d)
            """
            # Get window dimensions
            _, c, h, w, d = window_input.shape
            
            # Create temporal input for this window
            # We need to extract the same spatial window from all timesteps
            window_temporal = torch.zeros((1, T, c, h, w, d), 
                                         dtype=window_input.dtype, 
                                         device=window_input.device)
            
            # Since we're using the last timestep as reference, we need to extract
            # the corresponding window from all previous timesteps
            # For simplicity, we'll replicate the window across time
            # In practice, you might want to extract the actual window from each timestep
            for t in range(int(batch_seq_length.item())):
                # This is a simplified approach - ideally you'd extract the 
                # corresponding spatial window from each timestep
                window_temporal[:, t, ...] = window_input[0]
            
            # Process through the model
            output = model(window_temporal, batch_seq_length, batch_dates)
            
            return output
        
        # Apply sliding window inference on spatial dimensions
        batch_output = sliding_window_inference(
            inputs=spatial_reference,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=window_model_wrapper,
            overlap=overlap,
            mode=mode,
            sigma_scale=sigma_scale,
            padding_mode=padding_mode,
            cval=cval,
            sw_device=sw_device,
            device=device
        )
        
        outputs.append(batch_output)
    
    # Stack all batch outputs
    output = torch.cat(outputs, dim=0)
    
    return output


def proper_temporal_sliding_window_inference(
        model: torch.nn.Module,
        input: torch.Tensor,
        seq_lengths: torch.Tensor,
        dates: torch.Tensor,
        roi_size: Tuple[int, int, int] = (64, 64, 64),
        sw_batch_size: int = 1,
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[float, Tuple[float, ...]] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Optional[Union[torch.device, str]] = None,
        device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """
    Proper sliding window inference that extracts corresponding spatial windows 
    from all timesteps in the sequence.
    
    This implementation:
    1. Computes sliding window positions from the last timestep
    2. Extracts the same spatial window from all timesteps
    3. Processes each temporal window through the model
    4. Aggregates results with proper blending
    """
    # Get the shape
    B, T, C, H, W, D = input.shape
    
    # Process each batch item
    outputs = []
    
    for b in range(B):
        batch_input = input[b:b+1]  # (1, T, C, H, W, D)
        batch_seq_length = seq_lengths[b:b+1]
        batch_dates = dates[b:b+1] if dates is not None else None
        valid_t = int(batch_seq_length.item())
        
        # We'll manually handle the sliding window to maintain temporal consistency
        # Get the last timestep as reference for spatial dimensions
        spatial_shape = (H, W, D)
        
        # Compute padding if needed
        h_pad = max(roi_size[0] - spatial_shape[0], 0)
        w_pad = max(roi_size[1] - spatial_shape[1], 0)
        d_pad = max(roi_size[2] - spatial_shape[2], 0)

        h_l, h_r = h_pad // 2, h_pad - (h_pad // 2)
        w_l, w_r = w_pad // 2, w_pad - (w_pad // 2)
        d_l, d_r = d_pad // 2, d_pad - (d_pad // 2)

        pad_tuple = (d_l, d_r, w_l, w_r, h_l, h_r)

        pad_mode_str = (
            padding_mode.value if hasattr(padding_mode, "value")
            else str(padding_mode).lower()
        )
        if any(pad_tuple):
            padded_input = torch.nn.functional.pad(batch_input, pad_tuple, mode=pad_mode_str, value=cval)
        else:
            padded_input = batch_input
            
        # Get padded dimensions
        _, _, _, padded_h, padded_w, padded_d = padded_input.shape
        
        # Initialize output tensor
        output_shape = (1, model.out_channels, padded_h, padded_w, padded_d)
        output = torch.zeros(output_shape, dtype=input.dtype, device=input.device)
        count_map = torch.zeros(output_shape, dtype=input.dtype, device=input.device)
        
        # Compute importance map for blending
        importance_map = compute_importance_map(
            get_valid_patch_size((padded_h, padded_w, padded_d), roi_size),
            mode=mode,
            sigma_scale=sigma_scale,
            device=input.device
        )
        
        # Get all sliding window positions
        scan_interval = _get_scan_interval(roi_size, (padded_h, padded_w, padded_d), overlap)
        slices = dense_patch_slices((padded_h, padded_w, padded_d), roi_size, scan_interval)
        
        # Process windows in batches
        slice_batches = [slices[i:i + sw_batch_size] for i in range(0, len(slices), sw_batch_size)]
        
        for slice_batch in slice_batches:
            # Prepare batch of temporal windows
            window_batch = []
            batch_slices = []
            
            for slice_idx, (h_slice, w_slice, d_slice) in enumerate(slice_batch):
                # Extract temporal window from all timesteps
                temporal_window = padded_input[:, :valid_t, :, h_slice, w_slice, d_slice]
                window_batch.append(temporal_window)
                batch_slices.append((h_slice, w_slice, d_slice))
            
            # Stack windows for batch processing
            if window_batch:
                # Process each window separately (since they have temporal dimension)
                for window_idx, temporal_window in enumerate(window_batch):
                    # Process through model
                    window_output = model(temporal_window, batch_seq_length, batch_dates)
                    
                    # Get the slice for this window
                    h_slice, w_slice, d_slice = batch_slices[window_idx]
                    
                    # Accumulate results with importance weighting
                    if importance_map is not None:
                        window_importance = importance_map[
                            :h_slice.stop - h_slice.start,
                            :w_slice.stop - w_slice.start,
                            :d_slice.stop - d_slice.start
                        ]
                        window_output *= window_importance
                        
                    # Add to output
                    output[:, :, h_slice, w_slice, d_slice] += window_output
                    
                    # Update count map
                    if importance_map is not None:
                        count_map[:, :, h_slice, w_slice, d_slice] += window_importance
                    else:
                        count_map[:, :, h_slice, w_slice, d_slice] += 1.0
        
        # Normalize by count map
        output = output / count_map.clamp(min=1e-5)
        
        # Remove padding if it was added
        if any(p > 0 for p in pad_tuple):
            # pad_tuple is (d_l, d_r, w_l, w_r, h_l, h_r)
            d_l, d_r, w_l, w_r, h_l, h_r = pad_tuple

            # Crop in the correct order: H, W, D dimensions
            output = output[
                :, :,
                h_l:(output.shape[2] - h_r) if h_r > 0 else output.shape[2],
                w_l:(output.shape[3] - w_r) if w_r > 0 else output.shape[3],
                d_l:(output.shape[4] - d_r) if d_r > 0 else output.shape[4]
            ]
        
        outputs.append(output)
    
    return torch.cat(outputs, dim=0)


def _get_scan_interval(
        roi_size: Tuple[int, int, int],
        image_size: Tuple[int, int, int],
        overlap: float
) -> Tuple[int, int, int]:
    """
    Compute scan interval for sliding windows.
    """
    scan_interval = []
    for i in range(3):
        interval = int(roi_size[i] * (1 - overlap))
        scan_interval.append(max(interval, 1))
    return tuple(scan_interval)


def inference(
        model: torch.nn.Module,
        input: torch.Tensor,
        seq_lengths: torch.Tensor,
        dates: torch.Tensor,
        use_sliding_window: bool = True,
        roi_size: Union[Tuple[int, int, int], int] = (64, 64, 64),
        sw_batch_size: int = 1,
        overlap: float = 0.5,
        implementation: Literal["simple", "proper"] = "proper",
        **kwargs
) -> torch.Tensor:
    """
    Main inference function with optional sliding window.
    
    Args:
        model: TemporalUNETR model
        input: Input tensor (B, T, C, H, W, D)
        seq_lengths: Sequence lengths (B,)
        dates: Dates tensor (B, T)
        use_sliding_window: Whether to use sliding window inference
        roi_size: Size of sliding window
        sw_batch_size: Batch size for sliding windows
        overlap: Overlap between windows
        implementation: Which sliding window implementation to use
                       "simple" - simplified version using last timestep
                       "proper" - full temporal window extraction
        **kwargs: Additional arguments for sliding window inference
        
    Returns:
        Model output tensor (B, C, H, W, D)
        
    Example:
        >>> # For inference on 128x128x128 volumes with 64x64x64 patches
        >>> output = inference(
        ...     model=model,
        ...     input=input_sequence,  # (B, T, C, 128, 128, 128)
        ...     seq_lengths=seq_lengths,
        ...     dates=dates,
        ...     use_sliding_window=True,
        ...     roi_size=(64, 64, 64),
        ...     overlap=0.25,
        ...     sw_batch_size=4
        ... )
    """
    
    if use_sliding_window:
        if implementation == "proper":
            return proper_temporal_sliding_window_inference(
                model=model,
                input=input,
                seq_lengths=seq_lengths,
                dates=dates,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                **kwargs
            )
        else:
            return temporal_sliding_window_inference(
                model=model,
                input=input,
                seq_lengths=seq_lengths,
                dates=dates,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                **kwargs
            )
    else:
        # Default full-volume inference
        return model(input, seq_lengths, dates)