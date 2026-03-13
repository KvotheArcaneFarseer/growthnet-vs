"""
Temporal Encoders Module

This module provides temporal encoding mechanisms for time-series data in neural networks.
It includes three main encoder types:

1. MLPEncoder: Uses a multi-layer perceptron to encode temporal deltas between timesteps
2. BiasedPositionalEncoder: Implements sinusoidal positional encoding with learnable phase shifts
   based on temporal deltas between observations
3. DualEncoder: Combines both MLP and positional encoding approaches in a configurable order

These encoders are designed to work with spatiotemporal data where observations occur at
irregular time intervals, adding temporal awareness to spatial embeddings.
"""

import torch
import torch.nn as nn
import math

class MLPEncoder(nn.Module):
    """
    MLP-based temporal encoder that learns to encode time deltas between observations.
    
    This encoder processes the time differences between consecutive timesteps using a
    multi-layer perceptron and adds the resulting temporal encoding to spatial features.
    Time deltas are normalized using a logarithmic scale based on months (360-day units).
    """
    
    def __init__(
            self,
            embed_dim: int
    ) -> None:
        """
        Initialize the MLP temporal encoder.
        
        Args:
            embed_dim : int : Embedding dimension for the temporal encoding
        
        Returns:
            None
        """
        super().__init__()

        self.mlp = nn.Sequential(
                nn.Linear(1, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, embed_dim)
            )
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(
            self,
            x: torch.Tensor,
            dates: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply MLP-based temporal encoding to input features.
        
        Computes time deltas between consecutive timesteps, normalizes them using a
        logarithmic scale, passes them through the MLP, and adds the result to the
        input features.
        
        Args:
            x : torch.Tensor : Input spatial features with shape (B, T, N, D) where
                               B is batch size, T is time steps, N is number of tokens,
                               D is embedding dimension
            dates : torch.Tensor : Timestamp values with shape (B, T) where B is batch
                                   size and T is time steps
        
        Returns:
            x : torch.Tensor : Temporally-encoded features with shape (B, T, N, D)
        """
        # Preprocess dates (B, T)
        time_deltas = [dates[:, i] - dates[:, i - 1] for i in range(1, dates.shape[1])]
        time_deltas = torch.stack(time_deltas, dim=1)
        
        # Normalize time deltas (B, T, 1)
        time_deltas_norm = self._preprocess_time_deltas(time_deltas)

        # Generate temporal encodings
        temporal_enc = self.mlp(time_deltas_norm)

        # Add to spatial encodings
        temporal_enc = temporal_enc.unsqueeze(2) # (B, T, 1, D)
        x = x + (self.scale * temporal_enc)

        return x
        
    def _preprocess_time_deltas(
            self,
            time_deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize time deltas using logarithmic scale in units of months.
        
        Converts raw time delta values to a logarithmic scale where the unit is
        approximately one month (360 days). This normalization helps the model
        handle time differences across different scales more effectively.
        
        Args:
            time_deltas : torch.Tensor : Raw time differences with shape (B, T-1) where
                                         B is batch size and T-1 is number of time deltas
        
        Returns:
            normalized_deltas : torch.Tensor : Normalized time deltas with shape (B, T-1, 1)
        """
        # Clamp to prevent NaNs
        time_deltas = torch.clamp(time_deltas, min=0.0)

        return torch.log(1.0 + time_deltas.unsqueeze(-1) / 360)

class BiasedPositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoder with learnable temporal phase shifts.
    
    This encoder extends standard sinusoidal positional encoding by incorporating
    learnable phase shifts based on time deltas between observations. It uses
    sine and cosine functions at different frequencies, with phases modulated
    by temporal differences to capture irregular sampling patterns.
    """
    
    def __init__(
            self,
            embed_dim: int,
            max_len: int = 128
    ) -> None:
        """
        Initialize the biased positional encoder.
        
        Args:
            embed_dim : int : Embedding dimension for the positional encoding (must be even)
            max_len : int : Maximum sequence length to support (default: 128)
        
        Returns:
            None
        """
        super().__init__()

        # Set the position and division terms
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = (
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * 
            -(math.log(10000.0) / embed_dim)
        ).exp()

        # Register as buffer
        self.register_buffer("position", position)
        self.register_buffer("div_term", div_term)

        # Get the linear layer
        self.Wt = nn.Linear(
            1, embed_dim // 2, bias=False
        )
    
    def forward(
            self,
            x: torch.Tensor,
            dates: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply biased positional encoding with temporal phase shifts.
        
        Computes sinusoidal positional encodings where the phase is modulated by
        time deltas between consecutive observations. Handles variable-length sequences
        by padding invalid positions with the last valid time delta.
        
        Args:
            x : torch.Tensor : Input features with shape (B, T, N, D) where B is batch
                               size, T is time steps, N is number of tokens, D is
                               embedding dimension
            dates : torch.Tensor : Timestamp values with shape (B, T) where B is batch
                                   size and T is time steps
        
        Returns:
            x : torch.Tensor : Positionally-encoded features with shape (B, T, N, D)
        """
        B, T = dates.shape

        # Preprocess dates (B, T)
        time_deltas = [dates[:, i] - dates[:, i - 1] for i in range(1, T)]
        time_deltas = torch.stack(time_deltas, dim=1)

        # Get the indices with max values
        _, max_indxs = torch.max(dates, dim=1)
        max_indxs = torch.clamp(max_indxs - 1, min=0)
        last_indxs = [ [i for i in range(B)], max_indxs.tolist() ]
        last_values = time_deltas[last_indxs]

        # Fill the invalid sequences with last values
        for i in range(B):
            indx = max_indxs[i]
            time_deltas[i, indx:] = last_values[i]

        # Clamp to prevent NaNs
        time_deltas = torch.clamp(time_deltas, min=0.0)

        # Compute phase shift
        phi = self.Wt(time_deltas.unsqueeze(-1))

        # Get absolute position
        arc = (self.position[:T-1] * self.div_term).unsqueeze(0)

        # Compute position
        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=2).unsqueeze(2)

        # Add positional embedding
        x = x + pe

        return x

class DualEncoder(nn.Module):
    """
    Dual temporal encoder combining MLP and positional encoding approaches.
    
    This encoder applies both MLP-based temporal encoding and biased positional
    encoding in sequence. The order of application can be configured, allowing
    for flexibility in how temporal information is integrated into the features.
    """
    
    def __init__(
            self,
            embed_dim: int,
            max_len: int = 128,
            pe_first: bool = True
    ) -> None:
        """
        Initialize the dual encoder with both MLP and positional encoders.
        
        Args:
            embed_dim : int : Embedding dimension for the temporal encodings
            max_len : int : Maximum sequence length to support (default: 128)
            pe_first : bool : If True, apply positional encoding before MLP encoding;
                              if False, apply MLP encoding first (default: True)
        
        Returns:
            None
        """
        super().__init__()

        self.pe_first = pe_first

        # Create positional and mlp encoders
        self.positional_encoder = BiasedPositionalEncoder(
            embed_dim=embed_dim, max_len=max_len
        )
        self.mlp_encoder = MLPEncoder(
            embed_dim=embed_dim
        )
    
    def forward(
            self,
            x: torch.Tensor,
            dates: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply dual temporal encoding to input features.
        
        Applies both positional and MLP encodings in the order specified by
        the pe_first parameter during initialization.
        
        Args:
            x : torch.Tensor : Input features with shape (B, T, N, D) where B is batch
                               size, T is time steps, N is number of tokens, D is
                               embedding dimension
            dates : torch.Tensor : Timestamp values with shape (B, T) where B is batch
                                   size and T is time steps
        
        Returns:
            out : torch.Tensor : Temporally-encoded features with shape (B, T, N, D)
        """
        if self.pe_first:
            out = self.mlp_encoder(self.positional_encoder(x, dates), dates)
        else:
            out = self.positional_encoder(self.mlp_encoder(x, dates), dates)
        
        return out