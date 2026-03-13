"""
Temporal UNETR (TemporalUNETR) Implementation

This module implements a temporal extension of the UNETR (UNet Transformers) architecture
for processing sequential 3D medical images. The implementation combines Vision Transformers
(ViT) with temporal attention mechanisms to capture both spatial and temporal dependencies
in longitudinal medical imaging data.

Key Components:
    - TemporalTransformerBlock: Applies multi-head self-attention across the temporal dimension
    - TemporalSpatialEmbedding: Combines ViT spatial encoding with temporal transformer blocks
    - TemporalUNETR: Complete encoder-decoder architecture with temporal fusion strategies

The architecture supports:
    - Variable-length temporal sequences with padding masks
    - Multiple temporal aggregation methods (last, mean, max, concatenation)
    - Optional temporal encoders (MLP, positional, dual)
    - Pretrained ViT weights for transfer learning
    - Flexible skip connection strategies

Typical Use Cases:
    - Longitudinal tumor tracking and segmentation
    - Disease progression analysis in sequential scans
    - Multi-timepoint medical image analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.vit import ViT
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import (
    UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
)
from monai.utils import ensure_tuple_rep
from typing import Sequence, Literal
from src.networks.temporal_encoders import (
    MLPEncoder, 
    BiasedPositionalEncoder,
    DualEncoder
)

class TemporalTransformerBlock(nn.Module):
    """
    Transformer block for processing temporal sequences using multi-head self-attention.
    
    This block applies temporal attention across time steps for each spatial location,
    optionally followed by a feed-forward MLP block. Uses pre-normalization (LayerNorm
    before attention/MLP) and residual connections.
    """
    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1,
            qkv_bias: bool = False,
            use_mlp: bool = True
    ) -> None:
        """
        Initialize the temporal transformer block.
        
        Args:
            embed_dim : int : embedding dimension for the transformer
            num_heads : int : number of attention heads (default: 8)
            dropout : float : dropout probability for attention and MLP (default: 0.1)
            qkv_bias : bool : whether to use bias in query, key, value projections (default: False)
            use_mlp : bool : whether to include the MLP feed-forward block (default: True)
        
        Returns:
            None
        """
        super().__init__()

        self.num_heads = num_heads

        # Attention block
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            bias=qkv_bias,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # MLP Block
        self.use_mlp = use_mlp
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Forward pass through the temporal transformer block.
        
        Applies temporal self-attention followed by optional MLP with residual connections
        and layer normalization. Attention is masked to ignore padded timesteps.
        
        Args:
            x : torch.Tensor : input tensor of shape (B*N, T, D) where B is batch size,
                               N is number of spatial tokens, T is temporal sequence length,
                               and D is embedding dimension
            attn_mask : torch.Tensor | None : attention mask of shape (B*num_heads, T, T)
                                               with -inf for positions to mask and 0.0 for valid positions,
                                               or None if no masking needed
        
        Returns:
            x : torch.Tensor : output tensor of shape (B*N, T, D) after attention and MLP
        """

        # Apply temporal attention for each spatial location
        x = self.norm1(x)
        z, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=False
        )
        x = x + z

        # Apply mlp block if required
        if self.use_mlp:
            x = x + self.mlp(self.norm2(x))

        return x

class TemporalSpatialEmbedding(nn.Module):
    """
    Combined spatial-temporal embedding network using ViT and temporal transformers.
    
    Processes sequential 3D volumes by first extracting spatial features with a Vision
    Transformer, then applying temporal attention across time steps. Supports optional
    temporal encoders (MLP, positional, or dual) to incorporate temporal information
    such as time intervals between scans.
    """
    def __init__(
            self,
            in_channels: int,
            img_size: Sequence[int] | int,
            patch_size: Sequence[int],
            num_layers: int,
            embed_dim: int,
            mlp_dim: int,
            num_heads: int,
            temporal_depth: int,
            temporal_heads: int = 8,
            use_temporal_mlp: bool = True,
            use_temporal_encoder: Literal["mlp", "position", "dual"] | None = None,
            proj_type: str = "conv",
            dropout: float = 0.0,
            spatial_dims: int = 3,
            vit_from_pretrained: str | None = None
    ) -> None:
        """
        Initialize the temporal-spatial embedding network.
        
        Args:
            in_channels : int : number of input channels
            img_size : Sequence[int] | int : spatial dimensions of input images
            patch_size : Sequence[int] : size of patches for ViT tokenization
            num_layers : int : number of transformer layers in the spatial ViT
            embed_dim : int : embedding dimension for transformers
            mlp_dim : int : hidden dimension for MLP blocks in ViT
            num_heads : int : number of attention heads in spatial ViT
            temporal_depth : int : number of temporal transformer blocks
            temporal_heads : int : number of attention heads in temporal transformers (default: 8)
            use_temporal_mlp : bool : whether to use MLP in temporal blocks (default: True)
            use_temporal_encoder : Literal["mlp", "position", "dual"] | None : type of temporal
                                   encoder to use for date/time information (default: None)
            proj_type : str : type of projection for ViT patch embedding, "conv" or "perceptron" (default: "conv")
            dropout : float : dropout probability (default: 0.0)
            spatial_dims : int : number of spatial dimensions, 2 or 3 (default: 3)
            vit_from_pretrained : str | None : path to pretrained ViT weights (default: None)
        
        Returns:
            None
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.temporal_heads = temporal_heads
        self.use_temporal_encoder = use_temporal_encoder

        self.spatial_encoder = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=embed_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=False,
            dropout_rate=dropout,
            spatial_dims=spatial_dims,
            qkv_bias=False,
            save_attn=False
        )

        # Load pretrained vit weights
        if vit_from_pretrained:
            self.spatial_encoder.load_state_dict(torch.load(vit_from_pretrained, weights_only=True))

        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(
                embed_dim=embed_dim, 
                num_heads=temporal_heads, 
                dropout=dropout, 
                use_mlp=use_temporal_mlp
            )
            for _ in range(temporal_depth)
        ])

        # Additional temporal information
        if use_temporal_encoder == "mlp":
            self.temporal_encoder = MLPEncoder(embed_dim)
        elif use_temporal_encoder == "position":
            self.temporal_encoder = BiasedPositionalEncoder(embed_dim)
        elif use_temporal_encoder == "dual":
            self.temporal_encoder = DualEncoder(embed_dim)

    def forward(
            self,
            x: torch.Tensor,
            seq_lengths: torch.Tensor,
            dates: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for sequential 3D volumes with spatial and temporal encoding.
        
        Processes each timestep with spatial ViT, applies optional temporal encoding,
        then processes temporal relationships with temporal transformer blocks. Handles
        variable-length sequences with padding masks.
        
        Args:
            x : torch.Tensor : input tensor of shape (B, T, C, H, W, D) where B is batch size,
                               T is temporal sequence length, C is channels, H is height,
                               W is width, D is depth
            seq_lengths : torch.Tensor : actual sequence lengths of shape (B,), values indicate
                                          number of valid timesteps per batch element
            dates : torch.Tensor | None : temporal metadata of shape (B, T) for temporal encoders,
                                          typically containing time intervals or timestamps (default: None)
        
        Returns:
            out : torch.Tensor : temporally-encoded features of shape (B, T, N, E) where N is
                                 number of spatial tokens and E is embedding dimension
            skip_features : list[torch.Tensor] : list of intermediate spatial features from ViT
                                                 layers, each of shape (B*T, N, E)
        """
        # Get the shape
        B, T, C, H, W, D = x.shape

        # Flatten the batch and time dimensions for spatial processing
        x = x.reshape(B * T, C, D, H , W)

        # Encode with spatial encoder
        x_encoded, skip_features = self.spatial_encoder(x)

        # Get tokens and embeddings dim
        N, E = x_encoded.shape[1], x_encoded.shape[2]

        # Add temporal encoding if it is flagged
        if self.use_temporal_encoder is not None and dates is not None:
            # Reshape for temporal encoder
            x_encoded = x_encoded.reshape(B, T, N, E)

            # Pass through encoder
            x_temporal = self.temporal_encoder(x_encoded, dates)

            # Reshape
            x_temporal = x_temporal.reshape(B*T, N, E)
        else:
            x_temporal = x_encoded

        # Reshape to seprate time
        x_temporal = x_temporal.reshape(B, T, N, E)

        # Permute and reshape to (B*N, T, D)
        x_temporal = x_temporal.permute(0, 2, 1, 3).reshape(B * N, T, E)

        # Create attention mask if needed
        attn_mask = None
        if seq_lengths is not None:
            # Repeat seq_lengths for each spatial position
            seq_lengths_expanded = seq_lengths.repeat_interleave(N)

            # Create attention mask
            padding_mask = self._create_padding_mask(
                seq_lengths_expanded, T, x.device
            )
            attn_mask = self._create_attn_mask(padding_mask)

        # Pass through temporal blocks    
        for temporal_block in self.temporal_blocks:
            x_temporal = temporal_block(x_temporal, attn_mask)

        # Rearrange and reshape back
        x_temporal = x_temporal.reshape(B, N, T, E).permute(0, 2, 1, 3).reshape(B * T, N, E)

        # Reshape with embed_dim last
        out = x_temporal.reshape(B, T, -1, E)
        
        return out, skip_features
    
    def _create_padding_mask(
            self,
            seq_lengths: torch.Tensor,
            max_length: int,
            device: torch.device
    ) -> torch.Tensor:
        """
        Create a boolean padding mask for variable length sequences.
        
        Generates a mask where True indicates valid timesteps and False indicates padding.
        This is used to prevent attention to padded positions in temporal transformers.
        
        Args:
            seq_lengths : torch.Tensor : actual sequence lengths of shape (B*N,) where B is
                                          batch size and N is number of spatial tokens
            max_length : int : maximum sequence length (T) to generate mask for
            device : torch.device : device to create the mask on
        
        Returns:
            mask : torch.Tensor : boolean mask of shape (B*N, T) where True indicates valid
                                  positions and False indicates padding
        """
        # Get batch size
        batch_size = seq_lengths.shape[0]

        # Create mask
        mask = torch.arange(max_length, device=device).expand(batch_size, max_length)
        mask = mask < seq_lengths.unsqueeze(1)

        return mask
    
    def _create_attn_mask(
            self,
            padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert boolean padding mask to additive attention mask for multi-head attention.
        
        Transforms a boolean mask into the format expected by PyTorch's MultiheadAttention,
        where 0.0 allows attention and -inf blocks attention. Expands the mask for all
        attention heads.
        
        Args:
            padding_mask : torch.Tensor : boolean padding mask of shape (B*N, T) where True
                                          indicates valid positions and False indicates padding
        
        Returns:
            attn_mask : torch.Tensor : attention mask of shape (num_heads*B*N, T, T) with 0.0
                                       for valid attention positions and -inf for masked positions
        """
        # Get attention mask
        T = padding_mask.shape[1]

        # Create (B, T, T) mask
        attn_mask = padding_mask.unsqueeze(1).expand(-1, T, -1)

        # Fill in mask
        attn_mask = torch.where(attn_mask, 0.0, float('-inf'))
        
        # Expand with the number of heads
        attn_mask = attn_mask.repeat(self.temporal_heads, 1, 1)

        return attn_mask
        
class TemporalUNETR(nn.Module):
    """
    Temporal UNETR architecture for sequential 3D medical image segmentation.
    
    Extends the UNETR architecture to handle temporal sequences by combining a ViT-based
    spatial encoder with temporal attention mechanisms. Uses a U-Net decoder with skip
    connections from both spatial and temporal features. Supports multiple strategies
    for aggregating temporal information (last, mean, max, concatenation).
    
    The architecture follows an encoder-decoder structure:
        - Encoder: TemporalSpatialEmbedding processes sequences spatially and temporally
        - Decoder: U-Net style decoder with skip connections at multiple resolutions
        - Aggregation: Flexible temporal fusion before decoding
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: Sequence[int] | int,
            embedding_path: str | None = None,
            patch_size: int = 16,
            feature_size: int = 16,
            embed_dim: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            temporal_depth: int = 8,
            use_temporal_mlp: bool = True,
            use_temporal_encoder: Literal["mlp", "position", "dual"] | None = None,
            proj_type: str = "conv",
            norm_name: tuple | str = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            dropout: float = 0.0,
            spatial_dims: int = 3,
            aggregation_method: Literal["last", "mean", "max", "cat"] = "last",
            vit_from_pretrained: str | None = None
    ) -> None:
        """
        Initialize the Temporal UNETR model.
        
        Args:
            in_channels : int : number of input channels
            out_channels : int : number of output channels for segmentation
            img_size : Sequence[int] | int : spatial dimensions of input images
            embedding_path : str | None : path to pretrained embedding weights (default: None)
            patch_size : int : size of patches for ViT tokenization (default: 16)
            feature_size : int : base number of features for decoder (default: 16)
            embed_dim : int : embedding dimension for transformers (default: 768)
            mlp_dim : int : hidden dimension for MLP blocks in ViT (default: 3072)
            num_heads : int : number of attention heads in spatial ViT (default: 12)
            temporal_depth : int : number of temporal transformer blocks (default: 8)
            use_temporal_mlp : bool : whether to use MLP in temporal blocks (default: True)
            use_temporal_encoder : Literal["mlp", "position", "dual"] | None : type of temporal
                                   encoder for date/time information (default: None)
            proj_type : str : type of projection for ViT patch embedding (default: "conv")
            norm_name : tuple | str : normalization type for U-Net blocks (default: "instance")
            conv_block : bool : whether to use convolutional blocks in U-Net (default: True)
            res_block : bool : whether to use residual blocks in U-Net (default: True)
            dropout : float : dropout probability (default: 0.0)
            spatial_dims : int : number of spatial dimensions, 2 or 3 (default: 3)
            aggregation_method : Literal["last", "mean", "max", "cat"] : method for temporal
                                 aggregation ("last": use last valid timestep, "mean": average
                                 valid timesteps, "max": max pool timesteps, "cat": concatenate
                                 mean and last) (default: "last")
            vit_from_pretrained : str | None : path to pretrained ViT weights (default: None)
        
        Returns:
            None
        
        Raises:
            ValueError : if dropout is not in [0, 1] or embed_dim is not divisible by num_heads
        """
        super().__init__()

        # Check for dropout and dropout mismatches
        if not (0 <= dropout <= 1):
            raise ValueError(f"`dropout` {dropout} should be between 0 and 1.")
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"`embed_dim` {embed_dim} should be divisble by `num_heads` {num_heads}.")
        
        # Set attributes
        self.out_channels = out_channels
        self.num_layers = 12
        self.temporal_depth = temporal_depth
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.feat_size = tuple(
            img_d // p_d for img_d, p_d in zip(img_size, self.patch_size)
        )
        self.embed_dim = embed_dim
        self.aggregate_method = aggregation_method

        # Set the decoder 5 input dimension by whether we concatenate
        self.decoder5_in_dim = embed_dim * 2 if aggregation_method == "cat" else embed_dim
        
        # Embedding Space
        self.embedding_space = TemporalSpatialEmbedding(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            temporal_depth=self.temporal_depth,
            use_temporal_mlp=use_temporal_mlp,
            use_temporal_encoder=use_temporal_encoder,
            proj_type=proj_type,
            dropout=dropout,
            spatial_dims=spatial_dims,
            vit_from_pretrained=vit_from_pretrained
        )

        # Load the embedding if required
        if embedding_path:
            self.embedding_space.load_state_dict(torch.load(embedding_path))

        # Encoders
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.embed_dim,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.embed_dim,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.embed_dim,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block
        )

        # Decoders
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.decoder5_in_dim,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block
        )

        # Output
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels
        )
    
        # Projections for ensuring the correct input to Unet
        self.proj_axes = (0, spatial_dims +1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.embed_dim]
    
    def proj_feat(
            self,
            x: torch.Tensor,
            cat: bool = False
    ) -> torch.Tensor:
        """
        Reshape and permute features from token format to spatial format for U-Net processing.
        
        Converts flattened spatial tokens back to spatial dimensions and reorders dimensions
        to match U-Net input format (batch, channels, spatial_dims...).
        
        Args:
            x : torch.Tensor : input features of shape (B, N, E) where B is batch size,
                               N is number of spatial tokens, E is embedding dimension
            cat : bool : whether features have been concatenated (doubles embedding dimension)
                         (default: False)
        
        Returns:
            x : torch.Tensor : reshaped features of shape (B, E, F_1, F_2, ..., F_n) where
                               E is embedding dimension and F_i are feature map spatial dimensions
        """
        new_view = [x.shape[0]] + self.proj_view_shape[:-1] + [self.decoder5_in_dim] if cat else [x.shape[0]] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x
    
    def forward(
            self,
            x: torch.Tensor,
            seq_lengths: torch.Tensor,
            dates: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through the Temporal UNETR architecture.
        
        Processes sequential 3D volumes through spatial-temporal encoding, aggregates
        temporal information, and decodes to produce a segmentation map. Uses skip
        connections from intermediate encoder layers.
        
        Args:
            x : torch.Tensor : input sequential volumes of shape (B, T, C, H, W, D) where B is
                               batch size, T is temporal sequence length, C is input channels,
                               H is height, W is width, D is depth
            seq_lengths : torch.Tensor : actual sequence lengths of shape (B,), indicates number
                                          of valid timesteps per batch element
            dates : torch.Tensor | None : temporal metadata of shape (B, T) for temporal encoders
                                          (default: None)
        
        Returns:
            out : torch.Tensor : segmentation output of shape (B, out_channels, H, W, D) where
                                 out_channels is the number of segmentation classes
        """
        # Get shapes
        B, T, C, H, W, D = x.shape

        # Pass through embedding x_1 shape (B, T, N, E)
        x_1, skip_features = self.embedding_space(x, seq_lengths, dates)

        # Determine method of temporal fusion
        if self.aggregate_method == "last":
            x_1 = self.get_last_valid_features(x_1, seq_lengths)
        elif self.aggregate_method == "mean":
            x_1 = self.aggregate_mean(x_1, seq_lengths)
        elif self.aggregate_method == "max":
            x_1 = torch.max(x_1, dim=1).values
        elif self.aggregate_method == "cat":
            avg_features = self.aggregate_mean(x_1, seq_lengths)
            last_features = self.get_last_valid_features(x_1, seq_lengths)
            x_1 = torch.cat([avg_features, last_features], dim=-1)
        else:
            raise ValueError(f"aggregation method `{self.aggregate_method}` must be one of `last`, `mean`.")
        
        # Collect the final skip features
        x_2 = skip_features[3].reshape(B, T, -1, self.embed_dim)
        x_2 = self.get_last_valid_features(x_2, seq_lengths)
        x_3 = skip_features[6].reshape(B, T, -1, self.embed_dim)
        x_3 = self.get_last_valid_features(x_3, seq_lengths)
        x_4 = skip_features[9].reshape(B, T, -1, self.embed_dim)
        x_4 = self.get_last_valid_features(x_4, seq_lengths)

        # Encode
        last_input = self.get_last_valid_features(x, seq_lengths)
        encoded_1 = self.encoder1(last_input)
        encoded_2 = self.encoder2(self.proj_feat(x_2))
        encoded_3 = self.encoder3(self.proj_feat(x_3))
        encoded_4 = self.encoder4(self.proj_feat(x_4))

        # Decode
        decoded_4 = self.proj_feat(x_1, cat=True) if self.aggregate_method == "cat" else self.proj_feat(x_1) 
        decoded_3 = self.decoder5(decoded_4, encoded_4)
        decoded_2 = self.decoder4(decoded_3, encoded_3)
        decoded_1 = self.decoder3(decoded_2, encoded_2)

        # Output
        out = self.decoder2(decoded_1, encoded_1)
        out = self.out(out)
        
        return out
    
    def get_last_valid_features(
            self,
            features: torch.Tensor, 
            seq_lengths: torch.Tensor, 
            temporal_dim: int = 1
    ) -> torch.Tensor:
        """
        Extract features from the last valid timestep for each sequence in the batch.
        
        For variable-length sequences, retrieves the features at the last non-padded
        position based on the actual sequence length. Useful for aggregating temporal
        information by selecting the most recent valid timestep.
        
        Args:
            features : torch.Tensor : input features with temporal dimension, typically of shape
                                      (B, T, ...) where B is batch size and T is temporal length
            seq_lengths : torch.Tensor : actual sequence lengths of shape (B,), indicates the
                                          position of last valid timestep for each sequence
            temporal_dim : int : dimension index for the temporal axis (default: 1)
        
        Returns:
            result : torch.Tensor : features at last valid timestep of shape (B, ...) with
                                    temporal dimension removed
        
        Raises:
            ValueError : if temporal_dim is not 1 (other dimensions not implemented)
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Create indices for gathering
        indices = (seq_lengths - 1).long()  # Last valid index for each sequence
        
        # Handle different tensor shapes
        if temporal_dim == 1:  # Shape like (B, T, ...)
            indices = indices.view(batch_size, 1, *([1] * (len(features.shape) - 2)))
            indices = indices.expand(batch_size, 1, *features.shape[2:])
            result = torch.gather(features, temporal_dim, indices).squeeze(temporal_dim)
        else:
            raise ValueError(f"temporal dim {temporal_dim} not implemented.")
        
        return result
    
    def aggregate_mean(
            self,
            features: torch.Tensor,
            seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the mean of features over valid timesteps, ignoring padding.
        
        Aggregates temporal features by averaging only the valid (non-padded) timesteps
        for each sequence. Uses masking to exclude padded positions from the mean calculation.
        
        Args:
            features : torch.Tensor : input temporal features of shape (B, T, N, E) where B is
                                      batch size, T is temporal length, N is number of spatial
                                      tokens, E is embedding dimension
            seq_lengths : torch.Tensor : actual sequence lengths of shape (B,), indicates number
                                          of valid timesteps to average
        
        Returns:
            mean : torch.Tensor : averaged features of shape (B, N, E) with temporal dimension
                                  aggregated via mean over valid timesteps
        """
        B, T = features.shape[0], features.shape[1]

        # Create the mask
        mask = (torch.arange(T, device=features.device).expand(B, T) < seq_lengths.unsqueeze(1)).float()

        # Expand to match the sizes
        while len(mask.shape) < len(features.shape):
            mask = mask.unsqueeze(-1)
        
        # Apply mask and sum
        summed_features = torch.sum(features * mask, dim=1)

        # Divide for mean
        denom = torch.sum(mask, dim=1).clamp_min(1.0)

        return summed_features / denom
    
    def save_embedding(
            self,
            path: str
    ) -> None:
        """
        Save the embedding space state dictionary to disk.
        
        Saves only the TemporalSpatialEmbedding component weights, useful for transfer
        learning or loading pretrained encoders separately from the full model.
        
        Args:
            path : str : file path where the embedding state dictionary will be saved
        
        Returns:
            None
        """
        torch.save(self.embedding_space.state_dict(), path)