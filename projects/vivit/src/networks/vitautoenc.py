"""
Vision Transformer Autoencoder Module

This module implements a Vision Transformer (ViT) based autoencoder architecture
for medical image processing. The ViTAutoEnc class combines a ViT encoder with
transposed convolution layers for decoding, enabling reconstruction tasks on
volumetric medical images.

The implementation leverages MONAI's ViT implementation and provides functionality
for encoding images into a latent representation and decoding them back to the
original spatial dimensions through learnable upsampling operations.

Key Components:
    - ViT encoder for patch-based image encoding
    - Transposed convolution decoder for spatial reconstruction
    - Support for 2D and 3D medical images
    - Configurable patch sizes, hidden dimensions, and network depth

Dependencies:
    - torch
    - monai.networks
"""

import torch
import math
from typing import Sequence
from torch import nn
from monai.networks.layers import Conv
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep, is_sqrt

class ViTAutoEnc(nn.Module):
    """
    Vision Transformer Autoencoder for medical image reconstruction.
    
    This class implements an autoencoder architecture using a Vision Transformer (ViT)
    as the encoder and transposed convolutional layers as the decoder. The model processes
    images by dividing them into patches, encoding them through transformer layers, and
    reconstructing the original spatial dimensions through learned upsampling.
    
    Args:
        in_channels : int : Number of input channels in the image
        img_size : Sequence[int] | int : Spatial dimensions of the input image 
            (e.g., (96, 96, 96) for 3D or 224 for 2D)
        patch_size : Sequence[int] | int : Size of patches to divide the image into 
            (e.g., (16, 16, 16) for 3D or 16 for 2D)
        out_channels : int : Number of output channels (default: 1)
        deconv_chns : int : Number of channels in the intermediate decoder layer (default: 16)
        hidden_size : int : Dimensionality of the transformer hidden states (default: 768)
        mlp_dim : int : Dimensionality of the MLP layer in transformer blocks (default: 3072)
        num_layers : int : Number of transformer encoder layers (default: 12)
        num_heads : int : Number of attention heads in each transformer layer (default: 12)
        proj_type : str : Type of projection layer for patch embedding, either "conv" or "perceptron" (default: "conv")
        dropout_rate : float : Dropout probability for transformer layers (default: 0.0)
        spatial_dims : int : Number of spatial dimensions (2 for 2D, 3 for 3D) (default: 3)
        qkv_bias : bool : Whether to include bias in QKV projections (default: False)
        save_attn : bool : Whether to save attention weights during forward pass (default: False)
    
    Returns:
        None
    """
    def __init__(
            self,
            in_channels: int,
            img_size: Sequence[int] | int,
            patch_size: Sequence[int] | int,
            out_channels: int = 1,
            deconv_chns: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_layers: int = 12,
            num_heads: int = 12,
            proj_type: str = "conv",
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            qkv_bias: bool = False,
            save_attn: bool = False
    ) -> None:
        super().__init__()

        self.spatial_dims= spatial_dims
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=False,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn
        )

        conv_trans = Conv[Conv.CONVTRANS, self.spatial_dims]
        up_kernel_size = [int(math.sqrt(i)) for i in self.patch_size]
        self.conv3d_transpose = conv_trans(
            hidden_size,
            deconv_chns,
            kernel_size=up_kernel_size,
            stride=up_kernel_size
        )
        self.conv3d_transpose_1 = conv_trans(
            in_channels=deconv_chns,
            out_channels=out_channels,
            kernel_size=up_kernel_size,
            stride=up_kernel_size
        )
    
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the ViT autoencoder.
        
        Processes the input image through the ViT encoder to obtain latent representations,
        then reconstructs the image through transposed convolution layers. The method handles
        reshaping of the patch-based encoder output back to spatial dimensions.
        
        Args:
            x : torch.Tensor : Input image tensor of shape (B, C, H, W) for 2D or 
                (B, C, D, H, W) for 3D, where B is batch size, C is channels, 
                and D, H, W are spatial dimensions
        
        Returns:
            out : torch.Tensor : Reconstructed image tensor with same spatial dimensions as input,
                shape (B, out_channels, H, W) for 2D or (B, out_channels, D, H, W) for 3D
            hidden_states : list[torch.Tensor] : List of hidden state tensors from each 
                transformer layer, each of shape (B, num_patches, hidden_size), where 
                num_patches = (D//patch_size[0]) * (H//patch_size[1]) * (W//patch_size[2]) for 3D
        """
        spatial_size = x.shape[2:]
        out, hidden_states = self.vit(x)
        out = out.transpose(1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        out = torch.reshape(out, [out.shape[0], out.shape[1], *d])
        out = self.conv3d_transpose(out)
        out = self.conv3d_transpose_1(out)
        
        return out, hidden_states

    def save_vit_weights(
            self,
            path: str
    ) -> None:
        """
        Save the Vision Transformer encoder weights to a file.
        
        This method saves only the ViT encoder portion of the model (excluding the
        decoder layers) to the specified path. Useful for transfer learning or
        model checkpointing when only the encoder weights are needed.
        
        Args:
            path : str : File path where the ViT weights will be saved 
                (e.g., "model_weights.pth")
        
        Returns:
            None
        """
        torch.save(self.vit.state_dict(), path)