import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .swin import SwinBlockSequence

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """
        Multi-Head Attention Module.
        
        Args:
            embed_dim (int): Dimension of the input and output embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for the attention mechanism.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embed dimension must be divisible by the number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per head
        self.scale = self.head_dim ** -0.5     # Scaling factor for softmax
        
        # Learnable linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            query (Tensor): Shape (B, N, D)
            key (Tensor): Shape (B, M, D)
            value (Tensor): Shape (B, M, D)
            mask (Tensor, optional): Shape (B, 1, 1, M) or (B, N, M). Default is None.
        
        Returns:
            Tensor: Attention output of shape (B, N, D)
        """
        B, N, _ = query.shape
        _, M, _ = key.shape

        # Linear projections
        Q = self.q_proj(query)  # (B, N, D)
        K = self.k_proj(key)    # (B, M, D)
        V = self.v_proj(value)  # (B, M, D)

        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        K = K.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, M, head_dim)
        V = V.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, M, head_dim)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, N, M)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))  # Apply mask
        attn_weights = F.softmax(attn_weights, dim=-1)  # Normalize across the last dimension
        attn_weights = self.dropout(attn_weights)       # Apply dropout
        
        # Attention output
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, N, head_dim)

        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).reshape(B, N, self.embed_dim)  # (B, N, D)
        output = self.out_proj(attn_output)  # Final linear projection

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):
        """
        Transformer Block with Multi-Head Attention, Feed Forward Network, Layer Normalization and Residual Connection.
        
        Args:
            embed_dim (int): Dimension of the input and output embeddings.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        # Multi-Head Attention
        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer Normalization
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        """
        Forward pass for the Transformer block.
        
        Args:
            x (Tensor): Input tensor of shape (B, N, D), where B is batch size, N is sequence length, and D is embedding dimension.
            mask (Tensor, optional): Mask tensor of shape (B, 1, 1, N) or (B, N, N). Default is None.
        
        Returns:
            Tensor: Output tensor of shape (B, N, D)
        """
        # Multi-Head Attention + Residual Connection + LayerNorm
        attn_output = self.attn(x, x, x, mask)  # (B, N, D)
        x = self.layernorm1(x + attn_output)  # Add & Norm

        # Feed Forward Network + Residual Connection + LayerNorm
        ff_output = self.ffn(x)  # (B, N, D)
        x = self.layernorm2(x + ff_output)  # Add & Norm
        
        return x
    
    
class ST_Transformer(nn.Module):
    """
    Args:
        feature_channels: 384
        sequence_len: 3
        num_heads: [3, 6, 12, 24]
        window_size: [10, 10, 10, 10]
    """

    def __init__(self,
                 feature_channels,
                 sequence_len,
                 num_heads,
                 window_size):
        super().__init__()
        self._is_init = False
        
        self.feature_channels = feature_channels
        self.sequence_len = sequence_len

        self.Temporal_blocks = nn.ModuleList()
        self.Spatial_blocks = nn.ModuleList()
        
        blocks_num = len(num_heads)
        
        for i in range(blocks_num):     
            Temporal_block = TransformerBlock(embed_dim=feature_channels, num_heads=num_heads[i], ff_dim=feature_channels)
            
            Spatial_block = SwinBlockSequence(embed_dims=feature_channels, num_heads=num_heads[i], 
                                              feedforward_channels=feature_channels,
                                              depth=2, window_size=window_size[i])
            self.Temporal_blocks.append(Temporal_block)
            self.Spatial_blocks.append(Spatial_block)
            
        self.fusion = nn.Linear(feature_channels*sequence_len, feature_channels)
    
        
    def forward(self, x, T):
        '''
        x: (B, N, T*C, H, W) 
        
        '''
        x = x.permute(0, 1, 3, 4, 2).contiguous()                               # (B, N, H, W, T*C) 
        
        B, N, H, W, C = x.shape
        C = C // T
        
        x = x.view(B * N * H * W, T, C)                                         # (B*N*H*W, T, C)
        
        for Temporal_block in self.Temporal_blocks:                             
            x = Temporal_block(x)                                               # (B*N*H*W, T, C)
            
        x = x.view(B*N, H*W, T*C)                                               # (B*N, H*W, T*C)
        x = self.fusion(x)                                                      # (B*N, H*W, C)
        
        for Spatial_block in self.Spatial_blocks:
            x = Spatial_block(x, hw_shape=(H, W))                               # (B*N, H*W, C)
            
        x = x.permute(0, 2, 1).contiguous().view(B, N, C, H, W)                 # (B, N, C, H, W) 
            
        return x


# def initialize_st_transformer(model):
#     """initialize ST_Transformer model parameters."""
#     for module in model.modules():
#         if isinstance(module, nn.Linear):  # linear layer
#             trunc_normal_init(module, std=0.02)  # truncated normal initialization
#         elif isinstance(module, nn.LayerNorm):  # layer normalization
#             constant_init(module, val=1.0, bias=0.0)  # weight=1, bias=0
#         elif hasattr(module, 'relative_position_bias_table'):  # relative position bias
#             trunc_normal_init(module, std=0.02)
