from typing import Union, Optional
import torch.nn as nn
import torch
import math

class ViTEncoder(nn.Module):
    def __init__(self, vit_model,
                 image_size: int = 224,
                 patch_size: int = 16,
                 stride: int = 16,
                 hidden_dim: int = 768,
                 mlp_dim: int = 3072,
                 **kwargs):
        super().__init__()
        self.vit = vit_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

        # modify patchification process (for smaller resolution datasets)
        self.conv_proj = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.stride
        )
        # Initialize the class token, position embeddings, and patch projection
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self._initialize_pos_embedding()
        self._init_patch_proj()

    def _init_patch_proj(self):
        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

    def _initialize_pos_embedding(self):
        # Initialize position embeddings
        num_patches = ((self.image_size - self.patch_size) // self.stride + 1)**2
        seq_length = num_patches + 1  # +1 for class token
        self.vit.encoder.pos_embedding = nn.Parameter(torch.empty(1, seq_length, self.hidden_dim).normal_(std=0.02))  # from BERT

    def forward(self, x):
        # Process input to patch embeddings and permute
        x = self.conv_proj(x)  # shape (batch_size, hidden_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # shape (batch_size, num_patches, hidden_dim)
        n = x.shape[0]
        # Add class token
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # Pass through transformer encoder layers
        x = self.vit.encoder(x)
        # Extract CLS token embedding as feature vector
        cls_embedding = x[:, 0]  # shape (batch_size, hidden_dim)
        return cls_embedding