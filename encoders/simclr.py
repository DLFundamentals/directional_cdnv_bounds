import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Union, Optional

from encoders.base import BaseEncoder
    

class SimCLR(nn.Module):
    def __init__(
        self,
        model,
        layer: Union[str, int]  = -2,
        dataset: str = 'imagenet',
        width_multiplier: int = 1,
        pretrained: bool = False,
        hidden_dim: int = 512,
        projection_dim: int = 128,
        **kwargs
    ):
        super().__init__()

        # Wrap model based on architecture type
        if isinstance(model, models.ResNet): 
            self.encoder = ResNetEncoder(
                model, layer=layer, dataset=dataset,
                width_multiplier=width_multiplier, pretrained=pretrained
            )
        elif isinstance(model, models.VisionTransformer):
            use_old = kwargs.get('use_old', False) # TODO: remove later after fixing DCL VIT Imagenet wts
            if use_old and dataset == 'imagenet':
                self.encoder = OldViTEncoder(model) # TODO: fix this training issue for DCL ViT Imagenet
            else:
                self.encoder = ViTEncoder(
                    model,
                    image_size=kwargs.get('image_size', 224),
                    patch_size=kwargs.get('patch_size', 16),
                    stride=kwargs.get('stride', 16),
                    hidden_dim=kwargs.get('token_hidden_dim', 768),
                    mlp_dim=kwargs.get('mlp_dim', 3072)
                )
        else:
            raise NotImplementedError(f"Model {type(model)} not supported. Use ResNet or ViT.")

        # run a mock image tensor to instantiate parameters
        with torch.no_grad():
            if dataset == 'imagenet':
                h = self.encoder(torch.randn(1, 3, 224, 224))
            elif 'cifar' in dataset or dataset == 'svhn':
                h = self.encoder(torch.randn(1, 3, 32, 32))
            else:
                raise NotImplementedError(f"{dataset} not implemented")

        input_dim = h.shape[1]
        self.projector = SimCLRProjector(input_dim, hidden_dim, projection_dim)
        self.track_performance = kwargs.get('track_performance', False)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1) # flatten the tensor
        g_h = self.projector(h)
        return h, F.normalize(g_h, dim=-1)
    

# ========================== Projector ==========================

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class SimCLRProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, projection_dim=128):
        super().__init__()
        torch.manual_seed(42)

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim, bias=False),
            BatchNorm1dNoBias(projection_dim)
        )

    def forward(self, x):
        return self.projector(x)
    

# ========================== ResNet Encoder ==========================

class ResNetEncoder(BaseEncoder):
    """
    A wrapper around a ResNet model that extracts
    activations from a specified layer during forward pass
    """
    def __init__(
        self,
        model: nn.Module,
        layer: Union[str, int] = -2,
        dataset: str = 'imagenet',
        width_multiplier: int = 1,
        pretrained: bool = False,
        **kwargs
    ):
        """
        layer: the layer from which to extract the hidden representation
        dataset: the dataset on which the model will be pretrained, either 'imagenet' or 'cifar'
        width_multiplier: the width multiplier for the ResNet model (1, 2, 4, 8, ...)
        pretrained: whether to load pretrained weights (does not work for pretrained weights if width_multiplier > 1)
        """
        super().__init__(model, layer)
        self.width_multiplier = int(width_multiplier)
        self.pretrained = pretrained

        if self.width_multiplier != 1:
            assert not pretrained, 'Pretrained weights not available for wide ResNet.'

        self.create_wider_resnet()
        if 'cifar' in dataset or dataset=='svhn':
            self.modify_for_cifar()
        # for SSL pretraining, we do not need the final fc layer
        self.net.fc = nn.Identity()

    def create_wider_resnet(self):
        if self.width_multiplier == 1:
            return

        # modify the first conv layer
        self.net.conv1 = nn.Conv2d(3, 64 * self.width_multiplier, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.net.bn1 = nn.BatchNorm2d(64 * self.width_multiplier)

        if not self.pretrained:
            self._initialize_weights(self.net.conv1)
            self._initialize_weights(self.net.bn1)
          
        # modify the subsequent layers
        for layer in [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]:
            for block in layer:
                block.conv1 = self._wider_bottleneck(block.conv1)
                block.bn1 = nn.BatchNorm2d(block.conv1.out_channels)
                block.conv2 = self._wider_bottleneck(block.conv2)
                block.bn2 = nn.BatchNorm2d(block.conv2.out_channels)
                block.conv3 = self._wider_bottleneck(block.conv3)
                block.bn3 = nn.BatchNorm2d(block.conv3.out_channels)

                if block.downsample is not None:
                    block.downsample[0] = self._wider_bottleneck(block.downsample[0])
                    block.downsample[1] = nn.BatchNorm2d(block.downsample[0].out_channels)

    def _wider_bottleneck(self, conv):
        widened = nn.Conv2d(
            conv.in_channels * self.width_multiplier,
            conv.out_channels * self.width_multiplier,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None)
        )
        if not self.pretrained:
            self._initialize_weights(widened)
        del conv
        return widened

    def modify_for_cifar(self):
        # replace the first conv layer to adapt to CIFAR
        self.net.conv1 = nn.Conv2d(3, 64 * self.width_multiplier, 3, 1, 1, bias=False)
        # remove the first max pooling operation
        self.net.maxpool = nn.Identity()

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)


# ========================== ViT Encoders ==========================

class OldViTEncoder(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    def forward(self, x):
        # Process input to patch embeddings and permute
        x = self.vit._process_input(x)
        n = x.shape[0]
        # Add class token
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # Pass through transformer encoder layers
        x = self.vit.encoder(x)
        # Apply final LayerNorm if present
        if hasattr(self.vit.encoder, 'ln'):
            x = self.vit.encoder.ln(x)
        # Extract CLS token embedding as feature vector
        cls_embedding = x[:, 0]  # shape (batch_size, hidden_dim)
        return cls_embedding


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