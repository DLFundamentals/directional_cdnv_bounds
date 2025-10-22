import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Union, Optional

from encoders.resnet import ResNetEncoder
from encoders.vit import ViTEncoder
    

class SimCLR(nn.Module):
    def __init__(
        self,
        model,
        layer: Union[str, int]  = -2,
        dataset: str = 'imagenet',
        width_multiplier: int = 1,
        hidden_dim: int = 512,
        projection_dim: int = 128,
        **kwargs
    ):
        super().__init__()

        # Wrap model based on architecture type
        if isinstance(model, models.ResNet): 
            self.encoder = ResNetEncoder(
                model, 
                dataset=dataset,
                layer=layer,
                width_multiplier=kwargs.get('width_multiplier', width_multiplier), 
                pretrained=False
            )
        elif isinstance(model, models.VisionTransformer):
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
            if 'imagenet' in dataset:
                h = self.encoder(torch.randn(1, 3, 224, 224))
            elif 'cifar' in dataset or dataset == 'svhn':
                h = self.encoder(torch.randn(1, 3, 32, 32))
            else:
                raise NotImplementedError(f"{dataset} not implemented")

        input_dim = h.shape[1]
        hidden_dim = kwargs.get('hidden_dim', hidden_dim)
        projection_dim = kwargs.get('projection_dim', projection_dim)
        self.projector = SimCLRProjector(input_dim, hidden_dim, projection_dim)

        # load state dict
        ckpt_path = kwargs.get('ckpt_path', None)
        if ckpt_path:
            self.load_snapshot(ckpt_path)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1) # flatten the tensor
        g_h = self.projector(h)
        return h, F.normalize(g_h, dim=-1)
    
    def load_snapshot(self, snapshot_path: str, device: str ='cuda'):
        snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
        state_dict = snapshot['MODEL_STATE']
        epochs_trained = snapshot['EPOCHS_RUN']
        print(f"Loaded model from epoch {epochs_trained}")
        self.load_state_dict(state_dict)
        self.eval()
        print("SSL Model loaded successfully")

# ========================== Projector ==========================
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
    
class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False