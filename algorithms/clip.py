import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPConfig, AutoProcessor
import yaml

class CLIPAdapter(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        hidden_size = clip_model.config.vision_config.hidden_size
        
    def forward(self, x):
        pixel_values = x.to(self.clip_model.device)
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        h = vision_outputs.pooler_output  # TODO: verify this is the correct output
        return h, None

def create_clip_model(dataset: str, 
                        vit_size: str = "B",
                        vision_patch_size: int = 32):
    if dataset in ['imagenet', 'mini_imagenet']:
        if vit_size == "B" and vision_patch_size == 32:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        elif vit_size == "B" and vision_patch_size == 16:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        elif vit_size == "L" and vision_patch_size == 14:
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        else:
            raise ValueError("Unsupported CLIP model size or patch size.")
        return model
    else: 
        vision_config = {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'image_size': 32 if 'cifar' in dataset.lower() else 224,
            'patch_size': 4 if 'cifar' in dataset.lower() else 32,
            'num_channels': 3,
            'hidden_act': 'quick_gelu',
            'layer_norm_eps': 1e-5,
            'attention_dropout': 0.0,
            'projection_dim': 512,
        }
        clip_config = CLIPConfig.from_pretrained(
            "openai/clip-vit-base-patch32",
            vision_config=vision_config
        )
        model = CLIPModel(clip_config)
        return model


def create_clip_adapter(dataset: str, **kwargs):
    vit_size = kwargs.get('vit_size', 'B')
    vision_patch_size = kwargs.get('vision_patch_size', 32)
    model = create_clip_model(dataset, vit_size, vision_patch_size)
    return CLIPAdapter(model)