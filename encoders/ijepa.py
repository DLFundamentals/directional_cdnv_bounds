import torch
import torch.nn as nn
import torchvision.models as models
from transformers import IJepaModel, IJepaConfig, AutoImageProcessor


class IJepaAdapter(nn.Module):
    def __init__(self, ijepa_model):
        super().__init__()
        self.ijepa_model = ijepa_model
        self.encoder = ijepa_model.encoder
        hidden_size = ijepa_model.config.hidden_size
        #self.processor = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")

    
    
    def forward(self, x):
        #processed = self.processor(x, return_tensors="pt")
        encoder_outputs = self.ijepa_model(x)
        h = encoder_outputs.last_hidden_state[:, 0]  # CLS token     
        return h, None


def create_ijepa_encoder(dataset: str):
    if dataset in ['imagenet', 'mini_imagenet']:
        model = IJepaModel.from_pretrained("facebook/ijepa_vith14_1k")
        return model.encoder
    else:
        config = IJepaConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            image_size = 32 if 'cifar' in dataset.lower() else 224,
            patch_size = 4 if 'cifar' in dataset.lower() else 16,
            num_channels=3,
            qkv_bias=True,
            hidden_act="gelu",
            layer_norm_eps=1e-6,
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
        )
        model = IJepaModel(config)
        return model.encoder


def create_ijepa_ssl_model(dataset: str, **kwargs):
    
    model = IJepaModel.from_pretrained("facebook/ijepa_vith14_1k")
    return IJepaAdapter(model)
