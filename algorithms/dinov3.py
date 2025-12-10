import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor, Dinov2Config  # Use Dinov2Config for DINOv3

class Dinov3Adapter(nn.Module):
    def __init__(self, dinov3_model, processor):
        super().__init__()
        self.dinov3_model = dinov3_model
        self.processor = processor
        hidden_size = dinov3_model.config.hidden_size
    
    def forward(self, x):
        processed = self.processor(x, return_tensors="pt", do_rescale=False)
        
        outputs = self.dinov3_model(processed['pixel_values'].to(x.device))
        
        h = outputs.last_hidden_state[:, 0]
        
        return h, None

def create_dinov3_model(dataset: str,
                        encoder_type: str = 'vit_s',
                        patch_size: int = 16):
    """
    Create a DINOv3 model for the specified dataset and configuration.
    
    Args:
        dataset: Dataset name (e.g., 'imagenet', 'mini_imagenet', 'cifar10', 'cifar100')
        encoder_type: Encoder architecture ('vit_s', 'vit_b', 'vit_l', 'vit_g')
        patch_size: Patch size (14 or 16)
    
    Returns:
        DINOv3 model
    """
    if dataset in ['imagenet', 'mini_imagenet']:
        model_name = _get_pretrained_model_name(encoder_type, patch_size)
        model = AutoModel.from_pretrained(model_name)
        return model
    else:
        config = Dinov2Config(  # Use Dinov2Config
            hidden_size=384 if encoder_type == 'vit_s' else 768,
            num_hidden_layers=12 if encoder_type == 'vit_s' else 12,
            num_attention_heads=6 if encoder_type == 'vit_s' else 12,
            intermediate_size=1536 if encoder_type == 'vit_s' else 3072,
            image_size=32 if 'cifar' in dataset.lower() else 224,
            patch_size=4 if 'cifar' in dataset.lower() else patch_size,
            num_channels=3,
            qkv_bias=True,
            hidden_act="gelu",
            layer_norm_eps=1e-6,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            drop_path_rate=0.0,
        )
        model = AutoModel.from_config(config)
        return model

def _get_pretrained_model_name(encoder_type: str, patch_size: int) -> str:
    """
    Get the pretrained model name based on encoder type and patch size.
    
    Available models:
    - ViT-S/14, ViT-S/16
    - ViT-B/14, ViT-B/16
    - ViT-L/14, ViT-L/16
    - ViT-g/14
    """
    encoder_type = encoder_type.lower()
    
    model_mapping = {
        ('vit_s', 14): "facebook/dinov3-vits14-pretrain-lvd1689m",
        ('vit_s', 16): "facebook/dinov3-vits16-pretrain-lvd1689m",
        ('vit_b', 14): "facebook/dinov3-vitb14-pretrain-lvd1689m",
        ('vit_b', 16): "facebook/dinov3-vitb16-pretrain-lvd1689m",
        ('vit_l', 14): "facebook/dinov3-vitl14-pretrain-lvd1689m",
        ('vit_l', 16): "facebook/dinov3-vitl16-pretrain-lvd1689m",
        ('vit_g', 14): "facebook/dinov3-vitg14-pretrain-lvd1689m",
    }
    
    key = (encoder_type, patch_size)
    if key not in model_mapping:
        raise ValueError(
            f"Unsupported combination: encoder_type={encoder_type}, patch_size={patch_size}. "
            f"Available combinations: {list(model_mapping.keys())}"
        )
    
    return model_mapping[key]

def create_dinov3_adapter(dataset: str, **kwargs):
    """
    Create a DINOv3 adapter for SSL training.
    
    Args:
        dataset: Dataset name
        **kwargs: Additional arguments including:
            - encoder_type: Encoder architecture (default: 'vit_s')
            - patch_size: Patch size (default: 16)
    
    Returns:
        Dinov3Adapter instance
    """
    encoder_type = kwargs.get('encoder_type', 'vit_s')
    patch_size = kwargs.get('patch_size', 16)
    
    model = create_dinov3_model(dataset, encoder_type, patch_size)
    
    if dataset in ['imagenet', 'mini_imagenet']:
        model_name = _get_pretrained_model_name(encoder_type, patch_size)
        processor = AutoImageProcessor.from_pretrained(model_name)
    else:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    
    return Dinov3Adapter(model, processor)