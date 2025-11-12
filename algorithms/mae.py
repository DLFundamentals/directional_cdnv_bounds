import torch
import torch.nn as nn
from transformers import ViTMAEModel, ViTMAEForPreTraining, ViTMAEConfig


class ViTMAEAdapter(nn.Module):
    def __init__(self, vitmae_model, use_pretraining=True):
        super().__init__()
        self.use_pretraining = use_pretraining
        
        if use_pretraining:
            self.vitmae_model = vitmae_model
            self.encoder = vitmae_model.vit
        else:
            self.vitmae_model = vitmae_model
            self.encoder = vitmae_model
        
        hidden_size = vitmae_model.config.hidden_size

    
    def forward(self, x, noise=None):
        if isinstance(x, (tuple, list)) and len(x) == 2:
            x1, x2 = x
            x = x1
        
        pixel_values = x
                
        if self.use_pretraining and self.training:
            # Pretraining mode - return loss
            outputs = self.vitmae_model(pixel_values=pixel_values, noise=noise)
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'mask': outputs.mask,
                'ids_restore': outputs.ids_restore,
                'hidden_states': outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None
            }
        else:
            with torch.no_grad():
                if self.use_pretraining:
                    outputs = self.vitmae_model.vit(pixel_values=pixel_values)
                else:
                    outputs = self.vitmae_model(pixel_values=pixel_values)
                
                h = outputs.last_hidden_state[:, 0]
                return h, None


def create_vitmae_encoder(dataset: str, pretrained: bool = True):
    if dataset in ['imagenet', 'mini_imagenet'] and pretrained:
        model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        return model
    else:
        config = ViTMAEConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=32 if 'cifar' in dataset.lower() else 224,
            patch_size=4 if 'cifar' in dataset.lower() else 16,
            num_channels=3,
            qkv_bias=True,
            hidden_act='gelu',
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            decoder_num_attention_heads=16,
            decoder_hidden_size=512,
            decoder_num_hidden_layers=8,
            decoder_intermediate_size=2048,
            mask_ratio=0.75,
            norm_pix_loss=False,
        )
        model = ViTMAEModel(config)
        return model


def create_mae_adapter(
    dataset: str,
    pretrained: bool = True,
    use_pretraining: bool = False,
    mask_ratio: float = 0.75,
    norm_pix_loss: bool = False,
    decoder_num_attention_heads: int = 16,
    decoder_hidden_size: int = 512,
    decoder_num_hidden_layers: int = 8,
    decoder_intermediate_size: int = 2048,
    **kwargs
):
    if dataset in ['imagenet', 'mini_imagenet'] and pretrained:
        if use_pretraining:
            model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        else:
            model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    else:
        # Create custom config
        config = ViTMAEConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=32 if 'cifar' in dataset.lower() else (84 if dataset == 'mini_imagenet' else 224),
            patch_size=4 if 'cifar' in dataset.lower() else (7 if dataset == 'mini_imagenet' else 16),
            num_channels=3,
            qkv_bias=True,
            hidden_act='gelu',
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            # Decoder config
            decoder_num_attention_heads=decoder_num_attention_heads,
            decoder_hidden_size=decoder_hidden_size,
            decoder_num_hidden_layers=decoder_num_hidden_layers,
            decoder_intermediate_size=decoder_intermediate_size,
            mask_ratio=mask_ratio,
            norm_pix_loss=norm_pix_loss,
        )
        
        if use_pretraining:
            model = ViTMAEForPreTraining(config)
        else:
            model = ViTMAEModel(config)
    
    return ViTMAEAdapter(model, use_pretraining=use_pretraining)