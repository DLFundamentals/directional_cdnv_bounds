import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipModel, AutoProcessor

class SiglipAdapter(nn.Module):
    def __init__(self, siglip_model, processor):
        super().__init__()
        self.siglip_model = siglip_model
        self.processor = None
        
        if hasattr(siglip_model.config, 'vision_config'):
            self.hidden_size = siglip_model.config.vision_config.hidden_size
        else:
            self.hidden_size = siglip_model.config.hidden_size

    def forward(self, x):
        if self.processor is not None:
            processed = self.processor(images=x, return_tensors="pt")
            pixel_values = processed['pixel_values'].to(x.device)
        else:
            pixel_values = x
        
        if isinstance(self.siglip_model, SiglipModel):
            outputs = self.siglip_model.vision_model(pixel_values=pixel_values)
        else:
            outputs = self.siglip_model(pixel_values=pixel_values)
        
        h = None
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            h = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            h = outputs.last_hidden_state
            if h.dim() == 3:
                h = h[:, 0, :]  # Take CLS token
        
        if h is None:
            raise RuntimeError(f"Could not extract features from model outputs: {type(outputs)}")
        
        return h, None

def create_siglip_model(dataset: str,
                        model_size: str = None,
                        patch_size: int = 16,
                        **kwargs):
    model_configs = {
        ('base', 16): 'google/siglip-base-patch16-224',
        ('large', 16): 'google/siglip-large-patch16-256',
        ('so400m', 14): 'google/siglip-so400m-patch14-384',
    }
    
    model_name = model_configs.get((model_size.lower(), patch_size))
    
    if model_name is None:
        model_name = 'google/siglip-base-patch16-224'
        print(f"Warning: No model for size={model_size}, patch={patch_size}. Using {model_name}")
    
    try:
        model = SiglipVisionModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Warning: Could not load pretrained model {model_name}: {e}")
        from transformers import SiglipVisionConfig
        config = SiglipVisionConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=224,
            patch_size=16,
        )
        model = SiglipVisionModel(config)
        processor = None
    
    return model, processor

def create_siglip_adapter(dataset: str, **kwargs):
    # Remove model-specific kwargs before forwarding to the model creator to
    # avoid passing duplicate keyword arguments (they may also be present in
    # `kwargs` coming from factory.build_ssl_model).
    model_size = kwargs.pop('model_size', 'base')
    patch_size = kwargs.pop('patch_size', 16)
    model, processor = create_siglip_model(dataset, model_size=model_size, patch_size=patch_size, **kwargs)
    return SiglipAdapter(model, processor)