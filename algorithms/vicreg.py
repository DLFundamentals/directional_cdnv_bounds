import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from torchvision.models import resnet50

class VICRegAdapter(nn.Module):
    def __init__(self, vicreg_model, processor):
        super().__init__()
        self.vicreg_model = vicreg_model
        self.processor = processor
        self.feature_dim = vicreg_model.config.hidden_sizes[-1] if hasattr(vicreg_model.config, 'hidden_sizes') else 768

    def forward(self, x):
        processed = self.processor(x, return_tensors="pt", do_rescale=False)
        
        # Get embeddings from VICReg model
        outputs = self.vicreg_model(processed['pixel_values'].to(x.device))
        
        h = outputs[0] if isinstance(outputs, tuple) else outputs
            
        return h, None

def create_vicreg_model(dataset: str,
                        model_name: str = "Ramos-Ramos/vicreg-resnet-50",
                        **kwargs):
    
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load pretrained model {model_name}: {e}")
        model = resnet50(pretrained=False)
        model.fc = nn.Identity()
        processor = None
    
    return model, processor

def create_vicreg_adapter(dataset: str, **kwargs):
    model_name = kwargs.get('model_name', 'Ramos-Ramos/vicreg-resnet-50')
    model, processor = create_vicreg_model(dataset, model_name=model_name, **kwargs)    
    return VICRegAdapter(model, processor)