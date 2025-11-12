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
        breakpoint()
        outputs = self.vicreg_model(x.to(x.device))

        h = None
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                h = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                h = outputs.last_hidden_state
            elif isinstance(outputs, dict):
                for k in ('pooler_output', 'last_hidden_state', 'hidden_states', 'embeddings'):
                    if k in outputs and outputs[k] is not None:
                        h = outputs[k]
                        break

        if h is None:
            raise RuntimeError(f"Could not extract tensor features from model outputs: {type(outputs)}")

        if h.dim() == 3:
            # prefer CLS token if present (first token), otherwise mean-pool
            try:
                # take first token (commonly CLS)
                h = h[:, 0, :]
            except Exception:
                h = h.mean(dim=1)


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