from algorithms.clip import create_clip_adapter
from algorithms.ijepa import create_ijepa_adapter
from algorithms.simclr import SimCLR
from algorithms.mae import create_mae_adapter

SUPPORTED_ALGORITHMS = {
    'clip': create_clip_adapter,
    'ijepa': create_ijepa_adapter,
    'simclr': SimCLR,
    'mae': create_mae_adapter
}

def build_ssl_model(method: str, dataset: str, **kwargs):
    if method not in SUPPORTED_ALGORITHMS:
        raise NotImplementedError(f"SSL method '{method}' not supported.")
    if method == 'simclr':
        encoder_type = kwargs.get('encoder_type', 'resnet50')
        from encoders.factory import get_encoder
        encoder = get_encoder(encoder_type, dataset)
        return SUPPORTED_ALGORITHMS[method](model=encoder, dataset=dataset, **kwargs)
    return SUPPORTED_ALGORITHMS[method](dataset=dataset, **kwargs)