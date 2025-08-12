import os
import torch
import torchvision.models as models
from transformers import IJepaModel, IJepaConfig
from encoders.simclr import SimCLR  # adjust this import to your layout


def create_ijepa_encoder(dataset: str):
    """Create I-JEPA encoder using Hugging Face transformers"""
    if dataset in ['imagenet', 'mini_imagenet']:
        # Use pre-trained I-JEPA model for ImageNet/Mini-ImageNet
        model = IJepaModel.from_pretrained("facebook/ijepa_vith14_1k")
        # Extract just the encoder part
        return model.encoder
    else:
        # For other datasets, create a smaller configuration
        config = IJepaConfig(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            image_size=84 if dataset == 'mini_imagenet' else (32 if 'cifar' in dataset.lower() else 224),
            patch_size=7 if dataset == 'mini_imagenet' else (4 if 'cifar' in dataset.lower() else 16),
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
    if dataset in ['imagenet', 'mini_imagenet']:
        if dataset == 'mini_imagenet':
            config = IJepaConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                image_size=84, 
                patch_size=7,   
                num_channels=3,
                qkv_bias=True,
                hidden_act="gelu",
                layer_norm_eps=1e-6,
                attention_probs_dropout_prob=0.0,
                hidden_dropout_prob=0.0,
            )
            model = IJepaModel(config)
        else:
            model = IJepaModel.from_pretrained("facebook/ijepa_vith14_1k")
    else:
        config = IJepaConfig(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            image_size=32 if 'cifar' in dataset.lower() else 224,
            patch_size=4 if 'cifar' in dataset.lower() else 16,
            num_channels=3,
            qkv_bias=True,
            hidden_act="gelu",
            layer_norm_eps=1e-6,
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
        )
        model = IJepaModel(config)
    
    return model


SUPPORTED_ENCODERS = {
    'resnet50': lambda dataset: models.resnet50(pretrained=False),
    'vit_b': lambda dataset: models.VisionTransformer(
        patch_size=16 if dataset == 'imagenet' else (7 if dataset == 'mini_imagenet' else 4),
        image_size=224 if dataset == 'imagenet' else (84 if dataset == 'mini_imagenet' else 32),
        num_layers=12,
        num_heads=12,
        hidden_dim=768 if dataset in ['imagenet', 'mini_imagenet'] else 384,
        mlp_dim=3072 if dataset in ['imagenet', 'mini_imagenet'] else 1536,
    ),
    'ijepa': lambda dataset: create_ijepa_encoder(dataset=dataset),
}


def get_encoder(encoder_type: str, dataset: str):
    if encoder_type not in SUPPORTED_ENCODERS:
        raise NotImplementedError(f"Encoder type '{encoder_type}' not supported.")
    return SUPPORTED_ENCODERS[encoder_type](dataset)


def get_ssl_model(method: str, encoder, dataset: str, **kwargs):
    if method == 'simclr':
        return SimCLR(
            model=encoder,
            dataset=dataset,
            width_multiplier=kwargs.get('width_multiplier', 1),
            hidden_dim=kwargs.get('hidden_dim', 2048),
            projection_dim=kwargs.get('projection_dim', 128),
            image_size=224 if dataset == 'imagenet' else (84 if dataset == 'mini_imagenet' else 32),
            patch_size=16 if dataset == 'imagenet' else (7 if dataset == 'mini_imagenet' else 4),
            stride=16 if dataset == 'imagenet' else (7 if dataset == 'mini_imagenet' else 2),
            token_hidden_dim=768 if dataset in ['imagenet', 'mini_imagenet'] else 384,
            mlp_dim=3072 if dataset in ['imagenet', 'mini_imagenet'] else 1536,
            use_old=kwargs.get('use_old', False),
        )
    elif method == 'ijepa':
        return create_ijepa_ssl_model(dataset=dataset, **kwargs)
    raise NotImplementedError(f"SSL method '{method}' not supported.")


def load_latest_checkpoint(ssl_model, checkpoint_dir: str, device: str):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    best_checkpoint = sorted_checkpoints[-1]
    snapshot_path = os.path.join(checkpoint_dir, best_checkpoint)
    print(f"Loading checkpoint: {snapshot_path}")
    return load_snapshot(ssl_model, snapshot_path, device)


def load_snapshot(ssl_model, snapshot_path: str, device: str):
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    state_dict = snapshot['MODEL_STATE']
    epochs_trained = snapshot['EPOCHS_RUN']
    print(f"Loaded model from epoch {epochs_trained}")
    ssl_model.load_state_dict(state_dict)
    ssl_model = ssl_model.to(device)
    ssl_model.eval()
    print("SSL Model loaded successfully")
    return ssl_model


def build_ssl_encoder(
    method: str,
    encoder_type: str,
    dataset: str,
    checkpoint: str = None,
    device: str = 'cpu',
    **kwargs,
):
    if method == 'ijepa':
        ssl_model = get_ssl_model(method, None, dataset, **kwargs)
    else:
        encoder = get_encoder(encoder_type, dataset)
        ssl_model = get_ssl_model(method, encoder, dataset, **kwargs)

    if checkpoint:
        if os.path.isdir(checkpoint):
            ssl_model = load_latest_checkpoint(ssl_model, checkpoint, device)
        elif os.path.isfile(checkpoint):
            ssl_model = load_snapshot(ssl_model, checkpoint, device)
        else:
            raise FileNotFoundError(f"Checkpoint path {checkpoint} not found.")

    return ssl_model