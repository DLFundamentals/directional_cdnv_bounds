import os
import torch
import torchvision.models as models
from algorithms.simclr import SimCLR  # adjust this import to your layout

# we need this only for our implementation of SimCLR
SUPPORTED_ENCODERS = {
    'resnet50': lambda dataset: models.resnet50(pretrained=False),
    'vit_b': lambda dataset: models.VisionTransformer(
        patch_size=16 if 'imagenet' in dataset else 4,
        image_size=224 if 'imagenet' in dataset else 32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768 if dataset in ['imagenet', 'mini_imagenet'] else 384,
        mlp_dim=3072 if dataset in ['imagenet', 'mini_imagenet'] else 1536,
    ),
}


def get_encoder(encoder_type: str, dataset: str):
    if encoder_type not in SUPPORTED_ENCODERS:
        raise NotImplementedError(f"Encoder type '{encoder_type}' not supported.")
    return SUPPORTED_ENCODERS[encoder_type](dataset)


# def get_ssl_model(method: str, encoder, dataset: str, **kwargs):
#     if method == 'simclr':
#         return SimCLR(
#             model=encoder,
#             dataset=dataset,
#             width_multiplier=kwargs.get('width_multiplier', 1),
#             hidden_dim=kwargs.get('hidden_dim', 2048),
#             projection_dim=kwargs.get('projection_dim', 128),
#             image_size=224 if 'imagenet' in dataset else 32,
#             patch_size=16 if 'imagenet' in dataset else 4,
#             stride=16 if 'imagenet' in dataset else 2,
#             token_hidden_dim=768 if 'imagenet' in dataset else 384,
#             mlp_dim=3072 if 'imagenet' in dataset else 1536,
#         )
#     elif method == 'ijepa':
#         return create_ijepa_ssl_model(dataset=dataset, **kwargs)
#     elif method == 'clip':
#         return create_clip_ssl_model(dataset=dataset, **kwargs)
#     raise NotImplementedError(f"SSL method '{method}' not supported.")


# # Utils for loading checkpoints (need to decide where to place them)
# def load_latest_checkpoint(ssl_model, checkpoint_dir: str, device: str):
#     checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
#     if not checkpoint_files:
#         raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

#     sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
#     best_checkpoint = sorted_checkpoints[-1]
#     snapshot_path = os.path.join(checkpoint_dir, best_checkpoint)
#     print(f"Loading checkpoint: {snapshot_path}")
#     return load_snapshot(ssl_model, snapshot_path, device)