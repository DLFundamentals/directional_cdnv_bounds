from data_utils.augmentations_hub import simclr, mae, clip, ijepa

AUGMENTATION_REGISTRY = {
    'simclr': simclr.get_simclr_transforms,
    'mae': mae.get_mae_transforms,
    'ijepa': ijepa.get_ijepa_transforms,
    'clip': clip.get_clip_transforms,
    'siglip': clip.get_clip_transforms,  # share for now
}

def get_transforms(model_name, dataset='imagenet'):
    key = model_name.lower()
    if key not in AUGMENTATION_REGISTRY:
        raise NotImplementedError(f"No augmentations defined for {model_name}")
    return AUGMENTATION_REGISTRY[key](dataset)