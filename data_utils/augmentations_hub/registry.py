from data_utils.augmentations_hub import simclr, mae, clip, ijepa, vicreg, siglip, dinov2

AUGMENTATION_REGISTRY = {
    'simclr': simclr.get_simclr_transforms,
    'mae': mae.get_mae_transforms,
    'ijepa': ijepa.get_ijepa_transforms,
    'clip': clip.get_clip_transforms,
    'siglip': siglip.get_siglip_transforms,
    'vicreg': vicreg.get_vicreg_transforms,
    'dino': dinov2.get_dinov2_transforms,
    'dinov2': dinov2.get_dinov2_transforms

}

def get_transforms(method, dataset='imagenet'):
    key = method.lower()
    if key not in AUGMENTATION_REGISTRY:
        raise NotImplementedError(f"No augmentations defined for {method}")
    return AUGMENTATION_REGISTRY[key](dataset)
