import torch
from torchvision import transforms
from data_utils.augmentations_hub.common_transforms import RepeatChannelsIfNeeded

def get_mae_transforms(dataset: str = 'imagenet'):
    """
    Returns IJ-EPA specific data augmentation (train) and basic evaluation transforms for a given dataset.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    basic_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        RepeatChannelsIfNeeded(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return basic_transform, basic_transform