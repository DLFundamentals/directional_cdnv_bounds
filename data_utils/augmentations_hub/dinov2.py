from __future__ import annotations

from dataclasses import dataclass

from torchvision import transforms

from data_utils.augmentations_hub.common_transforms import RepeatChannelsIfNeeded


@dataclass(frozen=True)
class DinoMultiCropTransform:
    """DINO-style transforms providing global + local crops.

    This object is callable (defaults to global crop) and also exposes
    `global_transform` and `local_transform` so callers can explicitly pick.
    """

    global_transform: transforms.Compose
    local_transform: transforms.Compose

    def __call__(self, img):
        return self.global_transform(img)


def get_dinov2_transforms(dataset: str = "imagenet"):
    """Returns DINOv2-style multi-crop (train) and basic evaluation transforms.

    Registry contract: `(train_transform, basic_transform)`.
    - `train_transform` is a `DinoMultiCropTransform` for DINO-style training.
    - `basic_transform` is a single-crop eval pipeline.
    """

    dataset = (dataset or "imagenet").lower()

    # Treat mini-ImageNet like ImageNet (same mean/std and crop sizes).
    is_imagenet_like = ("imagenet" in dataset) or ("mini" in dataset)
    if not is_imagenet_like:
        raise NotImplementedError(
            f"dinov2 transforms are only implemented for imagenet-like datasets; got: {dataset}"
        )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

    global_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.32, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            RepeatChannelsIfNeeded(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    local_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(96, scale=(0.05, 0.32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            RepeatChannelsIfNeeded(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    basic_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            RepeatChannelsIfNeeded(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return (
        DinoMultiCropTransform(global_transform=global_transform, local_transform=local_transform),
        basic_transform,
    )
