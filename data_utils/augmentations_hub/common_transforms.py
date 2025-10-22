import torch
from torchvision import transforms

class RepeatChannelsIfNeeded:
    """
    Transform that repeats a single-channel tensor to three channels.
    Used to handle grayscale images in datasets expecting RGB input.
    """
    def __call__(self, img_tensor):
        if img_tensor.shape[0] == 1:
            print(f"Repeating grayscale tensor with shape {img_tensor.shape}")
            return img_tensor.repeat(3, 1, 1)
        return img_tensor