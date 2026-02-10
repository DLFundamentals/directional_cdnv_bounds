import os, json
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

COLOR2ID = {"red": 0, "green": 1, "blue": 2, "purple": 3, "dark_brown": 4, "yellow": 5}
SHAPE2ID = {"circle": 0, "triangle": 1, "square": 2, "pentagon": 3}
STYLE2ID = {"plus": 0, "minus": 1, "dots": 2, "cross": 3}
SIZE2ID  = {"small": 0, "big": 1}

LABEL_MAPS = {
    "color": COLOR2ID,
    "shape": SHAPE2ID,
    "style": STYLE2ID,
    "size_label": SIZE2ID,
}

class ImageFolderWithMeta(Dataset):
    """
    torchvision-style dataset:
      __getitem__ returns (PIL_image, int_label)
    """
    def __init__(self, root_dir, records, label_key):
        self.root_dir = root_dir
        self.records = records
        self.label_key = label_key
        self.map = LABEL_MAPS[label_key]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img_path = os.path.join(self.root_dir, r["file"])
        img = Image.open(img_path).convert("RGB")
        y = self.map[r[self.label_key]]
        return img, y
