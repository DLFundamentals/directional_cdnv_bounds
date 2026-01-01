from dataclasses import dataclass
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

from datasets import load_dataset


@dataclass
class MiniImageNetCfg:
    name: str
    hf_repo: str
    hf_cache_dir: str
    img_size: int = 224
    batch_size: int = 128
    num_workers: int = 8
    train_split: str = "train"
    test_split: str = "test"
    num_views: int = 1  # 1 for MAE, 2 for VICReg


class MiniImageNetDataModule(pl.LightningDataModule):
    """
    HuggingFace Datasets-backed mini-ImageNet.

    - prepare_data(): downloads/caches dataset (runs on rank 0 only in Lightning)
    - setup(): creates split objects
    - dataloaders: return torch DataLoaders

    HF example items typically look like:
      {"image": PIL.Image, "label": int, ...}
    """

    def __init__(self, cfg: MiniImageNetCfg):
        super().__init__()
        self.cfg = cfg
        self.ds_train = None
        self.ds_test = None

        # Global crop augmentation (for first 2 views in DINO)
        self.global_crop_tf = transforms.Compose([
            transforms.RandomResizedCrop(self.cfg.img_size, scale=(0.32, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        
        # Local crop augmentation (for remaining views in DINO)
        self.local_crop_tf = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.05, 0.32)),  # Smaller crops
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        
        # Test/validation transform (no augmentation)
        self.test_tf = transforms.Compose([
            transforms.Resize(self.cfg.img_size + 32),
            transforms.CenterCrop(self.cfg.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def prepare_data(self):
        # Download/cache the dataset. Lightning calls this only on rank 0.
        # We load both splits once so the cache is warm.
        load_dataset(self.cfg.hf_repo, split=self.cfg.train_split, cache_dir=self.cfg.hf_cache_dir)
        load_dataset(self.cfg.hf_repo, split=self.cfg.test_split, cache_dir=self.cfg.hf_cache_dir)

    def setup(self, stage: Optional[str] = None):
        # Create datasets (cheap if cached)
        self.ds_train = load_dataset(
            self.cfg.hf_repo,
            split=self.cfg.train_split,
            cache_dir=self.cfg.hf_cache_dir,
        )
        self.ds_test = load_dataset(
            self.cfg.hf_repo,
            split=self.cfg.test_split,
            cache_dir=self.cfg.hf_cache_dir,
        )

        # Tell HF to return torch tensors for label; keep image as PIL until transform
        self.ds_train.set_format(type="python")
        self.ds_test.set_format(type="python")

    def _collate(self, batch, train: bool):
        """
        Collate function that returns multiple views for SSL methods.
        For train=True and num_views > 1, applies augmentation multiple times.
        For DINO: first 2 views are global crops, remaining are local crops.
        """
        labels = [ex.get("label", -1) for ex in batch]
        
        if train and self.cfg.num_views > 1:
            # Create multiple augmented views of each image
            views = []
            for i in range(self.cfg.num_views):
                # First 2 views: global crops (224x224)
                # Remaining views: local crops (96x96)
                tf = self.global_crop_tf if i < 2 else self.local_crop_tf
                images = [tf(ex["image"].convert("RGB")) for ex in batch]
                views.append(torch.stack(images, dim=0))
            return views, torch.tensor(labels, dtype=torch.long)
        else:
            # Single view (for validation or MAE)
            images = [self.test_tf(ex["image"].convert("RGB")) for ex in batch]
            return [torch.stack(images, dim=0)], torch.tensor(labels, dtype=torch.long)

    def train_collate(self, batch):
        """Collate function for training loader (with augmentations)."""
        return self._collate(batch, train=True)

    def eval_collate(self, batch):
        """Collate function for eval/probe loaders (no augmentations)."""
        return self._collate(batch, train=False)

    def train_dataloader(self):
        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=True,  
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0 and not (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            )),
            drop_last=True,
            collate_fn=self.train_collate,
        )
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(self.ds_train, **kwargs)

    def val_dataloader(self):
        # Let Lightning’s `use_distributed_sampler=True` create the sampler in DDP.
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        use_persistent = (self.cfg.num_workers > 0) and not is_ddp

        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=False,  # Lightning will override with a DistributedSampler in DDP
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=use_persistent,
            drop_last=False,
            collate_fn=self.eval_collate,
        )
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 2

        return DataLoader(self.ds_test, **kwargs)

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            collate_fn=self.eval_collate,
        )

    
    def probe_train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            collate_fn=self.eval_collate,
        )
    
    def probe_test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            collate_fn=self.eval_collate,
        )