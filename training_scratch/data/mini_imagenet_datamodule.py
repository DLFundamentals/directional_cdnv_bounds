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

        # Augmentation for training (used for SSL)
        self.train_tf = transforms.Compose([
            transforms.RandomResizedCrop(self.cfg.img_size, scale=(0.2, 1.0)),
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
        """
        tf = self.train_tf if train else self.test_tf
        labels = [ex.get("label", -1) for ex in batch]
        
        if train and self.cfg.num_views > 1:
            # Create multiple augmented views of each image
            views = []
            for _ in range(self.cfg.num_views):
                images = [tf(ex["image"].convert("RGB")) for ex in batch]
                views.append(torch.stack(images, dim=0))
            return views, torch.tensor(labels, dtype=torch.long)
        else:
            # Single view (for validation or MAE)
            images = [tf(ex["image"].convert("RGB")) for ex in batch]
            return [torch.stack(images, dim=0)], torch.tensor(labels, dtype=torch.long)

    def train_dataloader(self):
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        sampler = DistributedSampler(self.ds_train, shuffle=True) if is_ddp else None
        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            drop_last=True,
            collate_fn=lambda b: self._collate(b, train=True),
        )
        # Optional, but usually helps
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        return DataLoader(self.ds_train, **kwargs)

    def val_dataloader(self):
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        sampler = DistributedSampler(self.ds_test, shuffle=False) if is_ddp else None
        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            drop_last=False,
            collate_fn=lambda b: self._collate(b, train=False),
        )
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        return DataLoader(self.ds_test, **kwargs)
    
    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            collate_fn=lambda b: self._collate(b, train=False),
        )

    
    def probe_train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            collate_fn=lambda b: self._collate(b, train=False),
        )
    
    def probe_test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.num_workers > 0),
            collate_fn=lambda b: self._collate(b, train=False),
        )