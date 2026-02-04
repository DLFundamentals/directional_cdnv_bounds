from dataclasses import dataclass
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset

from data_utils.augmentations_hub.registry import get_transforms


@dataclass
class MiniImageNetCfg:
    name: str
    hf_repo: str
    hf_cache_dir: str
    method: str = "dino"
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

        self.train_tfms, self.basic_tfms = self._get_transforms(self.cfg.method, self.cfg.name)
        self._init_view_transforms()

    def _get_transforms(self, method: str, dataset_name: str):
        """Method-aware transforms factory (mirrors data_utils/dataloaders.py)."""
        return get_transforms(
            method=method,
            dataset="cifar" if "cifar" in (dataset_name or "").lower() else dataset_name,
        )

    def _init_view_transforms(self):
        """Initialize per-view transforms.

        - For DINO: use global/local crops if provided by the factory.
        - Otherwise: use the single train transform for every view.
        """

        train = self.train_tfms
        method = (self.cfg.method or "").lower()

        if method in {"dinov2"} and hasattr(train, "global_transform") and hasattr(train, "local_transform"):
            self.global_crop_tf = train.global_transform
            self.local_crop_tf = train.local_transform
        else:
            self.global_crop_tf = train
            self.local_crop_tf = train

        self.test_tf = self.basic_tfms

    def prepare_data(self):
        # Download/cache the dataset. Lightning calls this only on rank 0.
        # We load both splits once so the cache is warm.
        load_dataset(self.cfg.hf_repo, split=self.cfg.train_split, cache_dir=self.cfg.hf_cache_dir)
        load_dataset(self.cfg.hf_repo, split=self.cfg.test_split, cache_dir=self.cfg.hf_cache_dir)

    def setup(self, stage: Optional[str] = None):
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

            method = (self.cfg.method or "").lower()
            is_dino = method in {"dino", "dinov2"} and hasattr(self.train_tfms, "global_transform")

            for i in range(self.cfg.num_views):
                # DINO: first 2 views are global crops, remaining are local crops.
                if is_dino:
                    tf = self.global_crop_tf if i < 2 else self.local_crop_tf
                else:
                    tf = self.global_crop_tf

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