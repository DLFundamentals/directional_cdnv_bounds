import json
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .synthetic_utils import LABEL_MAPS


@dataclass
class SyntheticShapesCfg:
	data_root: str = "./synthetic_shapes"
	metadata_file: str = "metadata.json"
	img_size: int = 224
	batch_size: int = 256
	num_workers: int = 8
	num_views: int = 6  # 2 global + 4 local for DINO
	val_fraction: float = 0.05
	seed: int = 123
	label_key: str = "shape"  # which attribute to supervise on for probes


class SyntheticShapesDataset(Dataset):
	"""Simple dataset backed by a list of metadata records.

	Returns dicts with PIL image and integer label based on cfg.label_key.
	"""

	def __init__(self, cfg: SyntheticShapesCfg, records: List[Dict[str, Any]]):
		self.cfg = cfg
		self.records = records
		self.label_map = LABEL_MAPS[self.cfg.label_key]

	def __len__(self):
		return len(self.records)

	def __getitem__(self, idx):
		r = self.records[idx]
		img_path = os.path.join(self.cfg.data_root, r["file"])
		img = Image.open(img_path).convert("RGB")
		label = self.label_map[r[self.cfg.label_key]]
		all_labels = {}
		for k, m in LABEL_MAPS.items():
			if k in r:
				all_labels[k] = m[r[k]]
		return {"image": img, "label": label, "labels": all_labels}


class SyntheticShapesDataModule(pl.LightningDataModule):
	"""DataModule producing DINO-style multi-crop views for synthetic_shapes.

	No color jitter or grayscale is used to keep color information intact.
	"""

	def __init__(self, cfg: SyntheticShapesCfg):
		super().__init__()
		self.cfg = cfg
		self.ds_train: Optional[Dataset] = None
		self.ds_val: Optional[Dataset] = None

		self.global_crop_tf = transforms.Compose(
			[
				transforms.RandomResizedCrop(self.cfg.img_size, scale=(0.32, 1.0)),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=(0.5, 0.5, 0.5),
					std=(0.5, 0.5, 0.5),
				),
			]
		)

		self.local_crop_tf = transforms.Compose(
			[
				transforms.RandomResizedCrop(96, scale=(0.05, 0.32)),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=(0.5, 0.5, 0.5),
					std=(0.5, 0.5, 0.5),
				),
			]
		)

		self.test_tf = transforms.Compose(
			[
				transforms.Resize(self.cfg.img_size + 32),
				transforms.CenterCrop(self.cfg.img_size),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=(0.5, 0.5, 0.5),
					std=(0.5, 0.5, 0.5),
				),
			]
		)

	def prepare_data(self):
		# Images are already on disk; nothing to download.
		pass

	def setup(self, stage: Optional[str] = None):
		# Load metadata once and create a train/val split.
		meta_path = os.path.join(self.cfg.data_root, self.cfg.metadata_file)
		with open(meta_path, "r") as f:
			records = json.load(f)

		n = len(records)
		val_size = max(1, int(n * self.cfg.val_fraction))
		indices = torch.randperm(n, generator=torch.Generator().manual_seed(self.cfg.seed)).tolist()
		val_idx = indices[:val_size]
		train_idx = indices[val_size:]

		train_records = [records[i] for i in train_idx]
		val_records = [records[i] for i in val_idx]

		self.ds_train = SyntheticShapesDataset(self.cfg, train_records)
		self.ds_val = SyntheticShapesDataset(self.cfg, val_records)

	def _collate(self, batch, train: bool, return_all_labels: bool = False):
		if return_all_labels:
			labels: Dict[str, torch.Tensor] = {}
			keys = set()
			for ex in batch:
				keys.update((ex.get("labels") or {}).keys())
			for k in sorted(keys):
				labels[k] = torch.tensor([int(ex["labels"][k]) for ex in batch], dtype=torch.long)
		else:
			labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
		if train and self.cfg.num_views > 1:
			views = []
			for i in range(self.cfg.num_views):
				tf = self.global_crop_tf if i < 2 else self.local_crop_tf
				images = [tf(ex["image"].convert("RGB")) for ex in batch]
				views.append(torch.stack(images, dim=0))
			return views, labels
		else:
			images = [self.test_tf(ex["image"].convert("RGB")) for ex in batch]
			views = [torch.stack(images, dim=0)]
			return views, labels

	def train_collate(self, batch):
		return self._collate(batch, train=True)

	def eval_collate(self, batch):
		return self._collate(batch, train=False)

	def probe_collate(self, batch):
		# Return all labelings so the linear probe can report multiple accuracies.
		return self._collate(batch, train=False, return_all_labels=True)

	def train_dataloader(self):
		kwargs = dict(
			batch_size=self.cfg.batch_size,
			shuffle=True,
			num_workers=self.cfg.num_workers,
			pin_memory=True,
			persistent_workers=self.cfg.num_workers > 0,
			drop_last=True,
			collate_fn=self.train_collate,
		)
		if self.cfg.num_workers > 0:
			kwargs["prefetch_factor"] = 2
		return DataLoader(self.ds_train, **kwargs)

	def val_dataloader(self):
		kwargs = dict(
			batch_size=self.cfg.batch_size,
			shuffle=False,
			num_workers=self.cfg.num_workers,
			pin_memory=True,
			persistent_workers=self.cfg.num_workers > 0,
			drop_last=False,
			collate_fn=self.eval_collate,
		)
		if self.cfg.num_workers > 0:
			kwargs["prefetch_factor"] = 2
		return DataLoader(self.ds_val, **kwargs)
	
	def probe_train_dataloader(self):
		kwargs = dict(
			batch_size=self.cfg.batch_size,
			shuffle=True,
			num_workers=self.cfg.num_workers,
			pin_memory=True,
			persistent_workers=self.cfg.num_workers > 0,
			collate_fn=self.probe_collate,
		)
		if self.cfg.num_workers > 0:
			kwargs["prefetch_factor"] = 2
		return DataLoader(self.ds_train, **kwargs)

	def probe_test_dataloader(self):
		return self.val_dataloader()

