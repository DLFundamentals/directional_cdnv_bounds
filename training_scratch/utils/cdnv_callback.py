import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
import os

# Add parent directories to path to import from eval_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from eval_utils.geometry import GeometricEvaluator


class CDNVCallback(pl. Callback):
    """
    Every N epochs: 
      - Extract features from train/val dataloaders
      - Compute CDNV and Directional CDNV using GeometricEvaluator
      - Log metrics to W&B/CSV
      
    Supports both ViT (MAE) and ResNet (VICReg) architectures.
    """
    def __init__(
        self,
        every_n_epochs=100,
        max_train_batches=200,
        max_val_batches=50,
        num_classes=10,
        enabled=True,
        compute_on_train=True,
        compute_on_val=True
    ):
        self.every_n_epochs = every_n_epochs
        self. max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.num_classes = num_classes
        self.enabled = enabled
        self.compute_on_train = compute_on_train
        self.compute_on_val = compute_on_val
        self.evaluator = None  # Will be initialized with device

    def _extract_backbone_features(self, backbone, images):
        """
        Extract features from backbone, handling both ViT and ResNet architectures.
        Returns normalized features [B, D]. 
        """
        if hasattr(backbone, 'vit'):
            vit = backbone.vit
            out = vit.forward_features(images)
            # Handle both [B, tokens, C] and [B, C] shapes
            if out.dim() == 3:
                feats = out[:, 0]  # CLS token
            else:
                feats = out
        else:
            # Most models have forward_features or we can use full forward
            if hasattr(backbone, 'forward_features'):
                feats = backbone.forward_features(images)
            else:
                # Fallback to regular forward pass
                feats = backbone(images)
            
            if feats.dim() > 2:
                feats = torch.flatten(feats, 1)
        
        return feats

    def extract_features(self, loader, backbone, device, max_batches=999999):
        """Extract features and labels from a dataloader."""
        feats_list, y_list = [], []

        for batch_idx, (views, y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = (
                views[0]. to(device, non_blocking=True)
                if isinstance(views, (list, tuple))
                else views.to(device)
            )

            with torch.no_grad():
                feats = self._extract_backbone_features(backbone, images)
                feats = F.normalize(feats, dim=1)

            feats_list.append(feats.cpu())
            y_list.append(y.cpu())

        return torch.cat(feats_list, dim=0), torch.cat(y_list, dim=0)

    def on_train_epoch_end(
        self, trainer: pl. Trainer, pl_module: pl. LightningModule
    ):
        if not self.enabled:
            return

        epoch = trainer.current_epoch + 1
        if self.every_n_epochs is not None and int(self.every_n_epochs) > 0:
            if epoch % int(self.every_n_epochs) != 0:
                return

        dm = trainer.datamodule
        if dm is None:
            return

        # Synchronize all ranks before running the expensive CDNV computation on rank 0.
        if trainer.strategy is not None:
            trainer.strategy.barrier()
        if not trainer.is_global_zero:
            if trainer.strategy is not None:
                trainer.strategy.barrier()
            return

        # IMPORTANT: preserve mode and set eval for stable features
        was_training = pl_module.training
        pl_module.eval()

        try:
            device = pl_module.device
            backbone = pl_module.backbone

            # Initialize evaluator with the correct device
            if self.evaluator is None:
                self.evaluator = GeometricEvaluator(
                    num_classes=self.num_classes,
                    device=device,
                )

            if self.compute_on_train:
                # Use the standard training loader so labels are a single Tensor.
                # (Synthetic probe loaders may return a dict of multiple labelings.)
                train_loader = dm.train_dataloader()
                Xtr, Ytr = self.extract_features(
                    train_loader, backbone, device, max_batches=self.max_train_batches
                )

                train_cdnv = self.evaluator.compute_cdnv(Xtr, Ytr)
                train_dir_cdnv = self.evaluator.compute_directional_cdnv(Xtr, Ytr)

                if train_cdnv is not None:
                    pl_module.log(
                        "cdnv/train_cdnv",
                        train_cdnv,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False,
                    )
                    pl_module.print(f"[CDNV] epoch={epoch} train_cdnv={train_cdnv:.6f}")

                if train_dir_cdnv is not None:
                    pl_module.log(
                        "cdnv/train_dir_cdnv",
                        train_dir_cdnv,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False,
                    )
                    pl_module.print(f"[CDNV] epoch={epoch} train_dir_cdnv={train_dir_cdnv:.6f}")

            if self.compute_on_val:
                val_loader = dm.val_dataloader()
                Xva, Yva = self.extract_features(
                    val_loader, backbone, device, max_batches=self.max_val_batches
                )

                val_cdnv = self.evaluator.compute_cdnv(Xva, Yva)
                val_dir_cdnv = self.evaluator.compute_directional_cdnv(Xva, Yva)

                if val_cdnv is not None:
                    pl_module.log(
                        "cdnv/val_cdnv",
                        val_cdnv,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False,
                    )
                    pl_module.print(f"[CDNV] epoch={epoch} val_cdnv={val_cdnv:.6f}")

                if val_dir_cdnv is not None:
                    pl_module.log(
                        "cdnv/val_dir_cdnv",
                        val_dir_cdnv,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False,
                    )
                    pl_module.print(f"[CDNV] epoch={epoch} val_dir_cdnv={val_dir_cdnv:.6f}")

        finally:
            if was_training:
                pl_module.train()
            # signal completion to other ranks
            if trainer.strategy is not None:
                trainer.strategy.barrier()