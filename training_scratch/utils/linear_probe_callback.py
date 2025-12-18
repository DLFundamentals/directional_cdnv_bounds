import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

class LinearProbeCallback(pl.Callback):
    """
    Every N epochs:
      - freeze teacher backbone
      - extract CLS features
      - train a linear classifier for a few epochs (fast)
      - log val accuracy to W&B/CSV
    """
    def __init__(self, every_n_epochs=10, max_epochs=5, lr=0.1, weight_decay=0.0,
                 max_train_batches=200, max_val_batches=50, batch_size=256, enabled=True):
        self.every_n_epochs = every_n_epochs
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.batch_size = batch_size
        self.enabled = enabled

    @staticmethod
    def _cls_features(backbone, images):
        # MaskedVisionTransformerTIMM wraps a timm ViT
        # Easiest: run the underlying ViT forward_features if available.
        vit = getattr(backbone, "vit", None)
        if vit is None:
            raise RuntimeError("Expected backbone.vit to exist (MaskedVisionTransformerTIMM(vit=...)).")

        # timm ViT commonly provides forward_features -> [B, tokens, C] or [B, C]
        out = vit.forward_features(images)

        # Handle both shapes
        if out.dim() == 3:
            cls = out[:, 0]            # [B, C]
        else:
            cls = out                 # [B, C]
        return cls

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.enabled or not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return

        dm = trainer.datamodule
        if dm is None:
            return

        # IMPORTANT: preserve mode and set eval for stable features
        was_training = pl_module.training
        pl_module.eval()

        try:
            train_loader = dm.train_dataloader()
            val_loader = dm.val_dataloader()
            device = pl_module.device

            # num classes
            num_classes = getattr(dm, "num_classes", None) or getattr(dm, "n_classes", None)
            if num_classes is None:
                ds = getattr(dm, "ds_train", None) or getattr(dm, "train_set", None)
                if ds is not None and hasattr(ds, "features") and "label" in ds.features:
                    try:
                        num_classes = int(ds.features["label"].num_classes)
                    except Exception:
                        pass
            if num_classes is None:
                # last resort: infer from a batch
                _, y0 = next(iter(train_loader))
                num_classes = int(y0.max().item() + 1)

            backbone = pl_module.backbone  # frozen for probe

            # feature dim
            with torch.no_grad():
                x0, _ = next(iter(train_loader))
                images0 = x0[0].to(device, non_blocking=True) if isinstance(x0, (list, tuple)) else x0.to(device)
                feat0 = self._cls_features(backbone, images0)
                feat_dim = feat0.shape[-1]

            # Extract fresh features EVERY probe run
            Xtr, Ytr = self.extract_features(train_loader, backbone, device, max_batches=self.max_train_batches)
            Xva, Yva = self.extract_features(val_loader, backbone, device, max_batches=self.max_val_batches)

            train_dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_dl = DataLoader(TensorDataset(Xva, Yva), batch_size=self.batch_size, shuffle=False, num_workers=0)

            clf = nn.Linear(feat_dim, num_classes).to(device)
            opt = torch.optim.AdamW(clf.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            def run_epoch_cached(dl, train=True):
                correct, total = 0, 0
                loss_sum = 0.0
                clf.train() if train else clf.eval()

                for X, y in dl:
                    X = X.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    logits = clf(X)
                    loss = F.cross_entropy(logits, y)  # mean over batch

                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()

                    bs = y.size(0)
                    correct += (logits.argmax(dim=1) == y).sum().item()
                    total += bs
                    loss_sum += loss.item() * bs  # sample-weighted

                return correct / max(total, 1), loss_sum / max(total, 1)

            for _ in range(self.max_epochs):
                train_acc, train_loss = run_epoch_cached(train_dl, train=True)

            val_acc, val_loss = run_epoch_cached(val_dl, train=False)

            pl_module.log("probe/train_acc", train_acc, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("probe/train_loss", train_loss, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("probe/val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("probe/val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)

            pl_module.print(f"[LinearProbe] epoch={epoch} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        finally:
            # restore the exact previous mode
            if was_training:
                pl_module.train()


    def extract_features(self, loader, backbone, device, max_batches=999999):
        feats_list, y_list = [], []

        for batch_idx, (views, y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = views[0].to(device, non_blocking=True) if isinstance(views, (list, tuple)) else views.to(device)

            with torch.no_grad():
                feats = self._cls_features(backbone, images)
                feats = F.normalize(feats, dim=1)

            feats_list.append(feats.cpu())
            y_list.append(y.cpu())

        return torch.cat(feats_list, dim=0), torch.cat(y_list, dim=0)

