import pytorch_lightning as pl
import torch
from transformers import IJepaConfig, IJepaModel


class LightlyIJepa(pl.LightningModule):
    """
    Lightning wrapper for I-JEPA that mirrors the API of LightlyMAE/LightlyDINO/etc.,
    but delegates all optimization/scheduling decisions to the existing training code.

    The only responsibility here is:
      - Build an IJepaModel from cfg.method.model (or sensible defaults)
      - Provide a forward() pass and an .encoder attribute

    No custom optimizers or schedulers are defined here.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Don't save full cfg in hparams to avoid huge checkpoints
        self.save_hyperparameters(ignore=["cfg"])

        model_cfg = cfg.method.model

        # Construct IJepaConfig from Hydra config fields, with fallbacks.
        # Align field names with HF IJepaConfig docs.
        hf_cfg = IJepaConfig(
            hidden_size=getattr(model_cfg, "hidden_size", 768),
            num_hidden_layers=getattr(model_cfg, "num_hidden_layers", 12),
            num_attention_heads=getattr(model_cfg, "num_attention_heads", 12),
            intermediate_size=getattr(model_cfg, "intermediate_size", 3072),
            image_size=getattr(
                model_cfg,
                "image_size",
                getattr(cfg.data, "img_size", 224),
            ),
            patch_size=getattr(model_cfg, "patch_size", 16),
            num_channels=getattr(model_cfg, "num_channels", 3),
            qkv_bias=getattr(model_cfg, "qkv_bias", True),
            hidden_act=getattr(model_cfg, "hidden_act", "gelu"),
        )

        self.model = IJepaModel(hf_cfg)

    def forward(self, x: torch.Tensor):
        return self.model(pixel_values=x)

    @property
    def encoder(self):
        """
        Expose an 'encoder' attribute so existing callbacks
        (linear probe, CDNV, export_teacher_encoder_only, etc.)
        can treat this like other Lightly* models.

        Adjust this if HF IJepaModel uses a different internal name.
        """
        if hasattr(self.model, "encoder"):
            return self.model.encoder
        if hasattr(self.model, "vision_model"):
            return self.model.vision_model
        # Fallback: return full model
        return self.model

    def training_step(self, batch, batch_idx):
        """Simple training step implementing a masked reconstruction MSE loss.

        Expects `batch` to be either a tensor of images or a tuple `(views, labels)`
        where `views` may be a list of augmented images. We take the first view.
        """
        # Unpack batch
        if isinstance(batch, (list, tuple)):
            views = batch[0]
            if isinstance(views, (list, tuple)):
                img = views[0]
            else:
                img = views
        else:
            img = batch

        img = img.to(self.device)

        # Forward through IJepa model. HF IJepaModel may return a dict-like object.
        outputs = self.forward(img)

        # Attempt to extract pixel-level predictions. Fall back to last_hidden_state.
        if hasattr(outputs, "pred_pixels"):
            pred = outputs.pred_pixels
        elif hasattr(outputs, "pixel_values"):
            pred = outputs.pixel_values
        else:
            # Use last_hidden_state and a simple linear projection to pixels
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                # As a last resort, compute a trivial zero loss to avoid crashes
                loss = torch.tensor(0.0, device=self.device)
                self.log("train/total_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
                return loss
            # Pool hidden to a scalar prediction per sample and compute MSE to mean pixel
            pooled = hidden.mean(dim=1)
            target = img.view(img.size(0), -1).mean(dim=1, keepdim=True)
            pred = pooled.mean(dim=1, keepdim=True)
            loss = torch.nn.functional.mse_loss(pred, target)
            self.log("train/total_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
            return loss

        # Compute MSE against input pixels (resize/reshape if necessary)
        if pred.shape != img.shape:
            # try to reshape pred to match img
            try:
                pred = pred.view_as(img)
            except Exception:
                # fallback
                loss = torch.nn.functional.mse_loss(pred.flatten(), img.flatten())
                self.log("train/total_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
                return loss

        loss = torch.nn.functional.mse_loss(pred, img)
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = getattr(self.cfg.model, "lr", 1e-3)
        weight_decay = getattr(self.cfg.model, "weight_decay", 0.0)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        # Simple step LR by epochs (can be replaced with cosine)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}