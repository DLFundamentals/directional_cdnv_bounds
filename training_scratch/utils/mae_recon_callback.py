import torch
import pytorch_lightning as pl

class MAEReconCallback(pl.Callback):
    def __init__(self, every_n_epochs=10, num_images=8, enabled=True, tag="viz/mae_recon"):
        self.every_n_epochs = every_n_epochs
        self.num_images = num_images
        self.enabled = enabled
        self._fixed_batch = None  # cached CPU batch for consistency
        self.tag = tag

    def on_fit_start(self, trainer, pl_module):
        if not self.enabled:
            return
        if not trainer.is_global_zero:
            return

        # Cache one batch from val dataloader for consistent visuals
        dm = trainer.datamodule
        if dm is None:
            return
        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))
        self._fixed_batch = batch  # keep on CPU

    @staticmethod
    def _unpatchify(patches, p, h, w):
        # patches: [B, L, p*p*3], where L = h*w
        B, L, D = patches.shape
        assert L == h * w
        patches = patches.reshape(B, h, w, p, p, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4)  # B,3,h,p,w,p
        imgs = patches.reshape(B, 3, h * p, w * p)
        return imgs
    
    @staticmethod
    def unnormalize(img):
        # img: Tensor [3, H, W], normalized
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
        return img * std + mean


    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.enabled or not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return
        if self._fixed_batch is None:
            return

        views, _ = self._fixed_batch
        images = views[0][: self.num_images].to(pl_module.device, non_blocking=True)

        # Use the same masking logic as training_step
        B = images.shape[0]
        idx_keep, idx_mask = pl_module._sample_mask(B, images.device)  # we’ll add this helper to your module

        with torch.no_grad():
            x_encoded = pl_module.forward_encoder(images=images, idx_keep=idx_keep)
            x_pred = pl_module.forward_decoder(x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)

            patches = pl_module._patchify(images)  # [B, L, D]
            # idx_mask includes CLS token; patches do not, so subtract 1
            masked_idx = idx_mask - 1

            # Fill in predicted patches for masked locations
            patches_recon = patches.clone()
            patches_recon = pl_module._set_at_index(patches_recon, masked_idx, x_pred)

            # Unpatchify to image
            p = pl_module.patch_size
            H = images.shape[-2]
            W = images.shape[-1]
            h = H // p
            w = W // p
            recon = self._unpatchify(patches_recon, p, h, w)

        # Log to W&B if available
        logger = trainer.logger
        if logger is not None and logger.__class__.__name__ == "WandbLogger":
            import wandb
            # Make a simple side-by-side grid: original then recon
            viz = []
            for i in range(B):
                # unnormalize before visualizing
                orig = self.unnormalize(images[i]).clamp(0,1).float().cpu()
                recon_un = self.unnormalize(recon[i]).clamp(0,1).float().cpu()
                sep = torch.ones(3, orig.shape[1], 4, device="cpu")  # white bar
                pair = torch.cat([orig, sep, recon_un], dim=2)
                viz.append(wandb.Image(
                                        pair,
                                        caption=f"epoch={epoch} | left=orig, right=recon | idx={i}"
                        ))
            logger.experiment.log({f"{self.tag}": viz, "trainer/global_step": trainer.global_step})
        else:
            # fallback: just log a scalar so you know it ran
            pl_module.log("viz/recon_logged", 1.0, on_step=False, on_epoch=True)
