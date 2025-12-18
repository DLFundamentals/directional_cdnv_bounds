import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf

import pytorch_lightning as pl
from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTIMM, MAEDecoderTIMM

# timm vit builders
from timm.models.vision_transformer import vit_base_patch32_224, vit_base_patch16_224, vit_large_patch16_224


def build_vit(vit_name: str):
    vit_name = vit_name.lower()
    if vit_name == "vit_base_patch32_224":
        return vit_base_patch32_224()
    if vit_name == "vit_base_patch16_224":
        return vit_base_patch16_224()
    if vit_name == "vit_large_patch16_224":
        return vit_large_patch16_224()
    raise ValueError(f"Unknown vit_name={vit_name}. Add it to build_vit().")

class LightlyMAE(pl.LightningModule):
    """
    Lightly MAE implementation (timm backbone) compatible with our Trainer/DataModule.
    Expects batches shaped like:
        batch = (views, labels)
    where views is a list/tuple and views[0] is images tensor [B,3,H,W].
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if OmegaConf.is_config(cfg):
            self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        else:
            self.save_hyperparameters(cfg)
            self.cfg = OmegaConf.create(cfg)
            

        vit = build_vit(cfg.stage.vit_name)

        self.mask_ratio = cfg.stage.mask_ratio
        self.patch_size = vit.patch_embed.patch_size[0]

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length

        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=cfg.stage.decoder_dim,
            decoder_depth=cfg.stage.decoder_depth,
            decoder_num_heads=cfg.stage.decoder_num_heads,
            mlp_ratio=cfg.stage.mlp_ratio,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)

        x_masked = utils.repeat_token(self.decoder.mask_token, (batch_size, self.sequence_length))
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        x_decoded = self.decoder.decode(x_masked)

        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        if self.global_step == 0 and self.global_rank == 0:
            self.print(f"world_size={self.trainer.world_size}, per_gpu_bs={self.cfg.data.batch_size}")

        views, _ = batch
        images = views[0]  # MAETransform returns one view

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)

        patches = utils.patchify(images, self.patch_size)
        target = utils.get_at_index(patches, idx_mask - 1)  # -1 for missing CLS token

        loss = self.criterion(x_pred, target)
        self.log("train/mae_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)


    def configure_optimizers(self):
        # linear scaling
        scaled_lr = self.cfg.stage.lr * self.trainer.world_size * self.cfg.data.batch_size / 256.0
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=scaled_lr,
            weight_decay=self.cfg.stage.weight_decay,
        )

        # warmup + cosine decay
        warmup_epochs = self.cfg.stage.warmup_epochs
        max_epochs = self.cfg.trainer.max_epochs
        min_lr = self.cfg.stage.min_lr

        def lr_lambda(epoch):
            # epoch is 0-indexed
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)

            t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            import math
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return (min_lr / scaled_lr) + (1 - min_lr / scaled_lr) * cosine
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
    
    def _sample_mask(self, batch_size, device):
        return utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=device,
        )

    def _patchify(self, images):
        return utils.patchify(images, self.patch_size)

    @staticmethod
    def _set_at_index(x, idx, values):
        # idx: [B, K] selecting K tokens per batch
        return utils.set_at_index(x, idx, values)