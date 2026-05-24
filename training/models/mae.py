import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # -------------------------
        # Stage-2: frozen reference model f0
        # -------------------------
        self.stage2_enabled = bool(getattr(cfg, "stage2", {}).get("enabled", False)) if OmegaConf.is_config(cfg) else bool(getattr(cfg, "stage2", {}).enabled)
        self.backbone_ref = None


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
    
    def on_fit_start(self):
        if self.stage2_enabled:
            print("=== Stage-2 enabled: creating frozen reference model ===")
            self.backbone_ref = copy.deepcopy(self.backbone)
            self.backbone_ref.eval()
            for p in self.backbone_ref.parameters():
                p.requires_grad = False

            # print("=== Stage-2: comparing online and reference model parameters ===")
            # # 2) dtype/device
            # print("online dtype/device:", next(self.backbone.parameters()).dtype, next(self.backbone.parameters()).device)
            # print("ref    dtype/device:", next(self.backbone_ref.parameters()).dtype, next(self.backbone_ref.parameters()).device)
            # # 1) exact parameter equality check on one tensor
            # k = next(iter(dict(self.backbone.named_parameters()).keys()))
            # p1 = dict(self.backbone.named_parameters())[k].detach().cpu()
            # p0 = dict(self.backbone_ref.named_parameters())[k].detach().cpu()
            # print("param max abs diff:", (p1 - p0).abs().max().item())
            # # 3) feature diff
            # self.backbone.eval()
            # self.backbone_ref.eval()
            # images = torch.randn(2, 3, self.cfg.data.img_size, self.cfg.data.img_size).to('cuda')
            # with torch.no_grad():
            #     f1 = self.extract_phi(self.backbone, images).detach().float().cpu()
            #     f0 = self.extract_phi(self.backbone_ref, images).detach().float().cpu()
            # print("feat max abs diff:", (f1 - f0).abs().max().item())


    def training_step(self, batch, batch_idx):
        if self.global_step == 0 and self.global_rank == 0:
            self.print(f"world_size={self.trainer.world_size}, per_gpu_bs={self.cfg.data.batch_size}")

        views, labels = batch
        images = views[0]  # MAETransform returns one view

        loss_mae = None
        if (not self.stage2_enabled) or self.cfg.stage2.keep_mae_loss:
            # your existing MAE loss code (unchanged)
            batch_size = images.shape[0]
            idx_keep, idx_mask = utils.random_token_mask(
                size=(batch_size, self.sequence_length),
                mask_ratio=self.mask_ratio,
                device=images.device,
            )
            x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
            x_pred = self.forward_decoder(x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)

            patches = utils.patchify(images, self.patch_size)
            target = utils.get_at_index(patches, idx_mask - 1)

            loss_mae = self.criterion(x_pred, target)
            self.log("train/mae_loss", loss_mae, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        loss_anchor = None
        loss_dir_cdnv = None
        if self.stage2_enabled:
            # self.backbone.eval() # sanity check
            feat_online = self.extract_phi(self.backbone, images)
            with torch.no_grad():
                feat_ref = self.extract_phi(self.backbone_ref, images)

            loss_anchor = F.mse_loss(feat_online, feat_ref)
            self.log("stage2/anchor_l2", loss_anchor, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

            loss_dir_cdnv = self.dir_cdnv_loss_batch(feat_online, labels, min_class_count=self.cfg.stage2.min_class_count)
            self.log("stage2/dir_cdnv_loss", loss_dir_cdnv, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if self.stage2_enabled:
            total = self.cfg.stage2.anchor_weight * loss_anchor + self.cfg.stage2.lambda_dir * loss_dir_cdnv
            if self.cfg.stage2.keep_mae_loss and loss_mae is not None:
                total = total + loss_mae
            self.log("stage2/total_loss", total, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            return total
        else:
            return loss_mae


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

    @staticmethod
    def _set_at_index(x, idx, values):
        # idx: [B, K] selecting K tokens per batch
        return utils.set_at_index(x, idx, values)

    def extract_phi(self, backbone, images):
        """
        Returns CLS features used for anchoring / geometry losses.
        Matches CDNVCallback behavior: vit.forward_features -> CLS token.
        """
        vit = backbone.vit
        out = vit.forward_features(images)
        if out.dim() == 3:
            return out[:, 0]  # CLS token
        return out
    
    def on_save_checkpoint(self, checkpoint):
        sd = checkpoint.get("state_dict", {})
        for k in list(sd.keys()):
            if k.startswith("backbone_ref."):
                sd.pop(k)

    def dir_cdnv_loss_batch(
        self,
        feats: torch.Tensor,          # [B, D]
        labels: torch.Tensor,         # [B]
        min_class_count: int = 2,
        max_pairs: int | None = None, # set e.g. 50 to subsample pairs; None = all pairs
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Differentiable minibatch approximation of your compute_directional_cdnv().

        For ordered class pairs (i -> j):
        v = mu_j - mu_i
        dir_var = E_{x in i} [ ((x - mu_i) · v_hat)^2 ]
        dir_cdnv = dir_var / ||v||^2
        Returns average over selected ordered pairs.
        """
        device = feats.device
        feats_f = feats.float()
        y = labels.to(device)

        # classes present in batch
        classes, counts = torch.unique(y, return_counts=True)
        valid = counts >= min_class_count
        classes = classes[valid]

        if classes.numel() < 2:
            return feats_f.new_zeros(())  # scalar 0

        # compute means per class (only for valid classes)
        means = {}
        for c in classes.tolist():
            idx = (y == c).nonzero(as_tuple=True)[0]
            means[c] = feats_f.index_select(0, idx).mean(dim=0)  # [D]

        # build list of ordered pairs
        cls_list = classes.tolist()
        pairs = [(i, j) for i in cls_list for j in cls_list if i != j]
        if len(pairs) == 0:
            return feats_f.new_zeros(())

        # optionally subsample pairs for speed
        if max_pairs is not None and len(pairs) > max_pairs:
            # torch-based sampling for determinism across devices
            perm = torch.randperm(len(pairs), device=device)[:max_pairs].tolist()
            pairs = [pairs[k] for k in perm]

        total = feats_f.new_zeros(())
        num_pairs = 0

        for i, j in pairs:
            mu_i = means[i]
            mu_j = means[j]

            v = mu_j - mu_i
            v_norm2 = torch.dot(v, v)  # ||v||^2
            if torch.isfinite(v_norm2) and v_norm2.item() > 0:
                v_hat = v / (v_norm2.sqrt() + eps)

                idx_i = (y == i).nonzero(as_tuple=True)[0]
                x_i = feats_f.index_select(0, idx_i)  # [Ni, D]
                proj = (x_i - mu_i) @ v_hat           # [Ni]
                dir_var = (proj ** 2).mean()          # scalar
                dir_cdnv = dir_var / (v_norm2 + eps)  # scalar

                total = total + dir_cdnv
                num_pairs += 1

        if num_pairs == 0:
            return feats_f.new_zeros(())

        dir_cdnv = total / num_pairs
        log_dir_cdnv = torch.log10(dir_cdnv + 1e-8)
        return log_dir_cdnv
