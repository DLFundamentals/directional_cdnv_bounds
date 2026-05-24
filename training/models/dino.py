import copy
from functools import partial

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf

import pytorch_lightning as pl
from timm.models.vision_transformer import vit_base_patch16_224
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import DINOv2ProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
)
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule


def freeze_eval_module(module: nn.Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class DINOv2Head(nn.Module):
    """Combined head for DINO and iBOT predictions."""
    def __init__(
        self, dino_head: DINOv2ProjectionHead, ibot_head: DINOv2ProjectionHead
    ) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2Model(nn.Module):
    """
    DINOv2 model with student-teacher architecture.
    Uses Vision Transformer Base backbone with multi-level objectives.
    """
    def __init__(
        self,
        ibot_separate_head: bool = False,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # Build ViT-Base backbone
        vit_teacher = vit_base_patch16_224(
            pos_embed="learn",
            dynamic_img_size=True,
            init_values=1e-5,
        )
        self.teacher_backbone = MaskedVisionTransformerTIMM(
            vit=vit_teacher,
            antialias=False,
            pos_embed_initialization="skip",
        )
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        update_drop_path_rate(
            self.student_backbone.vit,
            drop_path_rate=drop_path_rate,
            mode="uniform",
        )

        freeze_eval_module(self.teacher_backbone)

        # Projection heads for DINO and iBOT
        dino_head = partial(
            DINOv2ProjectionHead,
            input_dim=768,
        )

        teacher_dino_head = dino_head()
        student_dino_head = dino_head()

        ibot_head = partial(
            DINOv2ProjectionHead,
            input_dim=768,
        )

        if ibot_separate_head:
            teacher_ibot_head = ibot_head()
            student_ibot_head = ibot_head()
        else:
            teacher_ibot_head = teacher_dino_head
            student_ibot_head = student_dino_head

        self.teacher_head = DINOv2Head(
            dino_head=teacher_dino_head,
            ibot_head=teacher_ibot_head,
        )
        self.student_head = DINOv2Head(
            dino_head=student_dino_head,
            ibot_head=student_ibot_head,
        )

        freeze_eval_module(self.teacher_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through teacher backbone for inference."""
        return self.teacher_backbone(x)

    def forward_teacher(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward teacher: returns cls tokens and all features."""
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward student: returns cls tokens and masked features."""
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features


class LightlyDINO(pl.LightningModule):
    """
    Lightly DINOv2 implementation compatible with our Trainer/DataModule.
    Expects batches shaped like:
        batch = (views, labels)
    where views is a list/tuple of augmented views (global + local crops).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if OmegaConf.is_config(cfg):
            self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        else:
            self.save_hyperparameters(cfg)
            self.cfg = OmegaConf.create(cfg)

        # Build model
        self.model = DINOv2Model(
            ibot_separate_head=cfg.model.get("ibot_separate_head", False),
            drop_path_rate=cfg.model.get("drop_path_rate", 0.1),
        )

        # Create loss functions
        self.dino_criterion = DINOLoss()
        self.ibot_criterion = IBOTPatchLoss()
        self.koleo_criterion = KoLeoLoss()

        # Mask parameters
        self.mask_patch_size = cfg.model.get("mask_patch_size", 8)
        self.mask_ratio = cfg.model.get("mask_ratio", 0.6)

    def forward(self, x):
        """Forward pass through teacher backbone."""
        return self.model.forward(x)

    @property
    def backbone(self):
        """Expose the student backbone for probes/CDNV callbacks."""
        return self.model.student_backbone

    def training_step(self, batch, batch_idx):
        if self.global_step == 0 and self.global_rank == 0:
            self.print(
                f"world_size={self.trainer.world_size}, per_gpu_bs={self.cfg.data.batch_size}"
            )

        views, _ = batch
        # views should contain at least 2 global crops and multiple local crops
        global_views = torch.cat(views[:2])  # First 2 are global crops
        local_views = torch.cat(views[2:]) if len(views) > 2 else None

        # For synthetic_shapes, skip iBOT patch masking (random_block_mask) which
        # is tuned for 224x224 crops and fails on very small grids (e.g., 4x4).
        is_synthetic = (
            hasattr(self.cfg, "data")
            and hasattr(self.cfg.data, "name")
            and self.cfg.data.name == "synthetic_shapes"
        )

        if is_synthetic:
            # Teacher forward without patch-level masking
            with torch.no_grad():
                teacher_cls_token, _ = self.model.forward_teacher(global_views)
                teacher_cls_out = self.model.teacher_head.dino_head(teacher_cls_token)

            # Student forward on global views without mask
            student_global_cls_token, _ = self.model.forward_student(
                global_views, mask=None
            )
            student_global_cls_out = self.model.student_head.dino_head(
                student_global_cls_token
            )

            # No patch-level features when iBOT is disabled
            teacher_masked_out = None
            student_global_masked_out = None
            block_mask = None

        else:
            # Teacher forward (no gradient) and dynamic masking pattern for iBOT loss
            B = len(global_views)
            with torch.no_grad():
                teacher_cls_token, teacher_features = self.model.forward_teacher(
                    global_views
                )

            # sequence_length depends on the actual crop size; infer it from features
            sequence_length = teacher_features.shape[1]
            mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)

            # Mask patches except class token
            # Infer patch grid (H, W) from sequence_length - 1 instead of relying on
            # vit.patch_embed.grid_size, which is fixed for 224x224.
            import math

            tokens_no_cls = sequence_length - 1
            H = W = int(math.isqrt(tokens_no_cls))
            assert (
                H * W == tokens_no_cls
            ), f"Unexpected token count (without CLS): {tokens_no_cls}"
            block_mask = random_block_mask(size=(B, H, W), device=mask.device)
            mask[:, 1:] = block_mask.flatten(start_dim=1)

            # Compute teacher heads using the dynamically-sized mask
            with torch.no_grad():
                teacher_cls_out = self.model.teacher_head.dino_head(teacher_cls_token)
                teacher_masked_out = self.model.teacher_head.ibot_head(
                    teacher_features[mask]
                )

            # Student forward on global views
            (
                student_global_cls_token,
                student_global_masked_features,
            ) = self.model.forward_student(global_views, mask=mask)
            student_global_cls_out = self.model.student_head.dino_head(
                student_global_cls_token
            )
            student_global_masked_out = self.model.student_head.ibot_head(
                student_global_masked_features
            )

        # Student forward on local views (no masking for local crops)
        # Note: Local views have different batch size (4x global), so we process
        # them separately after the main DINO loss
        if local_views is not None:
            student_local_cls_token, _ = self.model.forward_student(
                local_views, mask=None
            )
            student_local_cls_out = self.model.student_head.dino_head(
                student_local_cls_token
            )
        else:
            student_local_cls_out = None

        # Calculate current global step
        total_steps = (
            self.cfg.trainer.max_epochs * len(self.trainer.train_dataloader)
        )
        global_step = (
            self.current_epoch * len(self.trainer.train_dataloader) + batch_idx
        )

        # Temperature schedule for teacher
        teacher_temp = linear_warmup_schedule(
            step=global_step,
            warmup_steps=int(30 / self.cfg.trainer.max_epochs * total_steps),
            start_value=0.04,
            end_value=0.07,
        )

        # Calculate losses
        # DINO loss: only on global views (teacher has 2 global views, student has 2 global + local)
        dino_loss = self.dino_criterion(
            teacher_out=list(teacher_cls_out.chunk(2)),
            student_out=list(student_global_cls_out.chunk(2)),
            teacher_temp=teacher_temp,
        )

        # iBOT loss: disabled for synthetic_shapes, enabled otherwise
        if is_synthetic:
            ibot_loss = torch.zeros((), device=self.device, dtype=dino_loss.dtype)
        else:
            ibot_loss = self.ibot_criterion(
                teacher_out=teacher_masked_out,
                student_out=student_global_masked_out,
                mask=block_mask,
                teacher_temp=teacher_temp,
            )
        koleo_loss = 0.1 * sum(
            self.koleo_criterion(t) for t in student_global_cls_token.chunk(2)
        )

        loss = dino_loss + ibot_loss + koleo_loss

        # Log losses
        self.log(
            "train/dino_loss",
            dino_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/ibot_loss",
            ibot_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/koleo_loss",
            koleo_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/total_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/teacher_temp",
            teacher_temp,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher network via momentum encoding after each batch."""
        total_steps = (
            self.cfg.trainer.max_epochs * len(self.trainer.train_dataloader)
        )
        global_step = (
            self.current_epoch * len(self.trainer.train_dataloader) + batch_idx
        )

        # Momentum schedule
        momentum = cosine_schedule(
            step=global_step,
            max_steps=total_steps,
            start_value=0.992,
            end_value=1.0,
        )

        # Update momentum
        update_momentum(
            self.model.student_backbone, self.model.teacher_backbone, m=momentum
        )
        update_momentum(self.model.student_head, self.model.teacher_head, m=momentum)

    def on_train_epoch_end(self):
        """Log learning rate at end of epoch."""
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        # Scale learning rate by world size and batch size
        scaled_lr = (
            self.cfg.model.lr
            * self.trainer.world_size
            * self.cfg.data.batch_size
            / 256.0
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=scaled_lr,
            weight_decay=self.cfg.model.weight_decay,
        )

        # Warmup + cosine decay scheduler
        warmup_epochs = self.cfg.model.warmup_epochs
        max_epochs = self.cfg.trainer.max_epochs
        min_lr = self.cfg.model.min_lr

        def lr_lambda(epoch):
            # epoch is 0-indexed
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / max(1, warmup_epochs)

            # Cosine decay after warmup
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
            },
        }
