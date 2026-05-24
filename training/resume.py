import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mae import LightlyMAE
from models.vicreg import LightlyVICReg
from data.mini_imagenet_datamodule import MiniImageNetDataModule, MiniImageNetCfg
from utils.export_teacher import export_teacher_encoder_only
from utils.ckpt_schedule import ScheduledCheckpoint
from utils.linear_probe_callback import LinearProbeCallback
from utils.cdnv_callback import CDNVCallback
from utils.mae_recon_callback import MAEReconCallback


@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="exp/vicreg_resnet50",
)
def main(cfg: DictConfig):

    # Allow specifying resume options either via environment variables or cfg keys.
    # This keeps the runtime structure identical to `train.py` while letting users
    # provide the checkpoint path and optional overrides.
    resume_ckpt = os.environ.get('RESUME_CKPT') or (cfg.get('resume_ckpt') if 'resume_ckpt' in cfg else None)
    if resume_ckpt is None:
        raise RuntimeError('Missing checkpoint path to resume. Set RESUME_CKPT env or add resume_ckpt to the config.')

    new_cdnv_every = os.environ.get('NEW_CDNV_EVERY_N_EPOCHS')
    new_cdnv_every = int(new_cdnv_every) if new_cdnv_every is not None else None

    wandb_run_id_override = os.environ.get('WANDB_RUN_ID')

    print("\n========== HYDRA CONFIG ==========")
    print(OmegaConf.to_yaml(cfg))
    print("=================================\n")

    # build data module
    data_cfg = MiniImageNetCfg(**cfg.data)
    data_module = MiniImageNetDataModule(data_cfg)

    # build model based on method
    if cfg.method.name.lower() == "mae":
        model = LightlyMAE(cfg)
    elif cfg.method.name.lower() == "vicreg":
        model = LightlyVICReg(cfg)
    else:
        raise ValueError(f"Unknown method: {cfg.method.name}. Supported: 'mae', 'vicreg'")

    # custom model checkpointing & logging
    sched_cb = ScheduledCheckpoint(
        dirpath=cfg.ckpt_schedule.dirpath,
        early_every=cfg.ckpt_schedule.early_every,
        early_until=cfg.ckpt_schedule.early_until,
        late_every=cfg.ckpt_schedule.late_every,
        save_last=cfg.ckpt_schedule.save_last,
    )
    if cfg.logging.backend == "wandb":
        # Try to extract run id from checkpoint to resume the same wandb run
        ckpt_meta = None
        try:
            ckpt_meta = torch.load(resume_ckpt, map_location='cpu')
        except Exception:
            ckpt_meta = None

        resume_run_id = wandb_run_id_override or (ckpt_meta.get('wandb_run_id') if isinstance(ckpt_meta, dict) and 'wandb_run_id' in ckpt_meta else None)
        init_kwargs = dict(
            project=cfg.logging.project,
            # entity=cfg.logging.entity,
            name=cfg.logging.run_name,
            # do not set log_model here; WandbLogger handles that
            tags=list(cfg.logging.tags) if 'tags' in cfg.logging else None,
        )
        if resume_run_id:
            wandb.init(id=resume_run_id, resume='allow', **{k: v for k, v in init_kwargs.items() if v is not None})
        else:
            wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})

        logger = WandbLogger(
            project=cfg.logging.project,
            # entity=cfg.logging.entity,
            name=cfg.logging.run_name,
            log_model=cfg.logging.log_model,
            tags=list(cfg.logging.tags)
        )                       
    else:
        logger = CSVLogger(save_dir=cfg.paths.exp_dir, name="logs")

    # linear probe callback
    probe_cb = LinearProbeCallback(**cfg.probe)
    # CDNV callback
    cdnv_cb = CDNVCallback(**cfg.cdnv)

    # reconstruction callback (only for MAE)
    callbacks = [sched_cb, probe_cb, cdnv_cb]
    if cfg.method.name.lower() == "mae" and cfg.viz.enabled:
        viz_cb = MAEReconCallback(**cfg.viz)
        callbacks.append(viz_cb)

    # trainer (keep same settings as train.py)
    trainer = pl.Trainer(
        default_root_dir=cfg.paths.exp_dir,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.max_epochs,
        use_distributed_sampler=cfg.trainer.use_distributed_sampler if "use_distributed_sampler" in cfg.trainer else False,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        precision=cfg.precision,
        callbacks=callbacks,
        enable_checkpointing=False, # match train.py
        logger=logger,
    )

    data_module.setup()

    # Resume training using the provided checkpoint path. This keeps checkpointing
    # and logging exactly like `train.py`, while ensuring WandB logs append to
    # the same run when a run id is available.
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)

    # export after training (only on global rank 0)
    if trainer.is_global_zero:
        export_path = f"{cfg.paths.exp_dir}/teacher_encoder_only.pt"
        export_teacher_encoder_only(model, export_path, extra_meta={"img_size": cfg.data.img_size})
        print(f"Exported teacher encoder to: {export_path}")


if __name__ == '__main__':
    main()
