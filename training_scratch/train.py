import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mae import LightlyMAE
from data.mini_imagenet_datamodule import MiniImageNetDataModule, MiniImageNetCfg
from utils.export_teacher import export_teacher_encoder_only
from utils.ckpt_schedule import ScheduledCheckpoint
from utils.linear_probe_callback import LinearProbeCallback
from utils.mae_recon_callback import MAEReconCallback

@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="exp/mae_mini_vitb",
)
def main(cfg: DictConfig):

    print("\n========== HYDRA CONFIG ==========")
    print(OmegaConf.to_yaml(cfg))
    print("=================================\n")

    # build data module
    data_cfg = MiniImageNetCfg(**cfg.data)
    data_module = MiniImageNetDataModule(data_cfg)

    # build model
    model = LightlyMAE(cfg)

    # custom model checkpointing & logging
    sched_cb = ScheduledCheckpoint(
        dirpath=cfg.ckpt_schedule.dirpath,
        early_every=cfg.ckpt_schedule.early_every,
        early_until=cfg.ckpt_schedule.early_until,
        late_every=cfg.ckpt_schedule.late_every,
        save_last=cfg.ckpt_schedule.save_last,
    )
    if cfg.logging.backend == "wandb":
        logger = WandbLogger(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=cfg.logging.run_name,
            log_model=cfg.logging.log_model,
            tags=list(cfg.logging.tags)
        )

    else:
        logger = CSVLogger(save_dir=cfg.paths.exp_dir, name="logs")

    # linear probe callback
    probe_cb = LinearProbeCallback(**cfg.probe)
    # reconstruction callback
    # viz_cb = MAEReconCallback(**cfg.viz)
    # trainer
    # # resume from checkpoint if specified TODO
    # if cfg.paths.resume_from_checkpoint is not None:
    #     print(f"Resuming from checkpoint: {cfg.paths.resume_from_checkpoint}")
    #     resume_ckpt = cfg.paths.resume_from_checkpoint
    # else:
    #     resume_ckpt = None

    trainer = pl.Trainer(
        default_root_dir=cfg.paths.exp_dir,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.max_epochs,
        use_distributed_sampler=cfg.trainer.use_distributed_sampler if "use_distributed_sampler" in cfg.trainer else False,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        precision=cfg.precision,
        callbacks=[sched_cb, probe_cb],
        enable_checkpointing=False, # since using custom checkpoint callback
        logger=logger,
    )

    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # trainer.fit(model, datamodule=data_module, 
    #             ckpt_path=None) # TODO resume_ckpt

    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    # export after training (only on global rank 0)
    if trainer.is_global_zero:
        export_path = f"{cfg.paths.exp_dir}/teacher_encoder_only.pt"
        export_teacher_encoder_only(model, export_path, extra_meta={"img_size": cfg.data.img_size})
        print(f"Exported teacher encoder to: {export_path}")


if __name__ == "__main__":
    main()
