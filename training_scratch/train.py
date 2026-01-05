import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb
import faulthandler, signal


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mae import LightlyMAE
from models.vicreg import LightlyVICReg
from models.dino import LightlyDINO
from models.ijepa import LightlyIJepa
from data.mini_imagenet_datamodule import MiniImageNetDataModule, MiniImageNetCfg
from utils.export_teacher import export_teacher_encoder_only
from utils.ckpt_schedule import ScheduledCheckpoint
from utils.linear_probe_callback import LinearProbeCallback
from utils.cdnv_callback import CDNVCallback
from utils.mae_recon_callback import MAEReconCallback

@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="exp/ijepa_vitB_mini.yaml",
)
def main(cfg: DictConfig):

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
    elif cfg.method.name.lower() == "dino":
        model = LightlyDINO(cfg)
    elif cfg.method.name.lower() == "ijepa":
        model = LightlyIJepa(cfg)
    else:
        raise ValueError(f"Unknown method: {cfg.method.name}. Supported: 'mae', 'vicreg', 'dino'")

    # custom model checkpointing & logging
    sched_cb = ScheduledCheckpoint(
        dirpath=cfg.ckpt_schedule.dirpath,
        early_every=cfg.ckpt_schedule.early_every,
        early_until=cfg.ckpt_schedule.early_until,
        late_every=cfg.ckpt_schedule.late_every,
        save_last=cfg.ckpt_schedule.save_last,
    )
    # Instantiate logger only on rank 0 to avoid hangs when running under DDP.
    # Check several common environment variables that indicate rank.
    def _is_rank0():
        for k in ("PL_GLOBAL_RANK", "GLOBAL_RANK", "RANK", "LOCAL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
            v = os.environ.get(k)
            if v is not None:
                try:
                    return int(v) == 0
                except Exception:
                    continue
        # If none are set, assume single-process (rank 0)
        return True

    if _is_rank0():
        if cfg.logging.backend == "wandb":
            logger = WandbLogger(
                project=cfg.logging.project,
                # entity=cfg.logging.entity,
                name=cfg.logging.run_name,
                log_model=cfg.logging.log_model,
                tags=list(cfg.logging.tags)
            )
        else:
            logger = CSVLogger(save_dir=cfg.paths.exp_dir, name="logs")
    else:
        logger = None

    # linear probe callback
    probe_cb = LinearProbeCallback(**cfg.probe)
    
    # CDNV callback
    cdnv_cb = CDNVCallback(**cfg.cdnv)
    
    # reconstruction callback (only for MAE)
    callbacks = [sched_cb, probe_cb, cdnv_cb]
    if cfg.method.name.lower() == "mae" and cfg.viz.enabled:
        viz_cb = MAEReconCallback(**cfg.viz)
        callbacks.append(viz_cb)
    
    print("=== Callbacks configured ===")
    for cb in callbacks:
        print(type(cb))
    print("============================")
    # trainer
    # Normalize strategy: if using plain 'ddp', enable find_unused_parameters to
    # avoid errors when some model parameters (e.g., frozen teacher) are unused.
    strategy = cfg.trainer.strategy if "strategy" in cfg.trainer else None
    if isinstance(strategy, str) and strategy == "ddp":
        strategy = "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
        default_root_dir=cfg.paths.exp_dir,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        strategy=strategy,
        max_epochs=cfg.trainer.max_epochs,
        use_distributed_sampler=cfg.trainer.use_distributed_sampler if "use_distributed_sampler" in cfg.trainer else False,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        precision=cfg.precision,
        callbacks=callbacks,
        enable_checkpointing=False, # since using custom checkpoint callback
        logger=logger,
    )

    # Lightning automatically calls setup() in each distributed process                                                                                                                                                         
    # Do NOT call data_module.setup() manually before trainer.fit() in DDP
    trainer.fit(model, datamodule=data_module)

    # export after training (only on global rank 0)
    if trainer.is_global_zero:
        export_path = f"{cfg.paths.exp_dir}/teacher_encoder_only.pt"
        export_teacher_encoder_only(model, export_path, extra_meta={"img_size": cfg.data.img_size})
        print(f"Exported teacher encoder to: {export_path}")


if __name__ == "__main__":
    main()