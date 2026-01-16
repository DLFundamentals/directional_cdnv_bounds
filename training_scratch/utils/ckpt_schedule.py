import os
import pytorch_lightning as pl


class ScheduledCheckpoint(pl.Callback):
    def __init__(self, dirpath: str, early_every: int, early_until: int, late_every: int, save_last: bool = True):
        self.dirpath = dirpath
        self.early_every = early_every
        self.early_until = early_until
        self.late_every = late_every
        self.save_last = save_last

    def _ensure_dir(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Make dir on rank 0, then sync so others don't race
        enabled = pl_module.cfg.stage2.enabled
        if enabled:
            active_path = os.path.join(self.dirpath, "stage2")
        else:
            active_path = os.path.join(self.dirpath, "stage1")
        if trainer.is_global_zero:
            os.makedirs(active_path, exist_ok=True)
        if trainer.strategy is not None:
            trainer.strategy.barrier()
        
        return active_path

    def _save_all_ranks(self, trainer: pl.Trainer, path: str):
        # IMPORTANT: call on all ranks to avoid DDP deadlocks
        trainer.save_checkpoint(path)
        if trainer.strategy is not None:
            trainer.strategy.barrier()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        active_path = self._ensure_dir(trainer, pl_module)
        path = os.path.join(active_path, "epoch_0000.ckpt")
        self._save_all_ranks(trainer, path)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch + 1 # epoch is 0-indexed

        if epoch <= self.early_until:
            do_save = (epoch % self.early_every == 0)
        else:
            do_save = (epoch % self.late_every == 0)

        if not do_save:
            return

        active_path = self._ensure_dir(trainer, pl_module)
        path = os.path.join(active_path, f"epoch_{epoch:04d}.ckpt")
        self._save_all_ranks(trainer, path)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.save_last:
            return
        active_path = self._ensure_dir(trainer, pl_module)
        path = os.path.join(active_path, "last.ckpt")
        self._save_all_ranks(trainer, path)