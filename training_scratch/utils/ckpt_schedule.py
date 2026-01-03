import os
import pytorch_lightning as pl


class ScheduledCheckpoint(pl.Callback):
    def __init__(self, dirpath: str, early_every: int, early_until: int, late_every: int, save_last: bool = True):
        self.dirpath = dirpath
        self.early_every = early_every
        self.early_until = early_until
        self.late_every = late_every
        self.save_last = save_last

    def _ensure_dir(self, trainer: pl.Trainer):
        # Make dir on rank 0, then sync so others don't race
        if trainer.is_global_zero:
            os.makedirs(self.dirpath, exist_ok=True)
        if trainer.strategy is not None:
            trainer.strategy.barrier()

    def _save_all_ranks(self, trainer: pl.Trainer, path: str):
        # IMPORTANT: call on all ranks to avoid DDP deadlocks
        trainer.save_checkpoint(path)
        if trainer.strategy is not None:
            trainer.strategy.barrier()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._ensure_dir(trainer)
        path = os.path.join(self.dirpath, "epoch_0000.ckpt")
        self._save_all_ranks(trainer, path)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch + 1  # 1-index for humans

        if epoch <= self.early_until:
            do_save = (epoch % self.early_every == 0)
        else:
            do_save = (epoch % self.late_every == 0)

        if not do_save:
            return

        self._ensure_dir(trainer)
        path = os.path.join(self.dirpath, f"epoch_{epoch:04d}.ckpt")
        self._save_all_ranks(trainer, path)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.save_last:
            return
        self._ensure_dir(trainer)
        path = os.path.join(self.dirpath, "last.ckpt")
        self._save_all_ranks(trainer, path)