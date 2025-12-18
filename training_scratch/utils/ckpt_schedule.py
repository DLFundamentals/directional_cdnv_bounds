import os
import pytorch_lightning as pl

class ScheduledCheckpoint(pl.Callback):
    def __init__(self, dirpath: str, early_every: int, early_until: int, late_every: int, save_last: bool = True):
        self.dirpath = dirpath
        self.early_every = early_every
        self.early_until = early_until
        self.late_every = late_every
        self.save_last = save_last
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch + 1  # 1-index for humans

        # decide whether to save
        if epoch <= self.early_until:
            do_save = (epoch % self.early_every == 0)
        else:
            do_save = (epoch % self.late_every == 0)

        if do_save:
            path = os.path.join(self.dirpath, f"epoch_{epoch:04d}.ckpt")
            trainer.save_checkpoint(path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            path = os.path.join(self.dirpath, "epoch_0000.ckpt")
            trainer.save_checkpoint(path)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.save_last and trainer.is_global_zero:
            path = os.path.join(self.dirpath, "last.ckpt")
            trainer.save_checkpoint(path)
