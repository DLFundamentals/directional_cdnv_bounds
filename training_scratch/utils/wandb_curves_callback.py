import pytorch_lightning as pl


class WandbCurvesCallback(pl.Callback):
    """Logs a small, stable set of epoch-indexed curves to W&B.

    This is intentionally separate from Lightning's normal logger integration so
    the x-axis is exactly the epoch number (wandb step == epoch).

    Curves logged (5):
      - curves/loss
      - curves/acc/color
      - curves/acc/shape
      - curves/acc/style
      - curves/acc/size_label

    The accuracy curves are sourced from LinearProbeCallback's cached results.
    """

    def __init__(
        self,
        label_keys=("color", "shape", "style", "size_label"),
        loss_key="loss",
    ):
        super().__init__()
        self.label_keys = tuple(label_keys)
        self.loss_key = loss_key
        self._defined = False

        # Cache latest values; CDNVCallback logs only at its own frequency.
        self._last_cdnv_train = None
        self._last_dir_cdnv_train = None
        self._last_cdnv_val = None
        self._last_dir_cdnv_val = None

    @staticmethod
    def _get_wandb_experiment(trainer: pl.Trainer):
        logger = trainer.logger
        if logger is None:
            return None

        # Lightning may wrap multiple loggers.
        if hasattr(logger, "experiment") and logger.__class__.__name__ == "WandbLogger":
            return logger.experiment

        # LoggerCollection
        loggers = getattr(logger, "loggers", None)
        if loggers is not None:
            for lg in loggers:
                if lg is not None and lg.__class__.__name__ == "WandbLogger":
                    return lg.experiment

        return None

    @staticmethod
    def _find_probe_callback(trainer: pl.Trainer):
        for cb in trainer.callbacks or []:
            if cb.__class__.__name__ == "LinearProbeCallback":
                return cb
        return None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero:
            return

        exp = self._get_wandb_experiment(trainer)
        if exp is None:
            return

        # Define metrics so W&B uses epoch as the x-axis.
        try:
            exp.define_metric("epoch")
            exp.define_metric("curves/loss", step_metric="epoch")
            for k in self.label_keys:
                exp.define_metric(f"curves/acc/{k}", step_metric="epoch")

            # CDNV / directional CDNV curves (4 plots).
            exp.define_metric("curves/cdnv/train", step_metric="epoch")
            exp.define_metric("curves/dir_cdnv/train", step_metric="epoch")
            exp.define_metric("curves/cdnv/val", step_metric="epoch")
            exp.define_metric("curves/dir_cdnv/val", step_metric="epoch")
            self._defined = True
        except Exception:
            # If define_metric isn't available for some reason, logging still works.
            self._defined = False

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero:
            return

        exp = self._get_wandb_experiment(trainer)
        if exp is None:
            return

        epoch = int(trainer.current_epoch)

        metrics = {"epoch": epoch}

        cbm = trainer.callback_metrics
        loss_val = cbm.get(self.loss_key)
        if loss_val is not None:
            try:
                metrics["curves/loss"] = float(loss_val.detach().cpu().item())
            except Exception:
                try:
                    metrics["curves/loss"] = float(loss_val)
                except Exception:
                    pass

        probe_cb = self._find_probe_callback(trainer)
        last_acc = getattr(probe_cb, "last_val_acc", None) if probe_cb is not None else None
        if isinstance(last_acc, dict):
            for k in self.label_keys:
                v = last_acc.get(k)
                if v is None:
                    continue
                try:
                    metrics[f"curves/acc/{k}"] = float(v)
                except Exception:
                    pass

        # CDNV metrics (from CDNVCallback via Lightning's callback_metrics).
        def _get_float(key):
            v = cbm.get(key)
            if v is None:
                return None
            try:
                return float(v.detach().cpu().item())
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return None

        v = _get_float("cdnv/train_cdnv")
        if v is not None:
            self._last_cdnv_train = v
        v = _get_float("cdnv/train_dir_cdnv")
        if v is not None:
            self._last_dir_cdnv_train = v
        v = _get_float("cdnv/val_cdnv")
        if v is not None:
            self._last_cdnv_val = v
        v = _get_float("cdnv/val_dir_cdnv")
        if v is not None:
            self._last_dir_cdnv_val = v

        if self._last_cdnv_train is not None:
            metrics["curves/cdnv/train"] = float(self._last_cdnv_train)
        if self._last_dir_cdnv_train is not None:
            metrics["curves/dir_cdnv/train"] = float(self._last_dir_cdnv_train)
        if self._last_cdnv_val is not None:
            metrics["curves/cdnv/val"] = float(self._last_cdnv_val)
        if self._last_dir_cdnv_val is not None:
            metrics["curves/dir_cdnv/val"] = float(self._last_dir_cdnv_val)

        # Use wandb step == epoch for the requested x-axis.
        exp.log(metrics, step=epoch)
