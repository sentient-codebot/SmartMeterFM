import sys
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm


class GlobalProgressBar(TQDMProgressBar):
    """Custom progress bar that shows global step progress instead of per-epoch progress.

    This progress bar will show steps/max_steps instead of batches within the current epoch.

    Args:
        refresh_rate: Determines at which rate (in number of steps) the progress bars get updated.
        process_position: Offset for multiple progress bars.
        leave: Whether to leave the progress bar on screen after completion.
    """

    def __init__(
        self,
        refresh_rate: int = 1,
        process_position: int = 0,
        leave: bool = False,
    ):
        super().__init__(
            refresh_rate=refresh_rate, process_position=process_position, leave=leave
        )
        self._enabled = True
        self._progress_bar: Tqdm | None = None

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = Tqdm(
            desc="Training",
            initial=0,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )
        return bar

    def on_train_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        self.train_progress_bar = self.init_train_tqdm()
        # Set total to max_steps instead of batches in epoch
        total_steps = (
            trainer.max_steps
            if trainer.max_steps != -1
            else trainer.max_epochs * self.total_train_batches
        )
        self.train_progress_bar.reset(total=total_steps)

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        # Don't reset the progress bar at epoch start
        pass

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        n = trainer.global_step
        if self._should_update(n, self.train_progress_bar.total):
            self.train_progress_bar.n = n
            self.train_progress_bar.refresh()
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        # Update the description to show both epoch and global progress
        self.train_progress_bar.set_description(
            f"Epoch {trainer.current_epoch}/{trainer.max_epochs-1}"
        )
