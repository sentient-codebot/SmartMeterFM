"""Periodic sampling callback for flow matching training.

Generates samples at regular intervals during training using the EMA model,
saves them to disk, and logs visualization figures to wandb/mlflow.
"""

import calendar
import copy
import logging
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import matplotlib
import pytorch_lightning as pl
import torch
from einops import rearrange

from ...conditions import SampleCondition, WPuQCondition
from ...utils.configuration import SampleConfig
from ..plot import plot_sampled_data_v2


logger = logging.getLogger(__name__)


class PeriodicSamplingCallback(pl.Callback):
    """Generate and log samples periodically during training.

    Snapshots the EMA velocity model via deepcopy, then runs ODE integration
    and plotting in a background thread on a dedicated CUDA stream so training
    is not blocked.
    """

    def __init__(
        self,
        sample_every: int,
        sample_config: SampleConfig,
        months: list[int],
        output_dir: str,
        log_wandb: bool = False,
        log_mlflow: bool = False,
        profile_inverse_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        condition_class: type[SampleCondition] = WPuQCondition,
    ):
        super().__init__()
        self.sample_every = sample_every
        self.sample_config = sample_config
        self.months = months
        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.log_mlflow = log_mlflow
        self._inverse_transform = profile_inverse_transform
        self._condition_class = condition_class

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: Future | None = None
        self._stream: torch.cuda.Stream | None = None
        self._wandb_metric_defined: bool = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if not trainer.is_global_zero:
            return

        # Check for completed background job and log results from main thread
        if self._pending is not None and self._pending.done():
            exc = self._pending.exception()
            if exc:
                logger.error("Sampling job failed: %s", exc)
            else:
                result = self._pending.result()
                if result is not None:
                    fig_path, sample_step = result
                    self._log_figure(fig_path, sample_step, trainer)
            self._pending = None

        if step == 0 or step % self.sample_every != 0:
            return

        # Previous job still running — skip this interval
        if self._pending is not None:
            logger.warning(
                "Previous sampling job still running at step %d, skipping.", step
            )
            return

        # Snapshot velocity model (fast for small models)
        velocity_model = copy.deepcopy(pl_module.get_ema_velocity_model()).eval()

        seq_len = pl_module.model_config.seq_length
        num_ch = pl_module.num_in_channel
        device = pl_module.device

        # Lazily create side stream
        if self._stream is None and device.type == "cuda":
            self._stream = torch.cuda.Stream(device=device)

        cfg_scale = self.sample_config.cfg_scale
        if cfg_scale != 1.0:
            from smartmeterfm.models.flow import ConditionedVelocityModelWrapper

            velocity_model = ConditionedVelocityModelWrapper(
                velocity_model=velocity_model,
                dict_labels=None,  # will be set per-batch in the loop
                dict_emb_extras={"force_drop_ids": None},
                cfg_scale=cfg_scale,
            )

        create_mask = getattr(pl_module, "create_mask", False)
        steps_per_day = getattr(pl_module, "steps_per_day", 96)

        self._pending = self._executor.submit(
            self._generate_and_log,
            velocity_model,
            seq_len,
            num_ch,
            device,
            step,
            cfg_scale,
            create_mask,
            steps_per_day,
        )

    def _generate_and_log(
        self,
        velocity_model,
        seq_len,
        num_ch,
        device,
        step,
        cfg_scale,
        create_mask,
        steps_per_day,
    ):
        """Run ODE integration, save samples, and plot. Runs in background thread.

        Returns:
            Tuple of (fig_path, step) for the main thread to log, or None on failure.
        """
        from smartmeterfm.models.flow import FlowModelPL

        num_sample = self.sample_config.num_sample
        num_steps = self.sample_config.num_sampling_step
        batch_size = self.sample_config.val_batch_size
        dt = 1.0 / num_steps

        stream_ctx = (
            torch.cuda.stream(self._stream)
            if self._stream is not None
            else _nullcontext()
        )

        samples_by_month: dict[int, torch.Tensor] = {}

        with stream_ctx:
            for month in self.months:
                all_samples = []
                remaining = num_sample

                # Derive month_length for padding masks (default non-leap year)
                weekday, days = calendar.monthrange(2013, month + 1)
                month_length = days - 28

                while remaining > 0:
                    bs = min(batch_size, remaining)
                    x_t = torch.randn(bs, seq_len, num_ch, device=device)
                    cond = self._condition_class(
                        month=month,
                        first_day_of_week=weekday,
                        month_length=month_length,
                    )
                    condition = cond.to_tensor_dict(batch_size=bs, device=device)

                    # Compute valid_length and zero padding in initial noise
                    valid_length = None
                    if create_mask and "month_length" in condition:
                        valid_length = FlowModelPL._convert_offset_month_length(
                            condition["month_length"], 28, steps_per_day
                        ).squeeze(1)
                        x_t = FlowModelPL._zero_padding(x_t, valid_length)

                    # For CFG wrapper, update labels per batch
                    if cfg_scale != 1.0:
                        velocity_model.dict_labels = condition
                        velocity_model.dict_emb_extras = {
                            "force_drop_ids": torch.zeros(
                                bs, dtype=torch.long, device=device
                            )
                        }

                    with torch.no_grad():
                        for i in range(num_steps):
                            t = torch.full((bs,), i * dt, device=device)
                            if cfg_scale != 1.0:
                                v = velocity_model(x_t, t, valid_length=valid_length)
                            else:
                                v = velocity_model(
                                    x_t, t, c=condition, valid_length=valid_length
                                )
                            x_t = x_t + v * dt

                    # Zero out padded positions in output
                    if valid_length is not None:
                        x_t = FlowModelPL._zero_padding(x_t, valid_length)

                    all_samples.append(x_t.cpu())
                    remaining -= bs

                samples_by_month[month] = torch.cat(all_samples, dim=0)

            # Wait for GPU work to finish before deleting model
            if self._stream is not None:
                self._stream.synchronize()

        del velocity_model

        # Save samples
        step_dir = os.path.join(self.output_dir, f"step_{step:07d}")
        os.makedirs(step_dir, exist_ok=True)
        torch.save(samples_by_month, os.path.join(step_dir, "samples.pt"))
        logger.info("Saved samples at step %d to %s", step, step_dir)

        # Plot — denormalize samples back to original units (e.g. kW)
        matplotlib.use("Agg")
        flat_samples = {}
        for month, tensor in samples_by_month.items():
            if self._inverse_transform is not None:
                # Model output is [b, seq_len, num_ch]; transforms expect [b, ch, seq_len]
                if tensor.dim() == 3:
                    tensor = rearrange(tensor, "b l c -> b c l")
                flat = self._inverse_transform(tensor).squeeze(1)
            else:
                if tensor.dim() == 3:
                    flat = rearrange(tensor, "b s c -> b (s c)")
                else:
                    flat = tensor
            flat_samples[f"Month {month + 1}"] = flat.numpy()

        fig_path = os.path.join(step_dir, "samples_overview.png")
        fig = plot_sampled_data_v2(flat_samples, save_filepath=fig_path)

        import matplotlib.pyplot as plt

        plt.close(fig)

        return (fig_path, step)

    def _log_figure(self, fig_path, sample_step, trainer):
        """Log a generated figure to wandb/mlflow. Must be called from the main thread."""
        if self.log_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    if not self._wandb_metric_defined:
                        wandb.define_metric(
                            "Samples/*", step_metric="Samples/train_step"
                        )
                        self._wandb_metric_defined = True
                    wandb.run.log(
                        {
                            "Samples/overview": wandb.Image(fig_path),
                            "Samples/train_step": sample_step,
                        }
                    )
                    logger.info(
                        "Logged sample figure to wandb (generated at step %d)",
                        sample_step,
                    )
            except Exception as e:
                logger.warning("Failed to log samples to wandb: %s", e)

        if self.log_mlflow:
            try:
                for lg in trainer.loggers or []:
                    if isinstance(lg, pl.loggers.MLFlowLogger):
                        lg.experiment.log_artifact(lg.run_id, fig_path)
                        break
            except Exception as e:
                logger.warning("Failed to log samples to mlflow: %s", e)

    def on_train_end(self, trainer, pl_module):
        """Wait for any pending background job and log its result before training ends."""
        if self._pending is not None:
            try:
                result = self._pending.result(timeout=120)
                if result is not None:
                    fig_path, sample_step = result
                    self._log_figure(fig_path, sample_step, trainer)
            except Exception as e:
                logger.error("Final sampling job failed: %s", e)
        self._executor.shutdown(wait=False)


class _nullcontext:
    """Minimal no-op context manager for Python <3.10 compatibility."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
