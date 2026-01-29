"""
Train VAE baseline model for conditional generation on WPuQ Heat Pump data.

This showcase script demonstrates how to train a Conditional VAE baseline
for energy profile generation using publicly available WPuQ data.

Usage:
    uv run python scripts/showcase/train_vae_baseline.py --config configs/showcase/wpuq_vae.toml --time_id vae_showcase_001

Example:
    # Basic training
    uv run python scripts/showcase/train_vae_baseline.py --config configs/showcase/wpuq_vae.toml --time_id test_run

    # With custom hyperparameters
    uv run python scripts/showcase/train_vae_baseline.py --config configs/showcase/wpuq_vae.toml --time_id test_run --latent_dim 64 --beta 0.01
"""

import argparse
import logging
import math
import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset

from smartmeterfm.data_modules.heat_pump import WPuQ
from smartmeterfm.models.baselines.cond_gen.vae import VAEModelPL
from smartmeterfm.utils.configuration import (
    ExperimentConfig,
    IntegerEmbedderArgs,
    save_config,
)
from smartmeterfm.utils.eval import (
    MkMMD,
    MultiMetric,
    calculate_frechet,
    kl_divergence,
    ks_test_d,
    source_mean,
    source_std,
    target_mean,
    target_std,
    ws_distance,
)


def setup_logging(run_id: str, level: int = logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s - {run_id} - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_torch_optimizations():
    """Configure PyTorch optimizations for better performance."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed.*")


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_embedder_args(dim_base: int, num_months: int = 12) -> dict:
    """Get embedder arguments for month conditioning."""
    return {
        "month": IntegerEmbedderArgs(
            num_embedding=num_months,
            dim_embedding=dim_base,
            dropout=0.1,
        ).to_dict(),
    }


def calculate_batch_sizes(args, config, num_gpus):
    """Calculate batch sizes and gradient accumulation for multi-GPU training."""
    global_batch_size = config.train.batch_size

    if args.local_batch_size is not None:
        local_batch_size = args.local_batch_size
        effective_batch_size = local_batch_size * num_gpus

        if args.maintain_batch_size:
            accumulate_grad_batches = math.ceil(global_batch_size / effective_batch_size)
            effective_batch_size = local_batch_size * num_gpus * accumulate_grad_batches
        else:
            accumulate_grad_batches = config.train.gradient_accumulate_every
    else:
        if args.maintain_batch_size:
            local_batch_size = max(1, global_batch_size // num_gpus)
            accumulate_grad_batches = config.train.gradient_accumulate_every
            effective_batch_size = local_batch_size * num_gpus * accumulate_grad_batches
        else:
            local_batch_size = global_batch_size
            accumulate_grad_batches = config.train.gradient_accumulate_every
            effective_batch_size = local_batch_size * num_gpus * accumulate_grad_batches

    return local_batch_size, accumulate_grad_batches, effective_batch_size


def setup_data_module(config: ExperimentConfig):
    """Set up the WPuQ data module."""
    logging.info("Setting up WPuQ Heat Pump data module...")

    data_config = config.data
    wpuq_data = WPuQ(data_config)

    train_profiles = wpuq_data.dataset.profile["train"]
    val_profiles = wpuq_data.dataset.profile["val"]

    train_labels = wpuq_data.dataset.label["train"]
    val_labels = wpuq_data.dataset.label["val"]

    train_dataset = TensorDataset(
        train_profiles,
        train_labels["month"],
    )
    val_dataset = TensorDataset(
        val_profiles,
        val_labels["month"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    sample_shape = train_profiles.shape[1:]
    logging.info(f"Data module setup complete. Sample shape: {sample_shape}")
    logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    return train_loader, val_loader, sample_shape


def setup_metrics():
    """Set up evaluation metrics for the model."""
    mkmmd = MkMMD(kernel_type="rbf", num_kernel=1, kernel_mul=2.0, coefficient="auto")
    dict_eval_fn = {
        "MkMMD": mkmmd,
        "DirectFD": calculate_frechet,
        "source_mean": source_mean,
        "source_std": source_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "kl_divergence": kl_divergence,
        "ws_distance": ws_distance,
        "ks_test_d": ks_test_d,
    }
    return lambda: MultiMetric(dict_eval_fn)


def create_vae_model(config: ExperimentConfig, sample_shape: tuple, latent_dim: int, beta: float):
    """Create and configure the VAE Model.

    Args:
        config: Experiment configuration.
        sample_shape: Shape of input samples (seq_len, channels).
        latent_dim: Dimension of the latent space.
        beta: Beta parameter for beta-VAE.

    Returns:
        VAEModelPL: Configured VAE model.
    """
    logging.info("Creating VAE model...")

    # Update model config
    config.model.seq_length = sample_shape[0]
    config.model.num_in_channel = sample_shape[1]

    # Get embedder arguments
    emb_args = get_embedder_args(latent_dim)

    # Set up metrics
    make_metrics = setup_metrics()

    # Calculate output length (flattened profile)
    output_length = sample_shape[0] * sample_shape[1]

    # Create model
    pl_vae = VAEModelPL(
        num_in_channel=1,  # VAE uses flattened input
        train_config=config.train,
        latent_dim=latent_dim,
        output_length=output_length,
        beta=beta,
        label_embedder_name="wpuq_month",
        label_embedder_args=emb_args,
        metrics_factory=make_metrics,
        create_mask=False,
    )

    num_params = count_parameters(pl_vae.vae)
    logging.info(f"VAE model created with {num_params:,} parameters")
    logging.info(f"Latent dimension: {latent_dim}, Beta: {beta}")

    return pl_vae


def setup_trainer(args, config, num_gpus, accumulate_grad_batches):
    """Set up the PyTorch Lightning trainer."""
    ddp_strategy = "auto"
    if num_gpus > 1:
        ddp_strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
        )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/{config.time_id}",
        monitor="Validation/total_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="vae-{step:06d}-{Validation/total_loss:.4f}",
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, lr_monitor]

    loggers = []

    if config.log_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
            wandb_logger = WandbLogger(
                project="smartmeterfm-showcase",
                name=config.time_id,
                save_dir="logs/",
            )
            loggers.append(wandb_logger)
        except ImportError:
            logging.warning("wandb not installed, skipping wandb logging")

    trainer = pl.Trainer(
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        max_steps=config.train.num_train_step,
        logger=loggers if loggers else None,
        val_check_interval=config.train.val_every,
        check_val_every_n_epoch=None,
        accelerator="gpu" if num_gpus > 0 else "cpu",
        devices=num_gpus if num_gpus > 0 else "auto",
        strategy=ddp_strategy,
        precision="bf16-mixed",
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    return trainer


class WPuQDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule wrapper for WPuQ data."""

    def __init__(self, train_loader, val_loader):
        super().__init__()
        self._train_loader = train_loader
        self._val_loader = val_loader

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader


def main():
    parser = argparse.ArgumentParser(
        description="Train VAE baseline for conditional generation on WPuQ data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/showcase/wpuq_vae.toml",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--time_id",
        type=str,
        required=True,
        help="Unique identifier for this training run",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs to use for training (default: all available)",
    )
    parser.add_argument(
        "--maintain_batch_size",
        action="store_true",
        help="Maintain the same effective batch size as single-GPU training",
    )
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=None,
        help="Batch size per GPU (overrides batch size in config)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Latent dimension for VAE (default: 128)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="Beta parameter for beta-VAE (default: 0.01). Note: beta >= 0.1 may cause posterior collapse",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    # Set up PyTorch optimizations
    setup_torch_optimizations()

    # Load configuration
    config = ExperimentConfig.from_toml(args.config)
    config.time_id = args.time_id

    # Setup logging
    run_id = f"train-vae-{args.time_id}"
    setup_logging(run_id=run_id, level=logging.INFO)

    # Calculate available GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus == 0:
        logging.warning("No GPUs available, falling back to CPU")

    # Calculate batch sizes
    local_batch_size, accumulate_grad_batches, effective_batch_size = calculate_batch_sizes(
        args, config, max(num_gpus, 1)
    )
    config.train.batch_size = local_batch_size

    logging.info(f"Training VAE model with {num_gpus} GPUs")
    logging.info(f"Local batch size per GPU: {local_batch_size}")
    logging.info(f"Gradient accumulation steps: {accumulate_grad_batches}")
    logging.info(f"Effective batch size: {effective_batch_size}")

    # Set up data module
    train_loader, val_loader, sample_shape = setup_data_module(config)
    data_module = WPuQDataModule(train_loader, val_loader)

    # Create model
    pl_vae = create_vae_model(config, sample_shape, args.latent_dim, args.beta)

    # Handle checkpoint resuming
    ckpt_path = None
    if args.resume_ckpt is not None:
        if os.path.exists(args.resume_ckpt):
            ckpt_path = args.resume_ckpt
            logging.info(f"Will resume from checkpoint: {ckpt_path}")
        else:
            logging.warning(f"Checkpoint not found: {args.resume_ckpt}")

    # Set up trainer
    trainer = setup_trainer(args, config, num_gpus, accumulate_grad_batches)

    # Save configuration
    if trainer.is_global_zero:
        os.makedirs("results/configs", exist_ok=True)
        save_config(config, args.time_id)
        logging.info(f"Saved configuration to results/configs/exp_config_{args.time_id}.yaml")
        logging.info(f"VAE baseline experiment starts: {run_id}")

    # Start training
    logging.info("VAE training initiated.")
    trainer.fit(pl_vae, data_module, ckpt_path=ckpt_path)
    logging.info("VAE training complete.")


if __name__ == "__main__":
    main()
