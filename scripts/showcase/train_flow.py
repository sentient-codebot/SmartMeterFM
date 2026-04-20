"""
Train Flow Matching model for conditional generation on WPuQ Heat Pump data.

This showcase script demonstrates how to train a Flow Matching model for
conditional energy profile generation using publicly available WPuQ data.

The WPuQ dataset contains heat pump electricity consumption data from Germany,
sampled at various resolutions (10s, 1min, 15min, 30min, 1h).

Usage:
    uv run python scripts/showcase/train_flow.py --config configs/showcase/wpuq_flow_small.toml --time_id flow_showcase_001

Example:
    # Basic training
    uv run python scripts/showcase/train_flow.py --config configs/showcase/wpuq_flow_small.toml --time_id test_run

    # Multi-GPU training
    uv run python scripts/showcase/train_flow.py --config configs/showcase/wpuq_flow_small.toml --time_id test_run --num_gpus 2

    # Resume from checkpoint
    uv run python scripts/showcase/train_flow.py --config configs/showcase/wpuq_flow_small.toml --time_id test_run --resume_ckpt path/to/checkpoint.ckpt
"""

import argparse
import calendar
import logging
import math
import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset

from smartmeterfm.conditions import LCLCondition
from smartmeterfm.data_modules.base import RESOLUTION_SECONDS
from smartmeterfm.data_modules.heat_pump import WPuQ
from smartmeterfm.data_modules.wpuq_household import WPuQHousehold
from smartmeterfm.models.flow import FlowModelPL
from smartmeterfm.utils.callbacks import PeriodicSamplingCallback
from smartmeterfm.utils.configuration import (
    ExperimentConfig,
    IntegerEmbedderArgs,
    PositionEmbedderArgs,
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


def make_collate_fn(label_keys: list[str]):
    """Create a collate function that packs labels into a condition dict.

    Derives ``first_day_of_week`` and ``month_length`` from ``year`` and
    ``month`` labels when both are present.
    """

    def collate_fn(batch):
        profiles = torch.stack([b[0] for b in batch])
        condition: dict[str, torch.Tensor] = {}
        for i, key in enumerate(label_keys, start=1):
            condition[key] = torch.stack([b[i] for b in batch])

        # Derive first_day_of_week and month_length from (year, month)
        if "year" in condition and "month" in condition:
            years = condition["year"].squeeze(1)  # [batch]
            months = condition["month"].squeeze(1)  # [batch], 0-indexed
            fdow = torch.zeros_like(years)
            ml = torch.zeros_like(years)
            for i in range(len(years)):
                y = years[i].item()
                m = months[i].item() + 1  # back to 1-indexed
                weekday, days = calendar.monthrange(y, m)
                fdow[i] = weekday
                ml[i] = days - 28
            condition["first_day_of_week"] = fdow.unsqueeze(1)
            condition["month_length"] = ml.unsqueeze(1)

        return profiles, condition

    return collate_fn


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
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="torch.distributed.*"
    )


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_embedder_args(dim_base: int, num_months: int = 12) -> dict:
    """Get embedder arguments for month conditioning.

    Args:
        dim_base: Base dimension for embeddings.
        num_months: Number of month categories (default: 12).

    Returns:
        Dictionary of embedder arguments.
    """
    return {
        "month": IntegerEmbedderArgs(
            num_embedding=num_months,
            dim_embedding=dim_base,
            dropout=0.1,
        ).to_dict(),
    }


def get_lcl_embedder_args(
    dim_base: int,
    num_months: int = 12,
    year_offset: int = 2012,
) -> dict:
    """Get embedder arguments for the ``lcl_label`` / ``lcl_context`` embedder.

    Four fields are embedded: ``month`` (integer), ``year`` (sinusoidal
    position), ``first_day_of_week`` (integer, 0-6) and ``month_length``
    (integer, 0-3 covering 28-31 day months).  Calendar fields are derived
    from ``(year, month)`` by the collate function.

    Args:
        dim_base: Base dimension for embeddings.
        num_months: Number of month categories (default: 12).
        year_offset: Subtracted from raw year before the position embedding
            lookup (default: 2012 → 2012-2013 map to 0-1).

    Returns:
        Dictionary of embedder arguments.
    """
    return {
        "month": IntegerEmbedderArgs(
            num_embedding=num_months,
            dim_embedding=dim_base,
            dropout=0.1,
        ).to_dict(),
        "year": PositionEmbedderArgs(
            dim_embedding=dim_base,
        ).to_dict(),
        "first_day_of_week": IntegerEmbedderArgs(
            num_embedding=7,
            dim_embedding=dim_base,
            dropout=0.1,
        ).to_dict(),
        "month_length": IntegerEmbedderArgs(
            num_embedding=4,
            dim_embedding=dim_base,
            dropout=0.1,
        ).to_dict(),
        "year_offset": year_offset,
    }


def calculate_batch_sizes(args, config, num_gpus):
    """Calculate batch sizes and gradient accumulation for multi-GPU training."""
    global_batch_size = config.train.batch_size

    if args.local_batch_size is not None:
        local_batch_size = args.local_batch_size
        effective_batch_size = local_batch_size * num_gpus

        if args.maintain_batch_size:
            accumulate_grad_batches = math.ceil(
                global_batch_size / effective_batch_size
            )
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
    """Set up the data module.

    Args:
        config: Experiment configuration.

    Returns:
        Tuple of (train_dataloader, val_dataloader, sample_shape, data_collection)
    """
    data_config = config.data
    dataset_name = getattr(data_config, "dataset", "wpuq")

    if dataset_name == "wpuq_household":
        logging.info("Setting up WPuQ Household data module...")
        data_collection = WPuQHousehold(data_config)
    elif dataset_name == "lcl_electricity":
        logging.info("Setting up LCL Electricity data module...")
        from smartmeterfm.data_modules.lcl_electricity import LCLElectricity

        data_collection = LCLElectricity(data_config)
    else:
        logging.info("Setting up WPuQ Heat Pump data module...")
        data_collection = WPuQ(data_config)

    # Get train and validation data
    train_profiles = data_collection.dataset.profile["train"]  # [N, seq_len, channels]
    val_profiles = data_collection.dataset.profile["val"]

    train_labels = data_collection.dataset.label["train"]
    val_labels = data_collection.dataset.label["val"]

    # Extract all available label keys (preserving order)
    label_keys = train_labels.label_names

    # Create PyTorch datasets with all labels
    train_dataset = TensorDataset(
        train_profiles, *[train_labels[k] for k in label_keys]
    )
    val_dataset = TensorDataset(val_profiles, *[val_labels[k] for k in label_keys])

    collate_fn = make_collate_fn(label_keys)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    sample_shape = train_profiles.shape[1:]  # [seq_len, channels]
    logging.info(f"Data module setup complete. Sample shape: {sample_shape}")
    logging.info(
        f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
    )

    return train_loader, val_loader, sample_shape, data_collection


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


def create_flow_model(config: ExperimentConfig, sample_shape: tuple):
    """Create and configure the Flow Model.

    Args:
        config: Experiment configuration.
        sample_shape: Shape of input samples (seq_len, channels).

    Returns:
        FlowModelPL: Configured flow model.
    """
    logging.info("Creating Flow model...")

    # Update model config with actual dimensions
    config.model.seq_length = sample_shape[0]
    config.model.num_in_channel = sample_shape[1]

    # Get embedder arguments — LCL uses month+year, others use month only
    dataset_name = getattr(config.data, "dataset", "wpuq")
    if dataset_name == "lcl_electricity":
        emb_args = get_lcl_embedder_args(config.model.dim_base)
        label_embedder_name = "lcl_label"
    else:
        emb_args = get_embedder_args(config.model.dim_base)
        label_embedder_name = "wpuq_month"

    # Set up metrics
    make_metrics = setup_metrics()

    # Compute steps_per_day from resolution
    resolution = config.data.resolution
    steps_per_day = 86400 // RESOLUTION_SECONDS[resolution]

    # Enable masking for monthly-segmented data
    is_monthly = getattr(config.data, "segment_type", "daily") == "monthly"

    # Create model
    pl_flow = FlowModelPL(
        flow_config=config.flow,
        model_config=config.model,
        train_config=config.train,
        sample_config=config.sample,
        num_in_channel=sample_shape[1],
        label_embedder_name=label_embedder_name,
        label_embedder_args=emb_args,
        context_embedder_name=None,  # No context for basic conditional generation
        context_embedder_args=None,
        metrics_factory=make_metrics,
        create_mask=is_monthly,
        steps_per_day=steps_per_day,
    )

    config.model.num_parameter = count_parameters(pl_flow.model)
    logging.info(f"Flow model created with {config.model.num_parameter:,} parameters")

    return pl_flow


def setup_trainer(
    args,
    config,
    num_gpus,
    accumulate_grad_batches,
    use_mps=False,
    profile_inverse_transform=None,
):
    """Set up the PyTorch Lightning trainer."""
    # Set up DDP strategy
    ddp_strategy = "auto"
    if num_gpus > 1 and not use_mps:
        ddp_strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
        )

    # Set up callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/{config.time_id}",
        monitor="Validation/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="flow-{step:06d}-{Validation/loss:.4f}",
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # Use LCLCondition for LCL datasets so the sampling callback
    # produces the correct condition keys (month + year, not season)
    dataset_name = getattr(config.data, "dataset", "wpuq")
    condition_cls = LCLCondition if dataset_name == "lcl_electricity" else None

    sampling_callback = PeriodicSamplingCallback(
        sample_every=config.train.save_and_sample_every,
        sample_config=config.train.val_sample_config,
        months=config.train.sample_months,
        output_dir=f"results/samples/{config.time_id}",
        log_wandb=config.log_wandb,
        log_mlflow=config.log_mlflow,
        profile_inverse_transform=profile_inverse_transform,
        **({"condition_class": condition_cls} if condition_cls else {}),
    )

    callbacks = [checkpoint_callback, lr_monitor, sampling_callback]

    # Set up loggers
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

    # Determine accelerator and precision
    if use_mps:
        accelerator = "mps"
        devices = 1
        precision = "32-true"  # MPS doesn't support bf16-mixed
    elif num_gpus > 0:
        accelerator = "gpu"
        devices = num_gpus
        precision = "bf16-mixed"
    else:
        accelerator = "cpu"
        devices = "auto"
        precision = "bf16-mixed"

    # Create trainer
    trainer = pl.Trainer(
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        max_steps=config.train.num_train_step,
        logger=loggers if loggers else None,
        val_check_interval=config.train.val_every,
        check_val_every_n_epoch=None,
        accelerator=accelerator,
        devices=devices,
        strategy=ddp_strategy,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    return trainer


class SimpleDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule wrapper for pre-built dataloaders."""

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
        description="Train Flow Model for conditional generation on WPuQ data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/showcase/wpuq_flow_small.toml",
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
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (sets pl.seed_everything)",
    )
    args = parser.parse_args()

    # Set up PyTorch optimizations
    setup_torch_optimizations()

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # Load configuration
    config = ExperimentConfig.from_toml(args.config)
    config.time_id = args.time_id

    # Setup logging
    run_id = f"train-flow-{args.time_id}"
    setup_logging(run_id=run_id, level=logging.INFO)

    # Calculate available GPUs / accelerator
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    use_mps = False
    if num_gpus == 0 and torch.backends.mps.is_available():
        use_mps = True
        num_gpus = 1  # treat MPS as a single device for batch size calculation
        logging.info("Using MPS (Apple Silicon) accelerator")
    elif num_gpus == 0:
        logging.warning("No GPUs available, falling back to CPU")

    # Calculate batch sizes
    local_batch_size, accumulate_grad_batches, effective_batch_size = (
        calculate_batch_sizes(args, config, max(num_gpus, 1))
    )
    config.train.batch_size = local_batch_size

    logging.info(f"Training Flow model with {num_gpus} GPUs")
    logging.info(f"Local batch size per GPU: {local_batch_size}")
    logging.info(f"Gradient accumulation steps: {accumulate_grad_batches}")
    logging.info(f"Effective batch size: {effective_batch_size}")

    # Set up data module
    train_loader, val_loader, sample_shape, data_collection = setup_data_module(config)
    data_module = SimpleDataModule(train_loader, val_loader)

    # Create model
    pl_flow = create_flow_model(config, sample_shape)

    # Handle checkpoint resuming
    ckpt_path = None
    if args.resume_ckpt is not None:
        if os.path.exists(args.resume_ckpt):
            ckpt_path = args.resume_ckpt
            logging.info(f"Will resume from checkpoint: {ckpt_path}")
        else:
            logging.warning(f"Checkpoint not found: {args.resume_ckpt}")

    # Set up trainer
    trainer = setup_trainer(
        args,
        config,
        num_gpus,
        accumulate_grad_batches,
        use_mps=use_mps,
        profile_inverse_transform=data_collection.profile_inverse_transform,
    )

    # Save configuration
    if trainer.is_global_zero:
        os.makedirs("results/configs", exist_ok=True)
        save_config(config, args.time_id)
        logging.info(
            f"Saved configuration to results/configs/exp_config_{args.time_id}.yaml"
        )
        logging.info(f"Flow baseline experiment starts: {run_id}")

    # Start training
    logging.info("Flow training initiated.")
    trainer.fit(pl_flow, data_module, ckpt_path=ckpt_path)
    logging.info("Flow training complete.")


if __name__ == "__main__":
    main()
