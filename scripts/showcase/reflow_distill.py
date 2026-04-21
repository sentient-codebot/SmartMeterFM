"""
Rectified-Flow / Reflow distillation: train a student on teacher-generated
(x_0, x_1) pairs along the deterministic straight-line path.

Architecture, embedders, and trainer plumbing mirror `train_flow.py`. Only the
train dataloader (ReflowPairDataset) and the initial-weight warm-start from
teacher checkpoint differ. Validation uses the regular LCL val loader, so
sampling-callback quality is measured against real profiles.

Usage:
    uv run python scripts/showcase/reflow_distill.py \
        --config configs/showcase/lcl_0421_reflow.toml \
        --time_id LCL-0421-REFLOW \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from train_flow import (  # type: ignore[import-not-found]
    SimpleDataModule,
    calculate_batch_sizes,
    create_flow_model,
    setup_data_module,
    setup_logging,
    setup_torch_optimizations,
    setup_trainer,
)

from smartmeterfm.data_modules.reflow_pairs import ReflowPairDataset
from smartmeterfm.models.flow import FlowModelPL
from smartmeterfm.utils.configuration import ExperimentConfig, save_config


def _warm_start_from_teacher(student: FlowModelPL, teacher_ckpt: str) -> None:
    """Copy teacher weights (online + EMA + buffers) into the student.

    `FlowModelPL.state_dict()` includes both `model.*` (online) and
    `ema.ema_model.*` + `ema.initted` + `ema.step` buffers — one load covers
    everything. Optimizer state is NOT copied; the student's AdamW starts
    fresh so moment estimates calibrate to the deterministic-pair loss.
    """
    logging.info(f"Warm-starting student from teacher: {teacher_ckpt}")
    teacher = FlowModelPL.load_from_checkpoint(
        teacher_ckpt,
        map_location=student.device,
        weights_only=False,
    )
    missing, unexpected = student.load_state_dict(teacher.state_dict(), strict=False)
    logging.info(
        f"load_state_dict: missing={len(missing)} keys, unexpected={len(unexpected)} keys"
    )
    if missing:
        logging.warning(f"Missing keys (first 10): {missing[:10]}")
    if unexpected:
        logging.warning(f"Unexpected keys (first 10): {unexpected[:10]}")
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Reflow distillation from a teacher flow-matching model"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--time_id", type=str, required=True)
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=torch.cuda.device_count(),
    )
    parser.add_argument("--maintain_batch_size", action="store_true")
    parser.add_argument("--local_batch_size", type=int, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    setup_torch_optimizations()
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # Load config — require [reflow] and reflow_mode
    config = ExperimentConfig.from_toml(args.config)
    if config.reflow is None:
        raise ValueError(f"{args.config} missing [reflow] block")
    if not config.train.reflow_mode:
        raise ValueError(f"{args.config} has reflow=... but train.reflow_mode=False")
    if not config.train.reflow_pairs_path:
        raise ValueError(
            f"{args.config} missing train.reflow_pairs_path — generate pairs first"
        )
    config.time_id = args.time_id

    run_id = f"reflow-{args.time_id}"
    setup_logging(run_id=run_id, level=logging.INFO)

    # GPU / accelerator
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    use_mps = False
    if num_gpus == 0 and torch.backends.mps.is_available():
        use_mps = True
        num_gpus = 1
    elif num_gpus == 0:
        logging.warning("No GPUs available, falling back to CPU")

    # Batch size bookkeeping (reuse train_flow logic)
    local_batch_size, accumulate_grad_batches, effective_batch_size = (
        calculate_batch_sizes(args, config, max(num_gpus, 1))
    )
    config.train.batch_size = local_batch_size
    logging.info(
        f"Reflow training: {num_gpus} GPU(s), local_bs={local_batch_size}, "
        f"accum={accumulate_grad_batches}, effective_bs={effective_batch_size}"
    )

    # Data:
    #   - val + sample_shape + data_collection from regular LCL data module
    #     (so val-sampling callback compares against real profiles)
    #   - train loader replaced with ReflowPairDataset-backed loader
    _real_train_loader, val_loader, sample_shape, data_collection = setup_data_module(
        config
    )

    pair_dataset = ReflowPairDataset(config.train.reflow_pairs_path)
    logging.info(
        f"Loaded {len(pair_dataset)} reflow pairs from "
        f"{config.train.reflow_pairs_path}; meta={pair_dataset.meta}"
    )
    # Sanity: pair tensor shape should match sample_shape [seq_len, channels]
    pair_shape = tuple(pair_dataset.x_0.shape[1:])
    if pair_shape != tuple(int(s) for s in sample_shape):
        raise ValueError(
            f"Reflow pair shape {pair_shape} != data module sample_shape "
            f"{tuple(sample_shape)} — regenerate pairs for this data config."
        )

    train_loader = DataLoader(
        pair_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    data_module = SimpleDataModule(train_loader, val_loader)

    # Student model (architecture + embedders come from config, same as teacher)
    student = create_flow_model(config, sample_shape)

    # Warm-start weights from teacher checkpoint
    _warm_start_from_teacher(student, config.reflow.teacher_checkpoint)

    # Trainer (reuses train_flow.setup_trainer; gradient_clip_val=1.0, bf16, etc.)
    ckpt_path = None
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt_path = args.resume_ckpt
        logging.info(f"Resuming reflow training from: {ckpt_path}")

    trainer = setup_trainer(
        args,
        config,
        num_gpus,
        accumulate_grad_batches,
        use_mps=use_mps,
        profile_inverse_transform=data_collection.profile_inverse_transform,
    )

    if trainer.is_global_zero:
        os.makedirs("results/configs", exist_ok=True)
        save_config(config, args.time_id)
        logging.info(f"Saved config to results/configs/exp_config_{args.time_id}.yaml")
        logging.info(f"Reflow experiment starts: {run_id}")

    logging.info("Reflow training initiated.")
    trainer.fit(student, data_module, ckpt_path=ckpt_path)
    logging.info("Reflow training complete.")


if __name__ == "__main__":
    main()
