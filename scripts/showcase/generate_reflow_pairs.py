"""
Generate (x_0, x_1_teacher, condition) pairs for Rectified-Flow / Reflow distillation.

Iterates the LCL training dataloader to harvest condition tuples (discarding
real profiles), samples fresh Gaussian noise x_0 pre-zero-padded to match what
the teacher sees internally via ``_prepare_x_0``, runs the teacher's ODE at
high NFE, and caches tuples to disk for ``scripts/showcase/reflow_distill.py``.

Usage:
    uv run python scripts/showcase/generate_reflow_pairs.py \
        --config configs/showcase/lcl_0421_reflow.toml \
        --out_path data/reflow_pairs/lcl_0421_01_200nfe.pt
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Reuse the data/model/embedder setup from train_flow.py so the dataloader
# and model shapes match exactly. scripts/showcase/ is on sys.path when this
# script is invoked directly (Python adds the script dir to sys.path[0]).
from train_flow import (  # type: ignore[import-not-found]
    setup_data_module,
    setup_logging,
    setup_torch_optimizations,
)

from smartmeterfm.interfaces.base.config import SampleConfig as InternalSampleConfig
from smartmeterfm.interfaces.base.config import SolverMethod
from smartmeterfm.interfaces.smartmeter import SmartMeterFMModel
from smartmeterfm.models.flow import FlowModelPL
from smartmeterfm.utils.configuration import ExperimentConfig


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher (x_0, x_1) pairs for reflow distillation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default=None,
        help="Overrides config.reflow.teacher_checkpoint if given",
    )
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--num_pairs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_step", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile on the teacher EMA model",
    )
    args = parser.parse_args()

    setup_torch_optimizations()
    setup_logging(run_id="gen-reflow-pairs", level=logging.INFO)

    # Config
    config = ExperimentConfig.from_toml(args.config)
    if config.reflow is None:
        raise ValueError(
            f"{args.config} has no [reflow] block — required for pair generation"
        )
    reflow_cfg = config.reflow

    teacher_ckpt = args.teacher_checkpoint or reflow_cfg.teacher_checkpoint
    num_pairs = args.num_pairs or reflow_cfg.num_pairs
    batch_size = args.batch_size or reflow_cfg.pair_batch_size
    num_step = args.num_step or reflow_cfg.teacher_num_step
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else reflow_cfg.cfg_scale
    out_path = args.out_path or config.train.reflow_pairs_path
    if out_path is None:
        raise ValueError(
            "out_path not set — pass --out_path or set train.reflow_pairs_path in the TOML"
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Seeds (torch + numpy; DataLoader generator seeded below)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataloader — reuse LCL train loader; override batch size so each yielded
    # batch is exactly pair_batch_size (keeps torch.compile on a static graph).
    config.train.batch_size = batch_size
    train_loader, _val_loader, sample_shape, _data_collection = setup_data_module(
        config
    )
    num_patches, patch_size = int(sample_shape[0]), int(sample_shape[1])
    logging.info(
        f"Pair-gen setup: num_pairs={num_pairs}, batch_size={batch_size}, "
        f"num_step={num_step}, cfg_scale={cfg_scale}, shape=({num_patches}, {patch_size})"
    )

    # Teacher model
    logging.info(f"Loading teacher checkpoint: {teacher_ckpt}")
    model = SmartMeterFMModel.from_checkpoint(teacher_ckpt, device=args.device)
    use_bf16 = model.device.type == "cuda"
    if use_bf16 and not args.no_compile:
        model.pl_model.ema.ema_model = torch.compile(model.pl_model.ema.ema_model)

    sample_cfg = InternalSampleConfig(
        use_ema=reflow_cfg.use_ema_teacher,
        batch_size=batch_size,
        num_step=num_step,
        method=SolverMethod.EULER,
        data_shape=(num_patches, patch_size),
    )

    # Storage dtype
    store_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[
        reflow_cfg.pairs_dtype
    ]

    # Buffers
    x0_buf: list[torch.Tensor] = []
    x1_buf: list[torch.Tensor] = []
    cond_buf: dict[str, list[torch.Tensor]] = {}
    collected = 0

    pbar = tqdm(total=num_pairs, desc="Generating pairs")
    # Loop over dataloader epochs until we have enough pairs.
    while collected < num_pairs:
        for profile, condition in train_loader:
            if collected >= num_pairs:
                break
            # drop_last=True on the train loader (see train_flow.py) guarantees
            # full batch_size; still guard in case that changes upstream.
            if profile.shape[0] != batch_size:
                continue

            # Draw fresh x_0, zero-pad to match teacher's _prepare_x_0
            x0 = torch.randn(batch_size, num_patches, patch_size, device=model.device)
            if model.create_mask and "month_length" in condition:
                valid_length = FlowModelPL._convert_offset_month_length(
                    condition["month_length"].to(model.device),
                    28,
                    model.steps_per_day,
                ).squeeze(1)
                x0 = FlowModelPL._zero_padding(x0, valid_length)

            # Teacher forward: integrate ODE at high NFE under bf16 autocast
            with torch.autocast(
                device_type=model.device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                x1 = model.sample(
                    sample_config=sample_cfg,
                    condition=condition,
                    cfg_scale=cfg_scale,
                    x_0=x0,
                )
            # x1 returned from model.sample is already zero-padded internally.

            x0_buf.append(x0.detach().to(dtype=store_dtype, device="cpu"))
            x1_buf.append(x1.detach().to(dtype=store_dtype, device="cpu"))
            for k, v in condition.items():
                cond_buf.setdefault(k, []).append(v.detach().cpu().clone())

            collected += batch_size
            pbar.update(batch_size)
    pbar.close()

    # Truncate to exactly num_pairs
    x0_all = torch.cat(x0_buf, dim=0)[:num_pairs].contiguous()
    x1_all = torch.cat(x1_buf, dim=0)[:num_pairs].contiguous()
    cond_all = {
        k: torch.cat(vs, dim=0)[:num_pairs].contiguous() for k, vs in cond_buf.items()
    }

    # Sanity stats
    logging.info("---- Sanity checks ----")
    logging.info(f"x_0 shape {tuple(x0_all.shape)}, dtype {x0_all.dtype}")
    logging.info(f"x_1 shape {tuple(x1_all.shape)}, dtype {x1_all.dtype}")
    for k, v in cond_all.items():
        vv = v.squeeze(-1) if v.ndim == 2 and v.shape[-1] == 1 else v
        uniq, counts = torch.unique(vv, return_counts=True)
        logging.info(f"cond[{k}] uniq={uniq.tolist()} counts={counts.tolist()}")

    # Look for a 30-day (Feb/Apr/Jun/Sep/Nov) sample to verify padding zeros.
    # month_length encoded as days - 28 → 0 = Feb, 2 = Apr/Jun/Sep/Nov, 3 = Jan/Mar/...
    if "month_length" in cond_all:
        ml_flat = cond_all["month_length"].squeeze(-1)
        idx_28 = torch.nonzero(ml_flat == 0)
        idx_30 = torch.nonzero(ml_flat == 2)
        for label, idx in (("28-day", idx_28), ("30-day", idx_30)):
            if idx.numel() > 0:
                i = idx[0].item()
                # valid_length in full timesteps = (ml + 28) * steps_per_day
                vlen = int((ml_flat[i].item() + 28) * model.steps_per_day)
                flat = patch_size * num_patches
                # positions [vlen:flat] must be zero across x0 and x1
                x0_flat = x0_all[i].reshape(flat)
                x1_flat = x1_all[i].reshape(flat)
                tail_x0 = x0_flat[vlen:].float().abs().sum().item()
                tail_x1 = x1_flat[vlen:].float().abs().sum().item()
                logging.info(
                    f"{label} sample idx={i}: valid_length={vlen}, "
                    f"|x0[vlen:]|={tail_x0:.3e}, |x1[vlen:]|={tail_x1:.3e} "
                    f"(both should be 0)"
                )

    # Persist cache
    meta = {
        "teacher_ckpt": os.path.abspath(teacher_ckpt),
        "teacher_ckpt_sha1": _file_sha1(teacher_ckpt),
        "teacher_num_step": num_step,
        "cfg_scale": cfg_scale,
        "use_ema_teacher": reflow_cfg.use_ema_teacher,
        "seed": args.seed,
        "pairs_dtype": reflow_cfg.pairs_dtype,
        "num_pairs": num_pairs,
        "data_shape": (num_patches, patch_size),
        "config_path": os.path.abspath(args.config),
    }
    blob = {"x_0": x0_all, "x_1": x1_all, "cond": cond_all, "meta": meta}
    torch.save(blob, out_path)
    size_mb = os.path.getsize(out_path) / (1024**2)
    logging.info(f"Saved {num_pairs} pairs to {out_path} ({size_mb:.1f} MB)")
    logging.info(f"Metadata: {meta}")


if __name__ == "__main__":
    main()
