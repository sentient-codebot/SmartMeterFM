"""Pair dataset for Rectified-Flow / Reflow distillation.

Yields deterministic ``(x_0, x_1, condition)`` triples cached to disk by
``scripts/showcase/generate_reflow_pairs.py``. x_0 is already zero-padded to
match the teacher's ``_prepare_x_0`` behavior, so training-time consumers can
use it directly without re-padding.

Cache layout (torch.save dict):
    {
        "x_0":  Tensor[N, num_patches, patch_size],   # pre-zero-padded noise
        "x_1":  Tensor[N, num_patches, patch_size],   # teacher output
        "cond": dict[str, Tensor[N, 1]],              # month, year, first_day_of_week, month_length
        "meta": {...},                                 # teacher ckpt, NFE, cfg, seed, ...
    }
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class ReflowPairDataset(Dataset):
    """Index-mapped dataset over a cached reflow pair file.

    The whole cache is loaded into RAM at init — at 100k pairs × [93, 16] fp32
    this is ~1.2 GB, fine on all compute nodes we use. Switch to mmap loading
    if cache ever exceeds a few GB.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.x_0: torch.Tensor = blob["x_0"]
        self.x_1: torch.Tensor = blob["x_1"]
        self.cond: dict[str, torch.Tensor] = blob["cond"]
        self.meta: dict = blob.get("meta", {})

        n = self.x_0.shape[0]
        assert self.x_1.shape[0] == n, (
            f"x_0/x_1 length mismatch: {self.x_0.shape[0]} vs {self.x_1.shape[0]}"
        )
        for k, v in self.cond.items():
            assert v.shape[0] == n, f"cond[{k}] length {v.shape[0]} != {n}"

    def __len__(self) -> int:
        return self.x_0.shape[0]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return (
            self.x_0[idx],
            self.x_1[idx],
            {k: v[idx] for k, v in self.cond.items()},
        )
