"""Composable transforms for time series profiles.

Following the Transform / Compose pattern from Large-Customer-Data-Utils,
extended with inverse() support for generation tasks.
"""

from typing import Annotated

import torch
from einops import rearrange
from torch import Tensor


class Transform:
    """Base class for all transforms."""

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def inverse(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class Compose:
    """Chain transforms. Supports forward and inverse."""

    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def inverse(self, x: Tensor) -> Tensor:
        for t in reversed(self.transforms):
            x = t.inverse(x)
        return x

    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + "("]
        for t in self.transforms:
            lines.append(f"    {t}")
        lines.append(")")
        return "\n".join(lines)


class MinMaxScaler(Transform):
    """Scale to [-1, 1] using global min/max."""

    def __init__(
        self,
        min_val: Tensor | float | None = None,
        max_val: Tensor | float | None = None,
    ):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, data: Tensor) -> "MinMaxScaler":
        self.max_val = 1.1 * torch.max(data)
        self.min_val = min(
            torch.tensor(0.0, dtype=data.dtype, device=data.device), torch.min(data)
        )
        return self

    def __call__(self, data: Tensor) -> Tensor:
        return (data - self.min_val) / (self.max_val - self.min_val) * 2.0 - 1.0

    def inverse(self, data: Tensor) -> Tensor:
        _to_tensor = lambda x: (
            torch.tensor(x, device=data.device)
            if not isinstance(x, torch.Tensor)
            else x
        )
        max_val, min_val = map(_to_tensor, (self.max_val, self.min_val))
        if max_val.ndim == 0:
            max_val = max_val.unsqueeze(0)
        if min_val.ndim == 0:
            min_val = min_val.unsqueeze(0)
        assert max_val.shape == min_val.shape
        if max_val.ndim == 1:
            max_val = rearrange(max_val, "l -> 1 1 l")
            min_val = rearrange(min_val, "l -> 1 1 l")
        elif max_val.ndim == 2:
            max_val = rearrange(max_val, "c l -> 1 c l")
            min_val = rearrange(min_val, "c l -> 1 c l")

        if data.ndim == 2:
            dim_data_in = 2
            data = rearrange(data, "s l -> s 1 l")
        else:
            dim_data_in = 3

        max_val, min_val = (x.to(data.device) for x in (max_val, min_val))

        mid_channel = data.shape[1] // 2
        data = data[:, mid_channel : mid_channel + 1, :]
        if dim_data_in == 2:
            return rearrange(data, "s 1 l -> s l")
        else:
            return data

    def __repr__(self) -> str:
        return f"MinMaxScaler(min={self.min_val}, max={self.max_val})"


class MeanStdScaler(Transform):
    """Standardize using mean/std."""

    def __init__(
        self,
        mean_val: Tensor | None = None,
        std_val: Tensor | None = None,
    ):
        self.mean_val = mean_val
        self.std_val = std_val

    def fit(self, data: Tensor) -> "MeanStdScaler":
        self.mean_val = torch.mean(data, dim=0, keepdim=True)
        self.std_val = torch.std(data, dim=0, keepdim=True)
        return self

    def __call__(self, data: Tensor) -> Tensor:
        return (data - self.mean_val) / self.std_val

    def inverse(self, data: Tensor) -> Tensor:
        _to_tensor = lambda x: (
            torch.tensor(x, device=data.device)
            if not isinstance(x, torch.Tensor)
            else x
        )
        mean_val, std_val = (_to_tensor(v) for v in (self.mean_val, self.std_val))
        if mean_val.ndim == 0:
            mean_val = mean_val.unsqueeze(0)
        if std_val.ndim == 0:
            std_val = std_val.unsqueeze(0)
        assert mean_val.shape == std_val.shape
        if mean_val.ndim == 1:
            mean_val = rearrange(mean_val, "l -> 1 1 l")
            std_val = rearrange(std_val, "l -> 1 1 l")
        elif mean_val.ndim == 2:
            mean_val = rearrange(mean_val, "c l -> 1 c l")
            std_val = rearrange(std_val, "c l -> 1 c l")

        if data.ndim == 2:
            dim_data_in = 2
            data = rearrange(data, "s l -> s 1 l")
        else:
            dim_data_in = 3

        data = data * std_val + mean_val
        if dim_data_in == 2:
            return rearrange(data, "s 1 l -> s l")
        else:
            return data

    def __repr__(self) -> str:
        return f"MeanStdScaler(mean={self.mean_val}, std={self.std_val})"


class Patchify(Transform):
    """Segment sequence into non-overlapping patches."""

    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def __call__(
        self, data: Annotated[Tensor, "batch, 1, seq_length"]
    ) -> Annotated[Tensor, "batch, patch_size, seq_length//patch_size"]:
        assert data.ndim == 3 and data.shape[1] == 1
        assert data.shape[2] % self.patch_size == 0
        return rearrange(data, "b 1 (L C) -> b C L", C=self.patch_size)

    def inverse(
        self, data: Annotated[Tensor, "batch, patch_size, seq_length//patch_size"]
    ) -> Annotated[Tensor, "batch, 1, seq_length"]:
        return rearrange(data, "b c l -> b 1 (l c)")

    def __repr__(self) -> str:
        return f"Patchify(patch_size={self.patch_size})"


class ChronoVectorize(Transform):
    """Concatenate temporally adjacent samples with circular padding."""

    def __init__(self, window_size: int):
        self.window_size = window_size

    def __call__(
        self, data: Annotated[Tensor, "batch, 1, seq_length"]
    ) -> Annotated[Tensor, "batch, window_size, seq_length"]:
        assert data.ndim == 3 and data.shape[1] == 1
        seq_length = data.shape[-1]
        data_padded = torch.nn.functional.pad(
            data, ((self.window_size - 1) // 2,) * 2, mode="circular"
        )
        data_unfolded = data_padded.unfold(2, self.window_size, 1)
        data_vectorized = data_unfolded[:, :, :seq_length, :]
        data_vectorized = rearrange(data_vectorized, "b 1 s w -> b w s")
        return data_vectorized.clone()

    def inverse(
        self, data: Annotated[Tensor, "batch, window_size, seq_length"]
    ) -> Annotated[Tensor, "batch, 1, seq_length"]:
        if data.ndim == 2:
            return data
        if data.shape[1] == 1:
            return data
        mid_channel = data.shape[1] // 2
        return data[:, mid_channel : mid_channel + 1, :]

    def __repr__(self) -> str:
        return f"ChronoVectorize(window_size={self.window_size})"
