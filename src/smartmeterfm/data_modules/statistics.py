"""
Statistical transforms: ECDF estimation, PIT, LogNorm.

Moved unchanged from utils.py.
"""

from functools import partial

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from scipy.interpolate import interp1d, splev, splrep
from scipy.stats import lognorm
from torch import Tensor

from .containers import DataShapeType


class MultiDimECDFV2:
    """
    Batched ECDF estimation.

    ref: https://github.com/MauricioSalazare/multi-copula/

    """

    def __init__(self, x, interp_mode="spline"):
        """
        Args:
            x: input data, shape (n_samples, dim1, dim2) or (n_samples, dim)
        """
        if len(x.shape) == 2:
            self.dim = x.shape[1]
            x = x
        else:
            self.dims = (x.shape[1], x.shape[2])
            self.dim = x.shape[1] * x.shape[2]
            x = rearrange(x, "n d1 d2 -> n (d1 d2)")
        self.x = x  # (n, cl)
        self.x_sorted, _ = torch.sort(
            x, dim=0
        )  # expected: DataShapeType.BATCH_SEQUENCE
        self.num_sample = x.shape[0]
        self.init_cdf(interp_mode=interp_mode)

    def init_cdf(self, interp_mode="spline") -> None:
        "returns x, cdf"
        collect_cdf = []
        collect_icdf = []
        bounds_y = []
        for channel_idx in range(self.x.shape[1]):
            x, counts = torch.unique(
                self.x_sorted.contiguous()[:, channel_idx],
                sorted=True,
                return_counts=True,
            )  # shape (n_unique,)
            events = torch.cumsum(counts, dim=0)  # shape (n_unique,)
            normalized_events = events.float() / self.num_sample
            x = x.cpu().numpy()
            normalized_events = normalized_events.cpu().numpy()
            if interp_mode == "linear":
                _x_bnd = np.r_[-np.inf, x, np.inf]  # see ref.
                _y_bnd = np.r_[0.0, normalized_events, 1.0]  # see ref
                cdf = interp1d(_x_bnd, _y_bnd)
                icdf = interp1d(_y_bnd, _x_bnd)
            else:
                cdf_rep = splrep(x, normalized_events, s=0.01)
                icdf_rep = splrep(normalized_events, x, s=0.01)
                cdf = partial(splev, tck=cdf_rep)
                icdf = partial(splev, tck=icdf_rep)
                y_min, y_max = normalized_events.min(), normalized_events.max()
            collect_cdf.append(cdf)
            collect_icdf.append(icdf)
            bounds_y.append((y_min, y_max))

        def apply_cdf(_x):
            "_x: tensor/array (n, dim)"
            converter = self.ArrayConverter(_x)
            _x = converter.convert(_x)  # (n, dim)
            _y = []
            for channel_idx in range(_x.shape[1]):
                __x = _x[:, channel_idx]  # (n,)
                __y = collect_cdf[channel_idx](__x)
                _y.append(__y)
            _y = np.stack(_y, axis=1)
            _y = converter.revert(_y)
            return _y

        def apply_icdf(_y):
            "_y: tensor/array (n, dim) in [0,1]"
            converter = self.ArrayConverter(_y)
            _y = converter.convert(_y)
            _x = []
            for channel_idx in range(_y.shape[1]):
                __y = _y[:, channel_idx]  # (n,)
                __y = np.where(
                    __y >= bounds_y[channel_idx][0], __y, bounds_y[channel_idx][0]
                )
                __y = np.where(
                    __y <= bounds_y[channel_idx][1], __y, bounds_y[channel_idx][1]
                )
                __x = collect_icdf[channel_idx](__y)
                _x.append(__x)
            _x = np.stack(_x, axis=1)
            _x = converter.revert(_x)
            return _x

        self.cdf = apply_cdf
        self.icdf = apply_icdf

    class ArrayConverter:
        def __init__(self, maybe_tensor):
            "tensor -> array"
            self.is_input_tensor = isinstance(maybe_tensor, torch.Tensor)
            if self.is_input_tensor:
                self.device = maybe_tensor.device
            else:
                self.device = None

        def convert(self, input):
            if self.is_input_tensor:
                return self._array(input)
            else:
                return self._do_nothing(input)

        def revert(self, input):
            if self.is_input_tensor:
                return self._tensor(input)
            else:
                return self._do_nothing(input)

        def _do_nothing(self, input):
            return input

        def _array(self, tensor):
            return tensor.cpu().numpy()

        def _tensor(self, array):
            return torch.from_numpy(array).to(self.device)

    def to(self, device):
        self.x = self.x.to(device)
        self.x_sorted = self.x_sorted.to(device)
        return self

    def transform(self, x):
        """
        Args:
            x: input data, shape (b, c, l) or (b, c*l) or (c, l) or (c*l)
        (x_sorted shape) (b, c*l)
        Returns:
            y: ECDF transformed data, shape (batch_size, dim1, dim2)
            or (batch_size, dim)
        """
        xndim = len(x.shape)
        input_shape_type = DataShapeType.BATCH_SEQUENCE
        if xndim == 3:
            input_shape_type = DataShapeType.BATCH_CHANNEL_SEQUENCE
            c, _ = x.shape[1], x.shape[2]
            x = rearrange(x, "b c l -> b (c l)")
        elif xndim == 2:
            if x.shape[0] * x.shape[1] == self.x_sorted.shape[1]:
                input_shape_type = DataShapeType.CHANNEL_SEQUENCE
                c, _ = x.shape[0], x.shape[1]
                x = rearrange(x, "c l -> 1 (c l)")
            else:
                input_shape_type = DataShapeType.BATCH_SEQUENCE
                pass
        elif xndim == 1:
            input_shape_type = DataShapeType.SEQUENCE
            x = rearrange(x, "cl -> 1 cl")
        else:
            raise ValueError("Invalid input shape")

        y = self.cdf(x)
        y = torch.clamp(y, 0.0, 1.0)

        if input_shape_type == DataShapeType.BATCH_CHANNEL_SEQUENCE:
            y = rearrange(y, "b (c l) -> b c l", c=c)
        elif input_shape_type == DataShapeType.CHANNEL_SEQUENCE:
            y = rearrange(y, "1 (c l) -> c l", c=c)
        elif input_shape_type == DataShapeType.SEQUENCE:
            y = rearrange(y, "1 cl -> cl")
        elif input_shape_type == DataShapeType.BATCH_SEQUENCE:
            pass

        return y

    def inverse_transform(self, y):
        yndim = len(y.shape)
        if yndim == 3:
            input_shape_type = DataShapeType.BATCH_CHANNEL_SEQUENCE
            c, _ = y.shape[1], y.shape[2]
            y = rearrange(y, "b c l -> b (c l)")
        elif yndim == 2:
            if y.shape[0] * y.shape[1] == self.x_sorted.shape[1]:
                input_shape_type = DataShapeType.CHANNEL_SEQUENCE
                c, _ = y.shape[0], y.shape[1]
                y = rearrange(y, "c l -> 1 (c l)")
            else:
                input_shape_type = DataShapeType.BATCH_SEQUENCE
                pass
        elif yndim == 1:
            input_shape_type = DataShapeType.SEQUENCE
            y = rearrange(y, "cl -> 1 cl")
        else:
            raise ValueError("Invalid input shape")

        x = self.icdf(y)

        if input_shape_type == DataShapeType.BATCH_CHANNEL_SEQUENCE:
            x = rearrange(x, "b (c l) -> b c l", c=c)
        elif input_shape_type == DataShapeType.CHANNEL_SEQUENCE:
            x = rearrange(x, "1 (c l) -> c l", c=c)
        elif input_shape_type == DataShapeType.SEQUENCE:
            x = rearrange(x, "1 cl -> cl")
        elif input_shape_type == DataShapeType.BATCH_SEQUENCE:
            pass
        return x

    def re_estimate(self, x):
        self.x = self.x.to(x.device)
        if len(x.shape) == 2:
            self.x = x
            self.dim = x.shape[1]
        else:
            self.x = rearrange(x, "n d1 d2 -> n (d1 d2)")
            self.dims = (x.shape[1], x.shape[2])
            self.dim = x.shape[1] * x.shape[2]
        self.num_sample = x.shape[0]
        self.x_sorted, _ = torch.sort(x, dim=0)
        self.init_cdf()

    def continue_estimate(self, x):
        self.x = self.x.to(x.device)
        if len(x.shape) == 3:
            x = rearrange(x, "n d1 d2 -> n (d1 d2)")
        self.x = torch.cat([self.x, x], dim=0)
        self.num_sample = self.x.shape[0]
        self.x_sorted, _ = torch.sort(self.x, dim=0)
        self.init_cdf()

    def __call__(self, x):
        """get ecdf of x"""
        x = self.transform(x)
        return x


class MultiDimECDFV1:
    """
    Batched ECDF estimation.
    """

    def __init__(self, x):
        if len(x.shape) == 2:
            self.dim = x.shape[1]
            x = x
        else:
            self.dims = (x.shape[1], x.shape[2])
            self.dim = x.shape[1] * x.shape[2]
            x = rearrange(x, "n d1 d2 -> n (d1 d2)")
        self.x = x
        self.num_sample = x.shape[0]
        self.x_sorted, _ = torch.sort(x, dim=0)

    @property
    def cdf(self) -> tuple[torch.Tensor, torch.Tensor]:
        "returns x, cdf"
        x, counts = torch.unique(
            self.x_sorted.contiguous(), dim=0, sorted=True, return_counts=True
        )
        events = torch.cumsum(counts, dim=0)
        cdf = events.float() / self.num_sample
        return x, cdf

    def to(self, device):
        self.x = self.x.to(device)
        self.x_sorted = self.x_sorted.to(device)
        return self

    def transform(self, x):
        x_sorted = self.x_sorted.to(x.device)
        xndim = len(x.shape)
        input_shape_type = DataShapeType.BATCH_SEQUENCE
        if xndim == 3:
            input_shape_type = DataShapeType.BATCH_CHANNEL_SEQUENCE
            c, _l = x.shape[1], x.shape[2]
            x = rearrange(x, "b c l -> b (c l)")
        elif xndim == 2:
            if x.shape[0] * x.shape[1] == self.x_sorted.shape[1]:
                input_shape_type = DataShapeType.CHANNEL_SEQUENCE
                c, _l = x.shape[0], x.shape[1]
                x = rearrange(x, "c l -> 1 (c l)")
            else:
                input_shape_type = DataShapeType.BATCH_SEQUENCE
                pass
        elif xndim == 1:
            input_shape_type = DataShapeType.SEQUENCE
            x = rearrange(x, "cl -> 1 cl")
        else:
            raise ValueError("Invalid input shape")

        indices = torch.searchsorted(
            x_sorted.transpose(0, 1).contiguous(),
            x.transpose(0, 1).contiguous(),
            side="right",
        ).transpose(0, 1)

        cdf_values = indices.float() / self.num_sample
        y = torch.clamp(cdf_values, 0.0, 1.0)

        if input_shape_type == DataShapeType.BATCH_CHANNEL_SEQUENCE:
            y = rearrange(y, "b (c l) -> b c l", c=c)
        elif input_shape_type == DataShapeType.CHANNEL_SEQUENCE:
            y = rearrange(y, "1 (c l) -> c l", c=c)
        elif input_shape_type == DataShapeType.SEQUENCE:
            y = rearrange(y, "1 cl -> cl")
        elif input_shape_type == DataShapeType.BATCH_SEQUENCE:
            pass

        return y

    def inverse_transform(self, y):
        x_sorted = self.x_sorted.to(y.device)
        yndim = len(y.shape)
        if yndim == 3:
            input_shape_type = DataShapeType.BATCH_CHANNEL_SEQUENCE
            c, _l = y.shape[1], y.shape[2]
            y = rearrange(y, "b c l -> b (c l)")
        elif yndim == 2:
            if y.shape[0] * y.shape[1] == x_sorted.shape[1]:
                input_shape_type = DataShapeType.CHANNEL_SEQUENCE
                c, _l = y.shape[0], y.shape[1]
                y = rearrange(y, "c l -> 1 (c l)")
            else:
                input_shape_type = DataShapeType.BATCH_SEQUENCE
                pass
        elif yndim == 1:
            input_shape_type = DataShapeType.SEQUENCE
            y = rearrange(y, "cl -> 1 cl")
        else:
            raise ValueError("Invalid input shape")

        y = torch.clamp(y, 0.0, 1.0)
        y_scaled = y * self.num_sample
        indices_lower = torch.floor(y_scaled).long()
        xlower = torch.gather(
            x_sorted, 0, indices_lower.clamp(min=1, max=self.num_sample) - 1
        )
        x = xlower

        if input_shape_type == DataShapeType.BATCH_CHANNEL_SEQUENCE:
            x = rearrange(x, "b (c l) -> b c l", c=c)
        elif input_shape_type == DataShapeType.CHANNEL_SEQUENCE:
            x = rearrange(x, "1 (c l) -> c l", c=c)
        elif input_shape_type == DataShapeType.SEQUENCE:
            x = rearrange(x, "1 cl -> cl")
        elif input_shape_type == DataShapeType.BATCH_SEQUENCE:
            pass
        return x

    def re_estimate(self, x):
        self.x = self.x.to(x.device)
        if len(x.shape) == 2:
            self.x = x
            self.dim = x.shape[1]
        else:
            self.x = rearrange(x, "n d1 d2 -> n (d1 d2)")
            self.dims = (x.shape[1], x.shape[2])
            self.dim = x.shape[1] * x.shape[2]
        self.num_sample = x.shape[0]
        self.x_sorted, _ = torch.sort(x, dim=0)

    def continue_estimate(self, x):
        self.x = self.x.to(x.device)
        if len(x.shape) == 3:
            x = rearrange(x, "n d1 d2 -> n (d1 d2)")
        self.x = torch.cat([self.x, x], dim=0)
        self.num_sample = self.x.shape[0]
        self.x_sorted, _ = torch.sort(self.x, dim=0)

    def __call__(self, x):
        """get ecdf of x"""
        x = self.transform(x)
        return x


class MultiDimECDF(MultiDimECDFV1): ...


class LogNorm(MultiDimECDF):
    """
    !not fully debugged yet.
    Batched LogNorm estimation
    !ASSUME x is in (-1,1)
    """

    def __init__(self, x, *args, **kwargs):
        assert x.min() >= -1.0
        x = self.to_zero_one(x)
        super().__init__(x)

    def to_zero_one(self, x):
        return x

    def to_minus_one_one(self, x):
        return x

    def init_cdf(self, *args, **kwargs) -> None:
        collect_cdf = []
        collect_icdf = []
        for channel_idx in range(self.x.shape[1]):
            x = self.x[:, channel_idx].cpu().numpy()
            shape, loc, scale = lognorm.fit(x)
            cdf = partial(lognorm.cdf, s=shape, loc=loc, scale=scale)
            icdf = partial(lognorm.ppf, s=shape, loc=loc, scale=scale)
            collect_cdf.append(cdf)
            collect_icdf.append(icdf)

        def apply_cdf(_x):
            converter = self.ArrayConverter(_x)
            _x = converter.convert(_x)
            _x = self.to_zero_one(_x)
            _y = []
            for channel_idx in range(_x.shape[1]):
                __x = _x[:, channel_idx]
                __y = collect_cdf[channel_idx](__x)
                _y.append(__y)
            _y = np.stack(_y, axis=1)
            _y = converter.revert(_y)
            return _y

        def apply_icdf(_y):
            converter = self.ArrayConverter(_y)
            _y = converter.convert(_y)
            _y = np.clip(_y, 0.0, 1.0)
            _x = []
            for channel_idx in range(_y.shape[1]):
                __y = _y[:, channel_idx]
                __x = collect_icdf[channel_idx](__y)
                _x.append(__x)
            _x = np.stack(_x, axis=1)
            _x = self.to_minus_one_one(_x)
            _x = converter.revert(_x)
            return _x

        self.cdf = apply_cdf
        self.icdf = apply_icdf


class PIT:
    """apply PIT to samples per channel."""

    def __init__(self, samples=None, perturb=False):
        self.ecdf = None
        if samples is not None:
            if perturb:
                samples = samples + torch.randn_like(samples) * 1e-6
            self.ecdf = MultiDimECDF(samples)
            self.transform = self.ecdf.transform
            self.inverse_transform = self.ecdf.inverse_transform

    def __call__(self, x):
        if self.ecdf is not None:
            x = self.ecdf(x)
        else:
            raise ValueError("PIT is not initialized.")
        return x

    def to(self, device):
        self.ecdf.to(device)
        return self

    def fit(self, x) -> None:
        self.ecdf = MultiDimECDF(x)
        self.transform = self.ecdf.transform
        self.inverse_transform = self.ecdf.inverse_transform

    def fit_transform(self, x):
        self.ecdf = MultiDimECDF(x)
        self.transform = self.ecdf.transform
        self.inverse_transform = self.ecdf.inverse_transform
        return self.ecdf.transform(x)


def standard_normal_cdf(x: Float[Tensor, "any"]):
    return 0.5 * (1 + torch.erf(x / np.sqrt(2)))


def standard_normal_icdf(y: Float[Tensor, "any"]):
    y = y.clamp(min=0.0, max=0.999999)
    return np.sqrt(2) * torch.erfinv(2 * y - 1)
