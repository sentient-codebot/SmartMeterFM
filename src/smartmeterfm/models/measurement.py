"""
This module implements the Measurement models.
"""

from abc import ABC, abstractmethod
from typing import Self

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from jaxtyping import Float
from torch import Tensor

# Operators
__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None) is not None:
            raise NameError(f"Operator {name} already registered.")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Operator {name} not registered.")
    return __OPERATOR__[name](**kwargs)


class AdditiveOperator(ABC):
    @abstractmethod
    def forward(self, data: Tensor, **kwargs) -> Tensor:
        """forward model, produces partial measurement from (noiseless) data"""
        raise NotImplementedError

    def __add__(self, other: Self) -> "CompositeOperator":
        """Add two operators together to create a composite operator."""
        if not isinstance(other, AdditiveOperator):
            raise TypeError("Can only add AdditiveOperator instances.")
        return CompositeOperator(self, other)

    def __radd__(self, other: Self) -> "CompositeOperator":
        """Allows 0 + operator for `sum` compatibility."""
        if other == 0:
            return self
        return NotImplemented


class CompositeOperator(AdditiveOperator):
    def __init__(self, op1: AdditiveOperator, op2: AdditiveOperator):
        self.op1 = op1
        self.op2 = op2

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        result1 = self.op1.forward(data, **kwargs)
        result2 = self.op2.forward(data, **kwargs)
        return torch.cat(
            [result1, result2],
            dim=-1,
        )

    def project(
        self,
        data,
        measurement,
        **kwargs,
    ):
        # Sequentially apply the project
        data_shape = data.shape
        # check dim
        _op1_forward = self.op1.forward(data.flatten(1), **kwargs)
        dim_op1 = _op1_forward.shape[1]
        projected = self.op1.project(
            data.flatten(1), measurement.flatten(1)[:, :dim_op1], **kwargs
        )
        projected = self.op2.project(
            projected, measurement.flatten(1)[:, dim_op1:], **kwargs
        )
        return projected.view(data_shape)


class LinearOperator(AdditiveOperator):
    """Base class for linear operators. Assume y = Ax
    Mathematical recap (assume A is full rank with #rows < #cols):
    A^+ = A^T @ (A @ A^T)^-1, pseudo inverse of A

    x = (x - A^+ @ A @ x) + A^+ @ A @ x = null component + range component
      = (I - A^+ @ A) x + (A^+ @ A) x
      = null component + range component

    """

    @abstractmethod
    def forward(self, data: Tensor, **kwargs) -> Tensor:
        "forward model, produces partial measurement from (noiseless) data"
        # Calculate A @ x
        raise NotImplementedError

    @abstractmethod
    def transpose(self, measurement: Tensor, **kwargs) -> Tensor:
        # Calculate A^+ @ y, pseudo inverse of A
        # transpose(forward(x)) = A^+ @ A @ x = range component of x
        raise NotImplementedError

    def ortho_project(
        self, data: Float[Tensor, "batch *"], **kwargs
    ) -> Float[Tensor, "batch *"]:
        # Calculate (I - A^+ @ A) x, project to the null space of A, null component
        data_shape = data.shape  # [batch, *]
        delta = self.transpose(
            self.forward(data.flatten(start_dim=1), **kwargs), **kwargs
        )  # [batch, dim]
        delta = delta.view(data_shape)  # [batch, *]
        return data - delta

    def project(
        self,
        data: Float[Tensor, "batch *"],
        measurement: Float[Tensor, "batch 1"],
        **kwargs,
    ) -> Float[Tensor, "batch *"]:
        # recombine null component of predicted x
        #   with range component of true x (from y)
        delta = self.transpose(measurement, **kwargs)  # [#batch, dim]
        delta = delta.expand(data.shape[0], -1)  # [batch, dim]
        delta = delta.view(data.shape)  # [batch, *]
        return self.ortho_project(data, **kwargs) + delta


@register_operator(name="denoise")  # denoise operator
class DenoiseOperator(LinearOperator):
    def forward(self, data):
        return data

    def transpose(self, data):
        return data

    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name="super_resolution")
class SuperResolutionOperator(LinearOperator):
    """Initialize the super resolution operator. The operator downsamples by averaging.

    Args:
        scale_factor: The factor by which to downsample the data. must be divisible by the sequence length.

    """

    def __init__(self, scale_factor: int):
        self.scale_factor = scale_factor

    def forward(
        self, data: Float[Tensor, "batch long"]
    ) -> Float[Tensor, "batch short"]:
        # Downsample the data by averaging
        data = data.flatten(1)  # [batch, sequence]
        return F.avg_pool1d(
            data.unsqueeze(1), kernel_size=self.scale_factor, stride=self.scale_factor
        ).squeeze(1)  # [batch, sequence // scale_factor]

    def transpose(
        self, measurement: Float[Tensor, "batch short"]
    ) -> Float[Tensor, "batch long"]:
        # Upsample by repeating values
        measurement = measurement.flatten(1)  # [batch, short_sequence]
        # Repeat each value scale_factor times
        upsampled = measurement.unsqueeze(-1).repeat(
            1, 1, self.scale_factor
        )  # [batch, short_sequence, scale_factor]
        return upsampled.flatten(1)  # [batch, long_sequence]


@register_operator(name="average")
class AverageOperator(LinearOperator):
    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length

    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch 1"]:
        # Calculate A @ x
        flat_data = data.flatten(start_dim=1)
        if flat_data.shape[1] != self.sequence_length:
            raise ValueError(f"Flattened data must have shape \
                [batch, {self.sequence_length}], but got {flat_data.shape}")
        mean = reduce(flat_data, "b L -> b ()", "mean")
        return mean

    def transpose(
        self, measurement: Float[Tensor, "batch 1"]
    ) -> Float[Tensor, "batch sequence"]:
        # Calculate A^+ @ y, pseudo inverse of A
        # transpose(forward(x)) = A^+ @ A @ x = range component of x
        if measurement.dim() != 2:
            raise ValueError(f"Input measurement must have shape \
                [batch 1], but got {measurement.shape}")
        pseudo_data = measurement.expand(-1, self.sequence_length)
        # shape [batch, sequence]
        return pseudo_data


@register_operator(name="inpainting")
class InpaintingOperator(LinearOperator):
    def __init__(self, mask: list | torch.Tensor):
        """Initialize the inpainting operator with a mask.

        Args:
            mask: Boolean tensor with shape [sequence] or [batch, sequence]
                 True indicates observed values, False indicates missing values
        """
        if isinstance(mask, list):
            mask = torch.tensor(mask)
        if mask.dim() == 1:
            # Convert to [1, sequence] if given a single mask
            self.mask = mask.unsqueeze(0).bool()
        else:
            self.mask = mask.bool()
        self.sequence_length = self.mask.shape[1]

    def forward(
        self, data: Float[Tensor, "batch sequence"]
    ) -> Float[Tensor, "batch sequence"]:
        """Apply mask to input data, keeping only observed values.

        Args:
            data: Input tensor with shape [batch, sequence]

        Returns:
            Masked data with missing values set to 0
        """
        if data.shape[1] != self.sequence_length or len(data.shape) != 2:
            raise ValueError(f"Input data must have shape \
                [batch, {self.sequence_length}], but got {data.shape}")

        # Expand mask to match batch size if needed
        if self.mask.shape[0] == 1 and data.shape[0] > 1:
            mask = self.mask.expand(data.shape[0], -1).to(data.device)
        else:
            mask = self.mask.to(data.device)

        return data * mask  # Element-wise multiplication keeps only observed values

    def transpose(
        self, measurement: Float[Tensor, "batch sequence"]
    ) -> Float[Tensor, "batch sequence"]:
        """Project observed values back to full space.

        Args:
            measurement: Masked tensor with observed values

        Returns:
            Tensor with observed values at their original positions
        """
        if measurement.shape[1] != self.sequence_length or len(measurement.shape) != 2:
            raise ValueError(f"Input measurement must have shape \
                [batch, {self.sequence_length}], but got {measurement.shape}")

        # Expand mask to match batch size if needed
        if self.mask.shape[0] == 1 and measurement.shape[0] > 1:
            mask = self.mask.expand(measurement.shape[0], -1).to(measurement.device)
        else:
            mask = self.mask.to(measurement.device)

        # For inpainting, the transpose operation returns the values at observed positions
        return measurement * mask


@register_operator(name="imputation")
class ImputationOperator(InpaintingOperator): ...  # alias


@register_operator(name="partial_sum")
class PartialSumOperator(LinearOperator):
    "the sum(s) of certain steps are given."

    def __init__(self):
        raise NotImplementedError("Partial sum operator not implemented yet.")


class NonLinearOperator(AdditiveOperator):
    @abstractmethod
    def forward(self, data: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def project(self, data, measurement, **kwargs):
        # if not fair, use your own definition
        raise NotImplementedError("Project function not implemented for this operator.")


@register_operator(name="max")
class MaxOperator(NonLinearOperator):
    # NOTE does not seem to work.
    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch 1"]:
        _flat_data = data.flatten(start_dim=1)
        max_value = torch.max(_flat_data, dim=1, keepdim=True).values
        return max_value

    def project(
        self, data: Float[Tensor, "batch *"], measurement: Float[Tensor, "batch 1"]
    ) -> Float[Tensor, "batch *"]:
        "should not be included in gradient computation because of in-place operation."
        data_shape = data.shape
        flat_data = data.flatten(start_dim=1).clone()
        measurement = measurement.expand_as(flat_data)  # [batch, dim]
        # Set all values >= given maximum to the maximum value
        projected_data = torch.where(
            flat_data > measurement,
            measurement,
            flat_data,
        )
        # Set the maximum value to the given maximum
        # 1. find the maximum value
        projected_data_max = torch.max(projected_data, dim=1, keepdim=True).values
        # 2. set each maximum value to the given maximum
        projected_data = torch.where(
            torch.isclose(
                projected_data,
                projected_data_max,
                rtol=1e-5,
                atol=1e-8,
            ),
            measurement,
            projected_data,
        )
        # Reshape back to original data shape
        projected_data = projected_data.view(data_shape)
        return projected_data


@register_operator(name="smooth_max")
class SmoothMaxOperator(NonLinearOperator):
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        if tau <= 0:
            raise ValueError("Tau must be positive.")

    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch 1"]:
        flat_data = data.flatten(start_dim=1)
        weights = torch.nn.functional.softmax(flat_data / self.tau, dim=1)
        # shape [batch, dim]
        out = reduce(flat_data * weights, "b d -> b ()", "sum")  # [batch, 1]
        true_max = torch.max(flat_data.detach(), dim=1, keepdim=True).values
        # shape [batch, 1]
        return out - (out.detach() - true_max)  # [batch, 1]

    def project(
        self, data: Float[Tensor, "batch *"], measurement: Float[Tensor, "batch 1"]
    ):
        "should not be included in gradient computation because of in-place operation."
        data_shape = data.shape
        flat_data = data.flatten(start_dim=1).clone()
        measurement = measurement.expand_as(flat_data)  # [batch, dim]
        # Set all values >= given maximum to the maximum value
        projected_data = torch.where(
            flat_data > measurement,
            measurement,
            flat_data,
        )
        # Set the maximum value to the given maximum
        # 1. find the maximum value
        projected_data_max = torch.max(projected_data, dim=1, keepdim=True).values
        # 2. set each maximum value to the given maximum
        projected_data = torch.where(
            torch.isclose(
                projected_data,
                projected_data_max,
                rtol=1e-5,
                atol=1e-8,
            ),
            measurement,
            projected_data,
        )
        # Reshape back to original data shape
        projected_data = projected_data.view(data_shape)
        return projected_data


@register_operator(name="smooth_min")
class SmoothMinOperator(NonLinearOperator):
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        if tau <= 0:
            raise ValueError("Tau must be positive.")

    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch 1"]:
        flat_data = data.flatten(start_dim=1)
        weights = torch.nn.functional.softmax(-flat_data / self.tau, dim=1)
        # shape [batch, dim]
        out = reduce(flat_data * weights, "b d -> b ()", "sum")
        # [batch, 1]
        true_min = torch.min(flat_data.detach(), dim=1, keepdim=True).values
        # shape [batch, 1]
        return out - (out.detach() - true_min)  # [batch, 1]

    def project(
        self, data: Float[Tensor, "batch *"], measurement: Float[Tensor, "batch 1"]
    ):
        "should not be included in gradient computation because of in-place operation."
        # ------- reverse the sign -------
        data = -data
        measurement = -measurement
        data_shape = data.shape
        flat_data = data.flatten(start_dim=1).clone()
        measurement = measurement.expand_as(flat_data)  # [batch, dim]
        # Set all values >= given maximum to the maximum value
        projected_data = torch.where(
            flat_data > measurement,
            measurement,
            flat_data,
        )
        # Set the maximum value to the given maximum
        # 1. find the maximum value
        projected_data_max = torch.max(projected_data, dim=1, keepdim=True).values
        # 2. set each maximum value to the given maximum
        projected_data = torch.where(
            torch.isclose(
                projected_data,
                projected_data_max,
                rtol=1e-5,
                atol=1e-8,
            ),
            measurement,
            projected_data,
        )
        # Reshape back to original data shape
        projected_data = projected_data.view(data_shape)
        # ------- reverse the sign bac -------
        projected_data = -projected_data
        return projected_data


@register_operator(name="lse_max")
class LogSumExpMaxOperator(NonLinearOperator):
    """LogSumExp is an upper bound of the max function.
    max(x) <= LSE(x) <= max(x) + log(n), where n is the number of elements in x.
    """

    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch 1"]:
        flat_data = data.flatten(start_dim=1)
        # lse_max = torch.logsumexp(flat_data, dim=1, keepdim=True) \
        #     - torch.log(torch.tensor(flat_data.shape[1], device=data.device))
        lse_max = torch.logsumexp(flat_data, dim=1, keepdim=True)
        true_max = torch.max(flat_data.detach(), dim=1, keepdim=True).values
        # shape [batch, 1]
        return lse_max - (lse_max.detach() - true_max)  # [batch, 1]


@register_operator(name="lse_min")
class LogSumExpMinOperator(NonLinearOperator):
    """inverse of LogSumExp, i.e. min(x) = -max(-x)."""

    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch 1"]:
        flat_data = data.flatten(start_dim=1)
        # lse_min = -torch.logsumexp(-flat_data, dim=1, keepdim=True) \
        #     + torch.log(torch.tensor(flat_data.shape[1], device=data.device))
        lse_min = -torch.logsumexp(-flat_data, dim=1, keepdim=True)
        true_min = torch.min(flat_data.detach(), dim=1, keepdim=True).values
        # shape [batch, 1]
        return lse_min - (lse_min.detach() - true_min)


@register_operator(name="_pnorm_max")
class PNormMaxOperator(NonLinearOperator):
    """use large p-norm to approximate inf-norm.
    [deprecated] unstable, not working at all.
    """

    def __init__(self, p: int = 100):
        super().__init__()
        self.p = p

    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch 1"]:
        flat_data = data.flatten(start_dim=1)

        # p-norm
        _norm = torch.linalg.vector_norm(flat_data, ord=self.p, dim=1, keepdim=True)
        # shape [batch, 1]
        _true_max = torch.max(flat_data.detach(), dim=1, keepdim=True).values
        # shape [batch, 1]
        # out = _norm - (_norm.detach() - true_max)
        out = _norm

        return out


@register_operator(name="ldn_odn")
class LDNODNOperator(NonLinearOperator):
    """LDN/ODN operator.

    LDN: sum of all positive values divided by sequence length.
    ODN: sum of all negative values divided by sequence length.
    LDN-ODN = net sum of all values.

    """

    def __init__(self, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length

    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch 2"]:
        flat_data = data.flatten(start_dim=1)
        # LDN
        ldn = (
            torch.sum(
                torch.nn.functional.relu(flat_data),
                dim=1,
                keepdim=True,
            )
            / self.sequence_length
        )  # [batch, 1]
        # ODN
        odn = (
            torch.sum(
                torch.nn.functional.relu(-flat_data),
                dim=1,
                keepdim=True,
            )
            / self.sequence_length
        )  # [batch, 1]
        # shape [batch, 2]
        return torch.cat([ldn, odn], dim=1)

    def project(
        self, data: Float[Tensor, "batch *"], measurement: Float[Tensor, "batch 2"]
    ) -> Float[Tensor, "batch *"]:
        """project the data to satisfy ldn and odn constraints."""
        data_shape = data.shape
        data = data.flatten(start_dim=1)
        data_ldn = (
            torch.sum(
                torch.nn.functional.relu(data),
                dim=1,
                keepdim=True,
            )
            / self.sequence_length
        )  # [batch, 1]
        data_odn = (
            torch.sum(
                torch.nn.functional.relu(-data),
                dim=1,
                keepdim=True,
            )
            / self.sequence_length
        )
        given_ldn = measurement[:, 0].unsqueeze(1)  # [batch, 1]
        given_odn = measurement[:, 1].unsqueeze(1)  # [batch, 1]

        # adjust separately by either LDN or ODN
        scaled_by_ldn = data * (given_ldn / data_ldn.clamp(min=1e-8))
        scaled_by_odn = data * (given_odn / data_odn.clamp(min=1e-8))

        # adjust the data
        data = torch.where(
            data >= 0,
            scaled_by_ldn,
            scaled_by_odn,
        )

        # reshape back to original data shape
        projected_data = data.view(data_shape)

        return projected_data


# Noises
__NOISE__ = {}


def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None) is not None:
            raise NameError(f"Noise {name} already registered.")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Noise {name} not registered.")
    return __NOISE__[name](**kwargs)


class Noise(ABC):
    """Define the noise model.

    Methods:
    - __call__: apply the noise model (forward) to the data.
    - forward: defines the noise.

    """

    def __call__(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch *"]:
        return self.forward(data)

    @abstractmethod
    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch *"]:
        """Apply the noise model to the data."""
        raise NotImplementedError

    def adjust_shape(self, measurement: Tensor, ref_tensor: Tensor) -> Tensor:
        """
        ref_tensor: [batch, dim]
        measurement: [batch, dim], [1, dim] or [dim]

        return: measurement in [batch, dim]
        """
        if measurement.dim() == 2 and ref_tensor.dim() >= 2:
            if ref_tensor.shape[0] % measurement.shape[0] == 0:
                # expand the measurement to match the batch size of ref_tensor
                expand_factor = ref_tensor.shape[0] // measurement.shape[0]
                measurement = rearrange(measurement, "B ... -> 1 B ...")
                measurement = measurement.expand(
                    expand_factor, *measurement.shape[1:]
                )  # [N, B, ...]
                measurement = rearrange(measurement, "N B ... -> (N B) ...")
        if measurement.shape == ref_tensor.shape:
            return measurement
        if measurement.shape[1:] == ref_tensor.shape[1:] and measurement.shape[0] == 1:
            return measurement.expand_as(ref_tensor)
        if measurement.shape == ref_tensor.shape[1:]:
            return measurement.unsqueeze(0)

        raise ValueError(f"Measurement shape {measurement.shape} \
                does not match pseudo measurement shape {ref_tensor.shape}.")

    @abstractmethod
    def log_likelihood(
        self,
        pseudo_measurement: Float[Tensor, "batch *"],
        measurement: Float[Tensor, "batch *"],
        valid_length: int | None = None,
    ) -> Float[Tensor, "batch"]:
        """Calculate the log likelihood of the pseudo measurement (from predicted data) given the noise model."""
        raise NotImplementedError


@register_noise(name="clean")
class CleanNoise(Noise):
    """Clean data, no noise."""

    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch *"]:
        return data

    def log_likelihood(self, *args, **kwargs):
        # Assuming clean data has zero noise, log likelihood is not defined
        raise NotImplementedError("Log likelihood not defined for clean data.")


@register_noise(name="gaussian")
class GaussianNoise(Noise):
    """Gaussian noise model."""

    def __init__(self, sigma: float | tuple[float, ...]):
        if isinstance(sigma, tuple):
            self.sigma = torch.tensor(sigma)
        else:
            self.sigma = torch.tensor([sigma])  # both one-dim
        if torch.any(self.sigma <= 0):
            raise ValueError("Sigma must be positive.")

    def forward(self, data: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch dim"]:
        noise = torch.randn_like(data) * self.sigma.view(1, -1)
        return data + noise

    def log_likelihood(
        self,
        pseudo_measurement: Float[Tensor, "batch dim"],
        measurement: Float[Tensor, "#batch dim"],
    ) -> Float[Tensor, "batch"]:
        """
        Return the unnormalized log likelihood of the pseudo measurement given\
            the noise model.
        Assuming Gaussian noise, log likelihood is calculated as:
            log p(y|x) = -0.5 * ((y - x) / sigma)^2 - log(sqrt(2 * pi * sigma^2))
            where y is the measurement and x is the pseudo measurement

        """
        if self.sigma.shape[0] != 1 and measurement.shape[1] != self.sigma.shape[0]:
            raise ValueError(
                f"Measurement shape {measurement.shape} does not match sigma\
                    shape {self.sigma.shape} if not singleton."
            )
        sigma = self.sigma.to(pseudo_measurement.device)
        measurement = self.adjust_shape(measurement, ref_tensor=pseudo_measurement)
        diff = (pseudo_measurement - measurement).pow(2) / sigma.view(1, -1).pow(2)
        # shape [batch, dim]
        norm = diff.sum(dim=1)  # shape [batch]
        log_likelihood = -0.5 * norm  # shape [batch]
        return log_likelihood  # shape [batch]


@register_noise(name="poisson")
class PoissonNoise(Noise):
    """Poisson noise model. Poisson distribution support is non-negative integers.
    NOTE: this is usually used for count data, e.g. photon counts. (-> images with
    discrete brightness values (0, 1, ..., 255). Applicability needs investigation
    for time series data. The scaling factor (255 for image) is set to 1000.
    )
    """

    def __init__(self, rate: float):
        if rate <= 0:
            raise ValueError("Rate must be positive.")
        self.rate = rate

    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch *"]:
        # NOTE assuming data range [-1, 1]
        device = data.device
        data = (data.clone().cpu() + 1.0) / 2.0
        data = data.clamp(min=0)
        data = torch.poisson(data * 1000.0 * self.rate) / 1000.0 / self.rate
        data = data * 2.0 - 1.0
        # data = data.clamp(min=-1, max=1)
        data = data.to(device)
        return data


@register_noise(name="laplace")
class LaplaceNoise(Noise):
    """Laplace noise model. Laplace distribution promotes sparsity."""

    def __init__(self, b: float | tuple[float, ...]):
        if isinstance(b, tuple):
            self.b = torch.tensor(b)
        else:
            self.b = torch.tensor([b])
        if any(self.b <= 0):
            raise ValueError("b must be positive.")

    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch *"]:
        raise NotImplementedError("Laplace forward not implemented yet.")
        noise = torch.randn_like(data) * self.b
        return data + noise

    def log_likelihood(
        self,
        pseudo_measurement: Float[Tensor, "batch dim"],
        measurement: Float[Tensor, "#batch dim"],
    ) -> Float[Tensor, "batch"]:
        """
        Return the unnormalized log likelihood of the pseudo measurement given\
            the noise model.
        Assuming Laplace noise, log likelihood is calculated as:
            log p(y|x) = -|y - x| / b
            where y is the measurement and x is the pseudo measurement
        """
        if self.b.shape[0] != 1 and measurement.shape[1] != self.b.shape[0]:
            raise ValueError(
                f"Measurement shape {measurement.shape} does not match b\
                    shape {self.b.shape} if not singleton."
            )
        b = self.b.to(pseudo_measurement.device)
        measurement = self.adjust_shape(measurement, ref_tensor=pseudo_measurement)
        diff = (pseudo_measurement - measurement).pow(1) / b.view(1, -1).pow(1)
        # shape [batch, dim]
        norm = diff.sum(dim=1)  # shape [batch]
        log_likelihood = -norm
        return log_likelihood


@register_noise(name="augmented_laplace")
class AugmentedLaplaceNoise(Noise):
    """Augmented Laplace noise model.
    = L(0, b^2) + epsilon * N(0, 2b^2)
    my note: not much difference.
    """

    def __init__(self, b: float):
        if b <= 0:
            raise ValueError("b must be positive.")
        self.b = b
        self.epsilon = 1e-4

    def forward(self, data: Float[Tensor, "batch *"]) -> Float[Tensor, "batch *"]:
        raise NotImplementedError("Augmented Laplace forward not implemented yet.")

    def log_likelihood(
        self,
        pseudo_measurement: Float[Tensor, "batch dim"],
        measurement: Float[Tensor, "#batch dim"],
    ) -> Float[Tensor, "batch"]:
        """
        Return the unnormalized log likelihood of the pseudo measurement given\
            the noise model.
        Assuming Laplace noise, log likelihood is calculated as:
            log p(y|x) = -|y - x| / b
            where y is the measurement and x is the pseudo measurement
        """
        measurement = self.adjust_shape(measurement, ref_tensor=pseudo_measurement)
        diff = (pseudo_measurement - measurement).norm(1, dim=1)
        l2_diff = (pseudo_measurement - measurement).norm(2, dim=1)
        log_likelihood = -diff / self.b - self.epsilon * 0.5 * (
            0.707 * l2_diff / self.b
        ).pow(2)
        return log_likelihood  # shape [batch]


class L1NormProximal(torch.autograd.Function):
    """applying proximal operator to the data during the gradient step.

    Forward:
    - L1 norm: sum(abs(x))

    Backward

    my note: the result does not seem to converge to the optimal.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        lambda_param: float = 0.005,
        dim: int | None = None,
        keepdim: bool = False,
    ):
        ctx.save_for_backward(x)
        ctx.lambda_param = lambda_param
        ctx.dim = dim
        ctx.keepdim = keepdim

        if dim is None:
            return torch.sum(torch.abs(x))
        else:
            return torch.sum(torch.abs(x), dim=dim, keepdim=keepdim)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        lambda_param = ctx.lambda_param
        dim = ctx.dim
        keepdim = ctx.keepdim

        if dim is not None and not keepdim:
            grad_output = grad_output.unsqueeze(dim)

        if grad_output.shape != x.shape:
            grad_output = grad_output.expand_as(x)

        # Instad of returning the standard L1 gradient, sign(x),
        # return the proximal gradient which implements soft thresholding
        # For values > lambda, shift by -lambda
        # For values < -lambda, shift by +lambda
        grad = torch.where(
            x > lambda_param,
            grad_output * (x - lambda_param),
            torch.zeros_like(x),
        )

        grad = torch.where(
            x < -lambda_param,
            grad_output * (x + lambda_param),
            grad,
        )

        # For values in [-lambda, lambda], gradient is zero
        # (already initialized to zero)

        return grad, None, None, None
