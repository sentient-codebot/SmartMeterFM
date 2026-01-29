import os
import traceback
from collections import namedtuple
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Annotated as Float
from typing import Annotated as Int
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from numpy import ndarray
from scipy import linalg
from scipy.stats import ks_2samp, pearsonr, wasserstein_distance
from torch import Tensor
from torch.signal import windows


def get_tqdm_by_env():
    """Use environment variable to determine tqdm type."""
    if os.getenv("JUPYTER_NOTEBOOK") or os.getenv("JPY_PARENT_PID"):
        try:
            from tqdm.notebook import tqdm

            return tqdm
        except ImportError:
            pass

    from tqdm import tqdm

    return tqdm


tqdm = get_tqdm_by_env()


def source_mean(source, target):
    return source.mean()


def target_mean(source, target):
    return target.mean()


def source_std(source, target):
    return source.std()


def target_std(source, target):
    return target.std()


def estimate_autocorrelation(
    x: Float[np.ndarray, "batch, sequence"],
) -> Float[np.ndarray, "sequence, sequence"]:
    """estimate autocorrelation, we deduct the mean before calculation."""
    assert isinstance(x, np.ndarray)
    x = x - np.mean(x, axis=1, keepdims=True)
    x = x.reshape(
        x.shape[0], x.shape[1], 1
    )  # shape: [batch, sequence, 1], batched column vectors
    x_T = np.transpose(
        x, axes=(0, 2, 1)
    )  # shape: [batch, 1, sequence], batched row vectors
    autocorrelation = np.matmul(x, x_T)  # shape: [batch, sequence, sequence]
    averaged_autocorrelation = np.mean(
        autocorrelation, axis=0
    )  # shape: [sequence, sequence]

    return averaged_autocorrelation


class MkMMD(nn.Module):
    """Calculate the multi-maximum mean discrepancy (MK-MMD)

    For future: add linear coefficients to each kernel.
    Args:
        ...
        coefficient: str, 'ones' or 'auto'. If 'ones', the coefficient of each kernel is 1.
                            If 'auto', the coefficient of each kernel is from Hamming window.

    Forward:
        source: batch of vectors. torch.tensor, shape (N, T)
        target: batch of vectors. torch.tensor, shape (N, T)

    Returns:
        tensor (scalr): MkMMD
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        kernel_mul: float = 2.0,
        num_kernel: int = 5,
        fix_sigma: float | None = None,
        coefficient: str = "ones",
    ):
        super().__init__()
        assert coefficient in {"auto", "ones"}
        self.coefficient = coefficient
        self.num_kernel = num_kernel
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type  # TODO: currently not used

    def gaussian_kernel(
        self,
        source,
        target,
        kernel_mul=2.0,
        num_kernel=5,
        fix_sigma=None,
        coefficient="ones",
    ):
        """
        L2_square = | A     | B     |
                    | B^T   | C     |
        """
        assert coefficient in {"auto", "ones"}
        _num_sample_s, num_sample_t = (
            int(source.shape[0]),
            int(target.shape[0]),
        )  # b1, b2
        # n_samples = num_sample_s + num_sample_t

        # source to source
        L2_squared_A = torch.cdist(source, source, p=2) ** 2  # shape: [b1, b1]
        L2_squared_A = L2_squared_A - torch.diag(
            torch.diagonal(L2_squared_A)
        )  # remove diagonal
        # target to target
        L2_squared_C = torch.cdist(target, target, p=2) ** 2  # shape: [b2, b2]
        L2_squared_C = L2_squared_C - torch.diag(
            torch.diagonal(L2_squared_C)
        )  # remove diagonal
        # source to target
        L2_squared_B = torch.cdist(source, target, p=2) ** 2  # shape: [b1, b2]

        # Compute bandwidth
        if fix_sigma is not None:
            if isinstance(fix_sigma, float):
                fix_sigma = torch.tensor(fix_sigma, device=L2_squared_A.device)
            bandwidth = fix_sigma
        else:
            # bandwidth = torch.sum(L2_squared.data) / (n_samples ** 2 - n_samples)
            bandwidth = torch.sum(L2_squared_C) / (num_sample_t**2 - num_sample_t)

        base_bandwidth = bandwidth / (kernel_mul ** (num_kernel // 2))  # base bandwidth
        bandwidth_list = [
            base_bandwidth * (kernel_mul**i) for i in range(num_kernel)
        ]  # this way the original bandwidth is in the middle of this list

        self.register_buffer(
            "bandwidth_list", torch.tensor(bandwidth_list)
        )  # shape: (num_kernel,)
        self.register_buffer("bandwidth", bandwidth)  # shape: scalar

        # Compute kernel values
        if coefficient == "ones":
            coef_val = torch.ones(
                num_kernel, device=L2_squared_A.device
            )  # shape: [num_kernel]
            coef_val = rearrange(coef_val, "k -> k () ()")  # shape: [num_kernel, 1, 1]
        elif coefficient == "auto":
            coef_val = windows.hamming(
                num_kernel, device=L2_squared_A.device
            )  # shape: [num_kernel]
            coef_val = (
                coef_val / coef_val.sum() * num_kernel
            )  # normalize, sum == num_kernel
            coef_val = rearrange(coef_val, "k -> k () ()")  # shape: [num_kernel, 1, 1]
        else:
            raise ValueError(f"coefficient {coefficient} not supported")

        def _calc_kernel_val(L2_squared):
            kernel_val = 0.0
            for kernel_idx, bandwidth_temp in enumerate(bandwidth_list):
                kernel_val += (
                    torch.exp(-L2_squared / bandwidth_temp).sum()
                    * coef_val[kernel_idx].item()
                )  # shape: scalar

            return kernel_val

        XX = _calc_kernel_val(
            L2_squared_A
        )  # upper left block, scalar, mean of shape (bs_source, bs_source)
        YY = _calc_kernel_val(
            L2_squared_C
        )  # lower right block, scalar, mean of shape (bs_target, bs_target)
        XY = _calc_kernel_val(
            L2_squared_B
        )  # upper right block, scalar, mean of shape (bs_source, bs_target)

        # Sum kernel values
        return XX, YY, XY

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X - f_of_Y
        loss = torch.mean(delta * delta)
        return loss

    @staticmethod
    def flatten_to_2d(x):
        if x.ndim == 2:
            return x
        elif x.ndim == 3:
            x = rearrange(x, "b d l -> b (d l)")
            return x
        else:
            raise ValueError(f"x should be 2d or 3d, but got {x.ndim}")

    def forward(self, source, target):
        "source/target: shape (batch, D, sequence). last 2d will be flattened."
        source, target = map(self.flatten_to_2d, (source, target))
        # compute the mkmmd of two samples of shape [batch, sequence]
        bs_source, bs_target = int(source.shape[0]), int(target.shape[0])
        XX, YY, XY = self.gaussian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            num_kernel=self.num_kernel,
            fix_sigma=self.fix_sigma,
            coefficient=self.coefficient,
        )  # shape: [bs_source+bs_target, bs_source+bs_target]
        XX *= 1 / (
            bs_source * max((bs_source - 1), 1.0)
        )  # upper left block, scalar, mean of shape (bs_source, bs_source)
        YY *= 1 / (
            bs_target * max((bs_target - 1), 1.0)
        )  # lower right block, scalar, mean of shape (bs_target, bs_target)
        XY *= 2 / (
            bs_source * bs_target
        )  # upper right block, scalar, mean of shape (bs_source, bs_target)
        loss = XX + YY - XY
        return loss


NormalizedMMDResult = namedtuple(
    "NormalizedMMDResult", ["p_value", "true_mmd", "null_mmds"]
)


def normalized_mmd(
    source: Tensor,
    target: Tensor,
    mmd_fn: Callable,
    num_permutations: int = 1000,
    return_all: bool = False,
) -> Float[Tensor, ""]:
    """Normalized MMD between two distributions.

    Args:
        source (Tensor): source distribution
        target (Tensor): target distribution
        mmd_fn (Callable): MMD function to use
        num_permutations (int): number of samples to use for normalization
        return_all (bool): if True, return all null MMDs and the true MMD

    Returns:
        Tensor: p_value
    """
    true_mmd = mmd_fn(source, target)
    combined = torch.cat((source, target), dim=0)
    n_x = source.shape[0]

    null_mmds = []
    for _ in tqdm(range(num_permutations), leave=False):
        perm_indices = torch.randperm(combined.shape[0])
        X_perm = combined[perm_indices[:n_x]]
        Y_perm = combined[perm_indices[n_x:]]
        null_mmds.append(mmd_fn(X_perm, Y_perm).item())

    null_mmds = torch.tensor(null_mmds)
    p_value = (null_mmds >= true_mmd.item()).float().mean()

    if return_all:
        return NormalizedMMDResult(
            p_value=p_value, true_mmd=true_mmd, null_mmds=null_mmds
        )

    return p_value


def reshape_3d_2d_pre_fn(func):
    def wrapped(source, target, *args, **kwargs):
        if source.ndim == 3:
            _source = source.squeeze(1)
        if target.ndim == 3:
            _target = target.squeeze(1)
        return func(_source, _target, *args, **kwargs)

    return wrapped


@torch.no_grad()
def kl_divergence(
    source: Float[Tensor, "batch1 channel sequence"],
    target: Float[Tensor, "batch2 channel sequence"],
) -> Float[Tensor, ""]:
    r"""source/target: shape (batch, D, sequence). last 2d will be flattened.
    mathematical definition:
        target distribution: p(x)
        source distribution: q(x).
    assume support(q) and support(p) are close enough.
        KL divergence: KL(p||q) = \int p(x) log(p(x)/q(x)) dx

    For future: use two infinite s

    Args:
        source (Tensor): source distribution
        target (Tensor): target distribution
    Returns:
        Tensor (scalar): KL divergence
    """
    source, target = source.float().cpu(), target.float().cpu()
    source, target = map(torch.flatten, (source, target))  # (batch_1,) (batch_2,)
    # find the optimal bins
    min = torch.min(source.min(), target.min())
    max = torch.max(source.max(), target.max())
    bins = torch.linspace(min, max, 200)
    # bin_width = (bins[1] - bins[0]).item()

    # compute histogram of source and target
    p_source, bin_source = torch.histogram(
        source,
        bins=bins,
    )
    p_target, bin_target = torch.histogram(target, bins=bins)
    #   NOTE: this way the bins are supposed to be aligned.

    # compute the prob
    p_source = (
        p_source * torch.logical_and(p_source > 0, p_target > 0).float()
    )  # remove zero bins
    p_target = (
        p_target * torch.logical_and(p_source > 0, p_target > 0).float()
    )  # remove zero bins
    p_source = p_source / p_source.sum()  # re-normalize
    p_target = p_target / p_target.sum()  # re-normalize

    # compute the kl divergence
    # NOTE: how to deal with when p_source == 0 but p_target != 0?
    kl_div = 0.0
    for _bin, p, q in zip(bins, p_source, p_target, strict=False):
        if p > 0 and q > 0:
            kl_div += p * torch.log(p / q)  # shape: scalar

    return kl_div


@torch.no_grad()
def ws_distance(
    source: Float[Tensor, "batch1 channel sequence"],
    target: Float[Tensor, "batch2 channel sequence"],
) -> Float[Tensor, ""]:
    """Calculates the Wasserstein distance between samples of two 1D distributions.

    Args:
        source (Tensor): source distribution
        target (Tensor): target distribution
    Returns:
        Tensor (scalar): Wasserstein distance

    """
    source = source.flatten().float().cpu().numpy()
    target = target.flatten().float().cpu().numpy()
    _ws_dist = wasserstein_distance(source, target)

    return _ws_dist


@torch.no_grad()
def ks_2samp_test(
    source: Float[Tensor, "batch1 channel sequence"],
    target: Float[Tensor, "batch2 channel sequence"],
) -> tuple[float, float]:
    """Calculates the Kolmogorov-Smirnov 2-sample statistic.

    returns:
        KstestResult(D statistic: float, pvalue: float)

            - D statistic: lower=better, the absolute max distance between the CDFs of the two samples.
            - pvalue: higher=better, a significance level of the test.

                if pvalue < (1-confidence_threshold), then reject the null hypothesis that the distributions are the same.
                otherwise, if pvalue >= (1-confidence_threshold), then accept the null hypothesis that the distributions are the same.

    """
    source = source.flatten().float().cpu().numpy()
    target = target.flatten().float().cpu().numpy()
    _ks_2samp = ks_2samp(source, target)

    return _ks_2samp.statistic, _ks_2samp.pvalue


def ks_test_d(*args, **kwargs) -> float:
    """wrapper for ks_2samp_test"""
    return ks_2samp_test(*args, **kwargs)[0]


def ks_test_p(*args, **kwargs) -> float:
    """wrapper for ks_2samp_test"""
    return ks_2samp_test(*args, **kwargs)[1]


@torch.no_grad()
def estimate_psd(
    x: Float[Tensor, "batch sequence"],
    window_size: Int,
) -> Float[Tensor, "sequence window_size"]:
    assert window_size % 2 == 1
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    padded = F.pad(
        x, (window_size // 2, window_size // 2), mode="circular"
    )  # shape: [batch, sequence+window_size-1]
    unfolded = padded.unfold(
        dimension=1, size=window_size, step=1
    )  # shape: [batch, sequence, window_size]
    fouriered = torch.fft.fft(
        unfolded, dim=2, norm="ortho"
    )  # shape: [batch, sequence, window_size]
    fourier_mag = torch.abs(fouriered)  # shape: [batch, sequence, window_size]
    fourier_mag = torch.mean(fourier_mag, dim=0)  # shape: [sequence, window_size]

    pool_size = window_size // 2 + 1
    pool_stride = window_size // 2
    padded_mag = F.pad(
        fourier_mag, (window_size // 2, window_size // 2), mode="circular"
    )  # shape: [batch, sequence+window_size-1]
    avg_mag = F.avg_pool1d(
        rearrange(padded_mag, "seq win -> 1 win seq"),
        kernel_size=pool_size,
        stride=pool_stride,
    )

    return rearrange(avg_mag, "1 win seq -> seq win").numpy()


@torch.no_grad()
def calculate_frechet(
    source: Float[Tensor | ndarray, "batch sequence"],
    target: Float[Tensor | ndarray, "batch sequence"],
) -> Float[ndarray, ""]:
    # Adjust data type
    if isinstance(source, Tensor):
        source = source.float().cpu().numpy()
    if isinstance(target, Tensor):
        target = target.float().cpu().numpy()

    if source.ndim == 3:
        mid_dim = source.shape[1] // 2
        source = source[:, mid_dim, :]
    if target.ndim == 3:
        mid_dim = target.shape[1] // 2
        target = target[:, mid_dim, :]

    # cov. numerical stability enhancement with eps
    eps = 1e-6

    # Calculating Frechet Distance
    mean_source = np.mean(source, axis=0)
    cov_source = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * eps

    mean_target = np.mean(target, axis=0)
    cov_target = np.cov(target, rowvar=False) + np.eye(target.shape[1]) * eps

    if cov_source.ndim < 2:
        cov_source = cov_source.reshape(1, 1)
        cov_target = cov_target.reshape(1, 1)

    diff = mean_source - mean_target
    covmean, _ = linalg.sqrtm(cov_source.dot(cov_target), disp=False)

    # Numerical error might give a complex component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    frechet_dist = (
        np.dot(diff, diff)
        + np.trace(cov_source)
        + np.trace(cov_target)
        - 2 * np.trace(covmean)
    )

    return frechet_dist


def _to_tensor(
    x: NormalizedMMDResult | int | float | Tensor | ndarray | dict | list | tuple,
    device=None,
):
    """Convert input to a tensor."""
    if isinstance(x, Tensor):
        return x.to(device) if device else x
    elif isinstance(x, NormalizedMMDResult):
        return _to_tensor(x._asdict(), device=device)
    elif isinstance(x, np.ndarray):
        _x = torch.from_numpy(x)
        if device:
            return _x.to(device)
        return _x
    elif isinstance(x, float | int):
        return torch.tensor(x, device=device)
    elif isinstance(x, list | tuple):
        return torch.tensor(x, device=device)
    elif isinstance(x, dict):
        return {k: _to_tensor(v, device=device) for k, v in x.items()}
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


class MultiMetric:
    def __init__(
        self, metric_fns: dict[str, Callable], compute_on_cpu: bool = False, **kwargs
    ):
        """
        non torchmetrics based implementation. it is so bad when it comes to ddp sync
        Args:
            metric_fns: Dictionary mapping metric names to their computation functions.
                       Each function should take (source_features, target_features) as input.
        """
        super().__init__()
        self._printed = False

        # Store metric names and functions
        self.metric_names = list(metric_fns.keys())
        self._device = torch.device("cpu")  # following
        self._register_metric_fns(metric_fns)
        self.compute_on_cpu = compute_on_cpu

        # Simple state storage, no DDP sync needed since we handle gathering elsewhere
        self.source_features = []
        self.target_features = []

    @property
    def device(self):
        "the device on which the metric is computed"
        return self._device

    def to(self, device):
        self._device = device
        return self

    def _register_metric_fns(self, metric_fns):
        """Register metric functions as class methods to avoid pickle issues"""
        for name, fn in metric_fns.items():
            setattr(MultiMetric, f"compute_{name}", staticmethod(fn))

    def update(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with pre-gathered data
        Note: This should only be called from the main process with gathered data
        """
        self.source_features.append(source.detach())
        self.target_features.append(target.detach())

    def compute(self) -> dict[str, torch.Tensor | dict]:
        """
        Compute metrics on accumulated data.
        Called by PyTorch Lightning at the end of validation epoch.
        """
        if len(self.source_features) == 0:
            return {}

        # Concatenate all accumulated features
        source_all = torch.cat(self.source_features, dim=0)
        target_all = torch.cat(self.target_features, dim=0)

        if self.compute_on_cpu:
            source_all = source_all.cpu()
            target_all = target_all.cpu()

        source_all = source_all.float()
        target_all = target_all.float()

        if not self._printed:
            print(f"source_all: {source_all.shape}, target_all: {target_all.shape}")
            self._printed = True

        # Compute all metrics
        results = {}
        for name in self.metric_names:
            metric_fn = getattr(self, f"compute_{name}")
            try:
                result = metric_fn(source_all, target_all)
                # Ensure result is a tensor
                if not isinstance(result, torch.Tensor):
                    result = _to_tensor(result, device=self.device)
                results[name] = result
            except Exception:
                traceback.print_exc()
                results[name] = torch.tensor(float("nan"), device=self.device)

        return results

    def reset(self):
        self.source_features = []
        self.target_features = []

    def __call__(self, source, target):
        self.update(source, target)
        _result = self.compute()
        self.reset()
        return _result

    def forward(self, source, target):
        self.update(source, target)
        _result = self.compute()
        self.reset()
        return _result


def _tensor_stats(tensor: torch.Tensor) -> str:
    return f"mean:{tensor.mean().item():.4f}|std:{tensor.std().item():.4f}|max:{tensor.max().item():.4f}|min:{tensor.min().item():.4f}|median:{tensor.median().item():.4f}"


def filter_samples_by_conditions(
    dataloader: torch.utils.data.DataLoader,
    conditions,
    num_samples: int | None = None,
    max_load_batches: int | None = None,
):
    """filter function for samples satifying a specific condition.

    Arguments:
        dataloader -- returns [profile, label] tuple
        conditions -- expected conditions to be satisfied. [str, Float[Tensor, "dim"]]

    Keyword Arguments:
        num_samples -- number of samples to return (default: {None}, all)
        max_load_batches -- only load a subset of samples to save time (default: {None})

    Return:
        Tensor -- filtered samples
    """
    all_profiles = []
    all_labels = []

    # pop None-values in the dictionary
    none_keys = [key for key, value in conditions.items() if value is None]
    for key in none_keys:
        conditions.pop(key)

    for idx, batch in enumerate(dataloader):
        profiles, labels = batch
        mask = torch.ones(profiles.shape[0], dtype=torch.bool)
        for _label_name, _label_tensor in conditions.items():
            mask = torch.logical_and(
                mask,
                torch.all(
                    _label_tensor.unsqueeze(0).long() == labels[_label_name].long(),
                    dim=(1),
                ),
            )
        # append profiles
        filtered_profiles = profiles[mask]
        all_profiles.append(filtered_profiles)
        # append labels
        filtered_labels = {}
        for _label_name, _label_tensor in labels.items():
            filtered_labels[_label_name] = _label_tensor[mask]
        all_labels.append(filtered_labels)

        if (
            idx is not None
            and max_load_batches is not None
            and idx == max_load_batches - 1
        ):
            break

    if len(all_profiles) == 0:
        return torch.tensor([]), {}

    all_profiles = torch.cat(all_profiles, dim=0)
    out_labels = {}
    for _label_name in all_labels[0].keys():
        _cat_label = torch.cat([_labels[_label_name] for _labels in all_labels], dim=0)
        if num_samples is not None and all_profiles.shape[0] > num_samples:
            _cat_label = _cat_label[:num_samples]
        out_labels[_label_name] = _cat_label

    # Limit number of samples if specified
    if num_samples is not None and all_profiles.shape[0] > num_samples:
        all_profiles = all_profiles[:num_samples]

    return all_profiles, out_labels


def crps_empirical_dimwise(
    source: Float[Tensor, "batch sequence"],
    target: Float[Tensor, "sequence"],
) -> Float[Tensor, "sequence"]:
    """Calculate the empirical CRPS for each dimension.

    Args:
        source (Tensor): collection of source (prediction) samples, shape (batch, sequence)
        target (Tensor): target (real observation), shape (sequence,)
    Returns:
        Tensor: empirical CRPS for each dimension, shape (sequence,)

    """
    B, D = source.shape
    device = source.device

    source_sorted, _ = torch.sort(source, dim=0)  # shape: (B, D)

    # evenly spread quantiles
    tau = (2 * torch.arange(1, B + 1, device=device) - 1) / (2 * B)
    tau = tau.unsqueeze(1)  # shape: (B, 1)

    # shape adjust for target
    target_expanded = target.unsqueeze(0)  # shape: (1, D)

    # compute quantile loss for all dimensions simultaneously
    diff = target_expanded - source_sorted  # shape: (B, D)
    losses = 2 * torch.where(
        diff >= 0,
        tau * diff,
        (tau - 1) * diff,
    )  # shape: (B, D)
    crps = torch.mean(losses, dim=0)  # shape: (D,)

    return crps


def crps_empirical_vs_empirical(targets, sources):
    """
    CRPS between two empirical distributions

    Args:
        targets: [batch_2_dim, D] - "true" observations (parallel universes)
        sources: [batch_dim, D] - forecast samples

    Returns:
        crps: [D] - CRPS for each dimension
    """
    _B, D = sources.shape
    _B2 = targets.shape[0]

    crps_dims = []

    for d in range(D):
        source_d = sources[:, d]  # [batch_dim]
        target_d = targets[:, d]  # [batch_2_dim]

        # Cross term: E[|X - Y|]
        cross_diff = torch.abs(
            source_d.unsqueeze(1) - target_d.unsqueeze(0)
        )  # [batch_dim, batch_2_dim]
        cross_term = torch.mean(cross_diff)

        # Source self-interaction: E[|X - X'|]
        source_diff = torch.abs(
            source_d.unsqueeze(1) - source_d.unsqueeze(0)
        )  # [batch_dim, batch_dim]
        source_term = torch.mean(source_diff)

        # Target self-interaction: E[|Y - Y'|]
        target_diff = torch.abs(
            target_d.unsqueeze(1) - target_d.unsqueeze(0)
        )  # [batch_2_dim, batch_2_dim]
        target_term = torch.mean(target_diff)

        # CRPS = cross - 0.5 * (source_self + target_self)
        crps_d = cross_term - 0.5 * (source_term + target_term)
        crps_dims.append(crps_d)

    return torch.stack(crps_dims)  # [D]


def calculate_pearsonr_per_sample(
    x: Float[Tensor | ndarray, "batch sequence"],
    shift: int,
) -> Float[Tensor | ndarray, "batch"]:
    """Calculate Pearson correlation coefficient for each sample in the batch.

    Args:
        x (Tensor|ndarray): a batch of sequence, shape (batch, sequence)
        shift (int): shift value to apply to the target distribution

    Returns:
        Tensor|ndarray: Pearson correlation coefficients for each sample, shape (batch,)
    """
    # type conversion
    if isinstance(x, np.ndarray):
        x_np = x
    elif isinstance(x, Tensor):
        x_np = x.float().cpu().numpy()

    # shift check
    nrows = x_np.shape[1] - abs(shift)
    if nrows <= 0:
        raise ValueError(
            f"Shift value {shift} exceeds the length of the sequence {x_np.shape[1]}. "
        )

    # Prepare shifted sequences
    # note: the sign does not actually influence the correlation
    if shift > 0:
        x1_np = x_np[:, :-shift]  # original sequence truncated
        x2_np = x_np[:, shift:]  # shifted sequence
    elif shift < 0:
        x1_np = x_np[:, -shift:]  # original sequence truncated
        x2_np = x_np[:, :shift]  # shifted sequence
    else:
        x1_np = x_np
        x2_np = x_np

    # Calculate Pearson correlation for each sample
    correlations = []
    for i in range(x1_np.shape[0]):
        corr, _ = pearsonr(x1_np[i], x2_np[i])
        correlations.append(corr)

    correlations = np.array(correlations)

    # Convert back to original input type
    if isinstance(x, Tensor):
        return torch.from_numpy(correlations).float()
    else:
        return correlations


def _compute_pearsonr_chunk(args):
    """Helper function for multiprocessing in metric_pearsonr."""
    chunk_data, shift = args
    return calculate_pearsonr_per_sample(chunk_data, shift)


def metric_pearsonr(
    source: Tensor,
    target: Tensor,
    shift: int,
    num_workers: int = 1,
):
    """
    Calculate the Pearson r distribution in both source and target.

    Args:
        source (Tensor): source distribution, shape (batch, sequence)
        target (Tensor): target distribution, shape (batch, sequence)
        shift (int): shift value to apply to the target distribution
        num_workers (int): number of workers for parallel processing (default: 1)
    Returns:
        dict: Pearson r for each sample in source and target, keys are 'source' and 'target'

    """
    # Always concatenate first
    _all = (
        torch.cat((source.flatten(1), target.flatten(1)), dim=0).cpu().numpy()
    )  # [N, sequence]
    n_source = source.shape[0]

    if num_workers > 1:
        # Parallel processing using ProcessPoolExecutor with chunking
        # Split into chunks for multiprocessing
        chunks = np.array_split(_all, num_workers)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_compute_pearsonr_chunk, (chunk, shift))
                for chunk in chunks
            ]
            chunk_results = [future.result() for future in futures]

        # Concatenate results from all chunks
        correlations = np.concatenate(chunk_results)  # [N,]
    else:
        # Sequential processing
        correlations = calculate_pearsonr_per_sample(_all, shift)  # [N,]

    # Split back into source and target
    source_corrs = correlations[:n_source]
    target_corrs = correlations[n_source:]

    return {
        "source": torch.from_numpy(source_corrs).to(source.device),
        "target": torch.from_numpy(target_corrs).to(target.device),
    }


# Peak Load Error
def peak_load_error(
    source: Tensor, target: Tensor, error_type: Literal["mae", "crps"] = "crps"
) -> Float[Tensor, ""]:
    """Calculate the Peak Load Error

    calculate peak load error between each sample in the batched source and the
    target.

    Args:
        source (Tensor): source distribution, shape (batch, sequence)
        target (Tensor): target distribution, shape (sequence)
        error_type (Literal["mae", "crps"]): type of error to calculate

    Returns:
        Float[Tensor, ""]: Peak Load Error
    """
    # Find peak loads (maximum values) in source and target
    source_pos_peaks = torch.max(
        source, dim=1, keepdim=True
    ).values  # shape: (batch, 1)
    source_neg_peaks = torch.max(
        -source, dim=1, keepdim=True
    ).values  # shape: (batch, 1)
    target_pos_peak = torch.max(target).reshape(1, 1)  # shape: (1, 1)
    target_neg_peak = torch.max(-target).reshape(1, 1)  # shape: (1, 1)

    # Calculate absolute error between peak loads
    def mae(source, target):
        return torch.abs(source - target).mean()

    def crps(source, target):
        return crps_empirical_dimwise(
            source=source,
            target=target.squeeze(0),
        ).mean()

    error_func = mae if error_type == "mae" else crps

    pos_peak_errors = error_func(source_pos_peaks, target_pos_peak)
    neg_peak_errors = error_func(source_neg_peaks, target_neg_peak)

    # Return mean peak load error across all samples
    return pos_peak_errors + neg_peak_errors


# peak quantile error
def sym_quantile_error(
    source: Tensor,
    target: Tensor,
    quantile: float,
    error_type: Literal["mae", "crps"] = "crps",
) -> Float[Tensor, ""]:
    """Calculate the Symmetrical Quantile Error

    calculate the error between the specified quantile points of source and
    target. will always also calculate the symmetrical quantiles. (e.g. 0.99 and 0.01)

    Args:
        source (Tensor): source distribution, shape (batch, sequence)
        target (Tensor): target distribution, shape (sequence)
        quantile (float): quantile to calculate (1.0 = max, 0.0 = min)
        error_type (Literal["mae", "crps"]): type of error to calculate

    Returns:
        Float[Tensor, ""]: symmetrical quantile error
    """
    # Find peak loads (maximum values) in source and target
    source_pos_quantile = torch.quantile(source, quantile, dim=1, keepdim=True)
    # shape: (batch, 1)
    source_neg_quantile = torch.quantile(source, 1 - quantile, dim=1, keepdim=True)
    # shape: (batch, 1)
    target_pos_quantile = torch.quantile(target, quantile, dim=0, keepdim=True)
    # shape: (1,)
    target_neg_quantile = torch.quantile(target, 1 - quantile, dim=0, keepdim=True)
    # shape: (1,)

    # Calculate absolute error between peak loads
    def mae(source, target):
        return torch.abs(source - target).mean()

    def crps(source, target):
        return crps_empirical_dimwise(
            source=source,
            target=target.squeeze(0),
        ).mean()

    error_func = mae if error_type == "mae" else crps

    pos_quantile_error = error_func(source_pos_quantile, target_pos_quantile)
    neg_quantile_error = error_func(source_neg_quantile, target_neg_quantile)

    # Return the sum
    return pos_quantile_error + neg_quantile_error


# Ramer Douglas Peucker (critical point) error
def critical_point_error(
    source: Tensor,
    target: Tensor,
    epsilon: float = 0.1,
) -> Float[Tensor, ""]:
    """Calculate the Critical Point Error

    Args:
        source (Tensor): source distribution, shape (batch, sequence)
        target (Tensor): target distribution, shape (sequence)
        epsilon (float): threshold for critical point detection. higher means stronger
        simplification. 0.05 detailed, 0.1 moderate, 0.3 only major features, 0.5 high
        simplification.

    Returns:
        Float[Tensor, ""]: Critical Point Error
    """

    def ramer_douglas_peucker(points: Tensor, epsilon: float) -> list[int]:
        """Ramer-Douglas-Peucker algorithm to find critical points."""
        if len(points) <= 2:
            return list(range(len(points)))

        # Find the point with the maximum distance from line formed by first and last points
        start, end = points[0], points[-1]
        max_dist = 0
        max_index = 0

        for i in range(1, len(points) - 1):
            # Calculate perpendicular distance from point to line
            point = points[i]
            # Distance from point to line (start, end)
            if torch.equal(start, end):
                dist = torch.norm(point - start)
            else:
                # Point-to-line distance formula for 2D points
                line_vec = end - start
                point_vec = point - start
                line_len = torch.norm(line_vec)
                if line_len == 0:
                    dist = torch.norm(point_vec)
                else:
                    # For 2D vectors, cross product magnitude is |a × b| = |a1*b2 - a2*b1|
                    cross = torch.abs(
                        line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
                    )
                    dist = cross / line_len

            if dist > max_dist:
                max_dist = dist
                max_index = i

        # If max distance is greater than epsilon, recursively simplify
        if max_dist > epsilon:
            # Recursively simplify the two segments
            left_indices = ramer_douglas_peucker(points[: max_index + 1], epsilon)
            right_indices = ramer_douglas_peucker(points[max_index:], epsilon)
            # Combine results (avoid duplicating the middle point)
            return left_indices + [max_index + idx for idx in right_indices[1:]]
        else:
            # If no point is far enough, just return start and end
            return [0, len(points) - 1]

    def extract_critical_points(sequence: Tensor, epsilon: float) -> Tensor:
        """Extract critical points from a sequence using RDP algorithm."""
        # Create points as (index, value) pairs for 2D RDP
        indices = torch.arange(
            len(sequence), dtype=torch.float32, device=sequence.device
        )
        points = torch.stack([indices, sequence], dim=1)  # shape: (sequence, 2)

        critical_indices = ramer_douglas_peucker(points, epsilon)
        return sequence[critical_indices]

    # Extract critical points from target
    target_critical = extract_critical_points(target, epsilon)

    # Calculate critical point errors for each source sample
    errors = []
    for i in range(source.shape[0]):
        source_critical = extract_critical_points(source[i], epsilon)

        # Calculate error between critical points
        # Use minimum length to avoid dimension mismatch
        min_len = min(len(source_critical), len(target_critical))
        if min_len > 0:
            error = torch.mean(
                torch.abs(source_critical[:min_len] - target_critical[:min_len])
            )
        else:
            error = torch.tensor(0.0, device=source.device)
        errors.append(error)

    # Return mean critical point error across all samples
    return torch.mean(torch.stack(errors))


def main(): ...


if __name__ == "__main__":
    main()
