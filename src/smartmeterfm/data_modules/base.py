"""Base classes for time series dataset collections."""

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Annotated

import torch
from einops import rearrange
from torch import Tensor

from .containers import DatasetWithMetadata, DataTransform, StaticLabelContainer
from .preprocessing import NAME_SEASONS


RANDOM_STATE = 0
g = torch.Generator()
g.manual_seed(RANDOM_STATE)

RESOLUTION_SECONDS = {
    "10s": 10,
    "1min": 60,
    "15min": 15 * 60,
    "30min": 30 * 60,
    "1h": 60 * 60,
}


class DatasetCollection(ABC):
    dataset: DatasetWithMetadata | None = None


class TimeSeriesDataCollection(DatasetCollection):
    """
    Base class for time series dataset.
        encapsulates dataset and metadata.

    Contains common methods:
        - hash_option
        - normalize_fn
        - denormalize_fn
        - shuffle_dataset
        - vectorize_transform
    """

    original_dict_cond_dim = {}  # to be overwritten by child class

    def __init__(self):
        self.all_dict_cond_dim = self.original_dict_cond_dim.copy()
        self.registered_profile_transform = []
        self.registered_label_transform = []

    @property
    def dict_cond_dim(self):
        "The actually used condition dimension"
        if hasattr(self, "process_option"):
            if "target_labels" in self.process_option:
                return {
                    key: value
                    for key, value in self.all_dict_cond_dim.items()
                    if key in self.process_option["target_labels"]
                }
        return self.original_dict_cond_dim

    @abstractmethod
    def load_dataset(self) -> DatasetWithMetadata:
        raise NotImplementedError

    @abstractmethod
    def create_dataset(self) -> DatasetWithMetadata:
        raise NotImplementedError

    @abstractmethod
    def save_dataset(self, dataset: DatasetWithMetadata) -> None:
        raise NotImplementedError

    # --- Hashing ---

    @staticmethod
    def hash_option(option: dict) -> str:
        str_option = ",".join(f"{k}:{v}" for k, v in sorted(option.items()))
        return hashlib.md5(str_option.encode()).hexdigest()[:12]

    @staticmethod
    def hash_set_string(set_string: set[str]) -> str:
        str_concat = ",".join(sorted(set_string))
        return hashlib.md5(str_concat.encode()).hexdigest()[:12]

    # --- Resolution adjustment ---

    def _get_pool_kernel_size(self) -> int | None:
        """Compute avg_pool1d kernel size for resolution downsampling.

        Returns None if target resolution matches base_res_second (no pooling).
        """
        target = RESOLUTION_SECONDS[self.process_option["resolution"]]
        if target <= self.base_res_second:
            return None
        return target // self.base_res_second

    def _apply_resolution_adjustment(self, tensor: Tensor) -> Tensor:
        """Apply avg_pool1d to downsample from base resolution. Expects [N, 1, L]."""
        k = self._get_pool_kernel_size()
        if k is None:
            return tensor
        return torch.nn.functional.avg_pool1d(tensor, kernel_size=k, stride=k)

    def _max_monthly_timesteps(self) -> int:
        """Max monthly timesteps after resolution adjustment (31-day month)."""
        base_timesteps = 31 * 24 * 60 * 60 // self.base_res_second
        pool_k = self._get_pool_kernel_size()
        if pool_k is None:
            return base_timesteps
        return base_timesteps // pool_k

    # --- Transforms (legacy decorator-based, kept for backward compat) ---

    def _register_profile_transform(self, name, transform_fn, inverse_fn, extras=()):
        self.registered_profile_transform.append(
            DataTransform(
                name=name,
                transform=transform_fn,
                inverse_transform=inverse_fn,
                returned_args=extras,
            )
        )

    def profile_inverse_transform(self, profile: Tensor) -> Tensor:
        for trans in reversed(self.registered_profile_transform):
            returned_args = trans.returned_args
            profile = trans.inverse_transform(profile, *returned_args)
        return profile

    def label_inverse_transform(self, label: Tensor) -> Tensor:
        for trans in self.registered_label_transform.reverse():
            returned_args = trans.returned_args
            label = trans.inverse_transform(self, label, *returned_args)
        return label

    # --- Vectorization ---

    def _vectorize_fn(
        self,
        data: Annotated[Tensor, "batch 1 seq"],
    ) -> Tensor:
        """Transform (N, 1, seq) to (N, D, seq') via patchify or chrono vectorize."""
        assert data.ndim in [2, 3]
        if data.ndim == 3:
            assert data.shape[1] == 1
        else:
            data = rearrange(data, "n L -> n 1 L")
        style = self.process_option["style_vectorize"]
        window_size = self.process_option["vectorize_window_size"]
        if style in ["chronological", "chrono"]:
            data = self.chrono_vectorize(data, window_size)
        elif style == "stft":
            data = self.stft_vectorize(data, window_size)
        elif style == "patchify":
            data = self.patchify_vectorize(data, window_size)
        else:
            raise ValueError("Invalid style")

        # Register for inverse
        self._register_profile_transform(
            "vectorize_",
            self._vectorize_fn,
            self._vectorize_inv,
        )
        return data

    def _vectorize_inv(
        self,
        data: Annotated[Tensor, "batch window_size seq"],
        *args,
    ) -> Tensor:
        if data.ndim == 2:
            return data
        if data.shape[1] == 1:
            return data
        style = self.process_option["style_vectorize"]

        if style in ["chronological", "chrono"]:
            data = self.inverse_chrono_vectorize(data)
        elif style == "stft":
            data = self.inverse_stft_vectorize(data)
        elif style == "patchify":
            data = self.inverse_patchify(data)
        else:
            raise ValueError("Invalid style")
        return data

    @staticmethod
    def chrono_vectorize(data, window_size=3):
        seq_length = data.shape[-1]
        data_padded = torch.nn.functional.pad(
            data, ((window_size - 1) // 2,) * 2, mode="circular"
        )
        data_unfolded = data_padded.unfold(2, window_size, 1)
        data_vectorized = data_unfolded[:, :, :seq_length, :]
        data_vectorized = rearrange(data_vectorized, "b 1 s w -> b w s")
        return data_vectorized.clone()

    @staticmethod
    def inverse_chrono_vectorize(data):
        mid_channel = data.shape[1] // 2
        data = data[:, mid_channel : mid_channel + 1, :]
        return data

    @staticmethod
    def patchify_vectorize(data, window_size=3):
        assert data.shape[2] % window_size == 0
        return rearrange(data, "b 1 (L C) -> b C L", C=window_size)

    @staticmethod
    def inverse_patchify(data):
        return rearrange(data, "b c l -> b 1 (l c)")

    @staticmethod
    def stft_vectorize(data, window_size=5):
        raise NotImplementedError

    @staticmethod
    def inverse_stft_vectorize(data):
        raise NotImplementedError

    # --- Normalization ---

    def _normalize_fn(self, data: torch.Tensor) -> Tensor:
        """Normalize the data. Stores scaling_factor on self."""
        method = self.process_option["normalize_method"]
        if method == "minmax":
            data, (max_value, min_value) = self._minmax_normalize_fn(data)
            self.scaling_factor = (max_value, min_value)
            # Register for inverse
            self._register_profile_transform(
                "normalize_",
                self._normalize_fn,
                self._normalize_inv,
                extras=(max_value, min_value),
            )
            return data
        elif method == "meanstd":
            data, (mean_value, std_value) = self._meanstd_normalize_fn(data)
            self.scaling_factor = (mean_value, std_value)
            self._register_profile_transform(
                "normalize_",
                self._normalize_fn,
                self._normalize_inv,
                extras=(mean_value, std_value),
            )
            return data
        else:
            raise ValueError("Invalid method")

    def _normalize_inv(self, data, *scaling_factor, **kwargs):
        if len(scaling_factor) == 0:
            scaling_factor = self.scaling_factor
        method = self.process_option["normalize_method"]
        if method == "minmax":
            return self._minmax_denormalize_fn(data, *scaling_factor)
        elif method == "meanstd":
            return self._meanstd_denormalize_fn(data, *scaling_factor)
        else:
            raise ValueError("Invalid method")

    @staticmethod
    def _minmax_normalize_fn(data):
        max_value = 1.1 * torch.max(data)
        min_value = min(
            torch.tensor(0.0, dtype=data.dtype, device=data.device), torch.min(data)
        )
        data = (data - min_value) / (max_value - min_value) * 2.0 - 1.0
        return data, (max_value, min_value)

    @staticmethod
    def _meanstd_normalize_fn(data):
        mean_value = torch.mean(data, dim=0, keepdim=True)
        std_value = torch.std(data, dim=0, keepdim=True)
        data = (data - mean_value) / std_value
        return data, (mean_value, std_value)

    @staticmethod
    def _minmax_denormalize_fn(data, max_value, min_value):
        assert data.ndim in [2, 3]
        _to_tensor = lambda x: (
            torch.tensor(x, device=data.device)
            if not isinstance(x, torch.Tensor)
            else x
        )
        max_value, min_value = map(_to_tensor, (max_value, min_value))
        if max_value.ndim == 0:
            max_value = max_value.unsqueeze(0)
        if min_value.ndim == 0:
            min_value = min_value.unsqueeze(0)
        assert max_value.shape == min_value.shape
        if max_value.ndim == 1:
            max_value = rearrange(max_value, "l -> 1 1 l")
            min_value = rearrange(min_value, "l -> 1 1 l")
        elif max_value.ndim == 2:
            max_value = rearrange(max_value, "c l -> 1 c l")
            min_value = rearrange(min_value, "c l -> 1 c l")

        if data.ndim == 2:
            dim_data_in = 2
            data = rearrange(data, "s l -> s 1 l")
        else:
            dim_data_in = 3

        max_value, min_value = (x.to(data.device) for x in (max_value, min_value))

        mid_channel = data.shape[1] // 2
        data = data[:, mid_channel : mid_channel + 1, :]
        if dim_data_in == 2:
            return rearrange(data, "s 1 l -> s l")
        else:
            return data

    @staticmethod
    def _meanstd_denormalize_fn(data, mean_value, std_value):
        assert data.ndim in [2, 3]
        _to_tensor = lambda x: (
            torch.tensor(x, device=data.device)
            if not isinstance(x, torch.Tensor)
            else x
        )
        mean_value, std_value = (_to_tensor(v) for v in (mean_value, std_value))
        if mean_value.ndim == 0:
            mean_value = mean_value.unsqueeze(0)
        if std_value.ndim == 0:
            std_value = std_value.unsqueeze(0)
        assert mean_value.shape == std_value.shape
        if mean_value.ndim == 1:
            mean_value = rearrange(mean_value, "l -> 1 1 l")
            std_value = rearrange(std_value, "l -> 1 1 l")
        elif mean_value.ndim == 2:
            mean_value = rearrange(mean_value, "c l -> 1 c l")
            std_value = rearrange(std_value, "c l -> 1 c l")

        if data.ndim == 2:
            dim_data_in = 2
            data = rearrange(data, "s l -> s 1 l")
        else:
            dim_data_in = 3

        data = data * std_value + mean_value
        if dim_data_in == 2:
            return rearrange(data, "s 1 l -> s l")
        else:
            return data

    # --- Data cleaning ---

    @staticmethod
    def clean_dataset(
        dataset: Annotated[Tensor, "sample channel seq_length"],
    ) -> tuple[Tensor, Tensor]:
        """Remove nan and inf. dataset shape [day, channel, seq_length]"""
        is_dim3 = dataset.ndim == 3
        if not is_dim3:
            dataset = rearrange(dataset, "sample seq_length -> sample 1 seq_length")
        notnan = ~torch.isnan(dataset).any(dim=-1).any(dim=-1)
        notinf = ~torch.isinf(dataset).any(dim=-1).any(dim=-1)
        dataset = dataset[notnan & notinf, :, :]
        if not is_dim3:
            dataset = rearrange(dataset, "sample 1 seq_length -> sample seq_length")
        return dataset, notnan & notinf

    @staticmethod
    def shuffle_fn(dataset: torch.Tensor) -> torch.Tensor:
        indices = torch.randperm(dataset.shape[0])
        return dataset[indices], indices

    # --- Shared pipeline ---

    def _finalize_dataset(
        self,
        all_profile: Tensor,
        all_labels: dict[str, Tensor],
        num_sample_task: list[int],
    ) -> DatasetWithMetadata:
        """Shared post-processing: normalize, vectorize, split into train/val/test, wrap.

        Args:
            all_profile: concatenated profiles [N, 1, seq_length]
            all_labels: dict mapping label name -> tensor [N, feature_dim]
            num_sample_task: list of per-task sample counts (must sum to N)
        """
        # normalize
        scaling_factor = None
        if self.process_option["normalize"]:
            all_profile = self._normalize_fn(all_profile)
            scaling_factor = self.scaling_factor

        # pit (deprecated)
        pit = None
        assert not self.process_option["pit_transform"], (
            "Not implemented. Deprecated. "
            "Advised to use PIT in the PL model or PL data module."
        )

        # vectorize
        if self.process_option["vectorize"]:
            all_profile = self._vectorize_fn(all_profile)

        # split into per-task chunks
        task_chunk = list(all_profile.split(num_sample_task, dim=0))
        label_chunk = {
            name: list(tensor.split(num_sample_task, dim=0))
            for name, tensor in all_labels.items()
        }

        profile_task_chunk = {}
        label_task_chunk = {}
        for task in self.record_tasks:
            profile_task_chunk[task] = rearrange(task_chunk.pop(0), "n c l -> n l c")
            label_task_chunk[task] = StaticLabelContainer({})
            for label_name, label_list in label_chunk.items():
                label_task_chunk[task] += StaticLabelContainer(
                    {label_name: label_list.pop(0)}
                )

        return DatasetWithMetadata(
            profile=profile_task_chunk,
            label=label_task_chunk,
            pit=pit,
            scaling_factor=scaling_factor,
        )

    # --- Properties ---

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def num_channel(self):
        return self.dict_season_tensor[NAME_SEASONS[0]].shape[1]

    @property
    def sequence_length(self):
        return self.dict_season_tensor[NAME_SEASONS[0]].shape[2]

    @property
    def sample_shape(self):
        return (1, self.num_channel, self.sequence_length)

    def inverse_pit(self, x, season):
        if self.process_option["pit_transform"]:
            if self.process_option["vectorize"]:
                bs = x.shape[0]
                x = rearrange(x, "bs channel sequence -> (bs channel) sequence")
            x = self.pit_season[season].inverse_transform(x)
            if self.process_option["vectorize"]:
                x = rearrange(x, "(bs channel) sequence -> bs channel sequence", bs=bs)
        return x
