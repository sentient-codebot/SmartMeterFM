"""Base classes for time series dataset collections."""

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Annotated

import torch
from einops import rearrange
from torch import Tensor

from .containers import DatasetWithMetadata, StaticLabelContainer
from .transforms import ChronoVectorize, Compose, MeanStdScaler, MinMaxScaler, Patchify


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
    """Base class for time series dataset collections.

    Subclasses must set `base_res_second`, `record_tasks`, `common_prefix`,
    and implement `load_dataset`, `create_dataset`, `save_dataset`.
    """

    original_dict_cond_dim = {}  # to be overwritten by child class

    def __init__(self):
        self.all_dict_cond_dim = self.original_dict_cond_dim.copy()
        self.profile_transform: Compose | None = None

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

    # --- Normalize + vectorize (delegating to Transform classes) ---

    def _normalize_fn(self, data: torch.Tensor) -> Tensor:
        """Normalize the data. Stores scaling_factor and scaler on self."""
        method = self.process_option["normalize_method"]
        if method == "minmax":
            scaler = MinMaxScaler()
            scaler.fit(data)
            self.scaling_factor = (scaler.max_val, scaler.min_val)
        elif method == "meanstd":
            scaler = MeanStdScaler()
            scaler.fit(data)
            self.scaling_factor = (scaler.mean_val, scaler.std_val)
        else:
            raise ValueError(f"Invalid normalize method: {method}")
        self._scaler = scaler
        return scaler(data)

    def _vectorize_fn(
        self,
        data: Annotated[Tensor, "batch 1 seq"],
    ) -> Tensor:
        """Transform (N, 1, seq) to (N, D, seq') via patchify or chrono vectorize."""
        assert data.ndim in [2, 3]
        if data.ndim == 2:
            data = rearrange(data, "n L -> n 1 L")
        else:
            assert data.shape[1] == 1
        style = self.process_option["style_vectorize"]
        window_size = self.process_option["vectorize_window_size"]
        if style in ["chronological", "chrono"]:
            vec = ChronoVectorize(window_size)
        elif style == "patchify":
            vec = Patchify(window_size)
        elif style == "stft":
            raise NotImplementedError("STFT vectorize not implemented")
        else:
            raise ValueError(f"Invalid vectorize style: {style}")
        self._vectorizer = vec
        return vec(data)

    def profile_inverse_transform(self, profile: Tensor) -> Tensor:
        """Apply inverse transforms in reverse order (devectorize then denormalize)."""
        if hasattr(self, "_vectorizer"):
            profile = self._vectorizer.inverse(profile)
        if hasattr(self, "_scaler"):
            profile = self._scaler.inverse(profile)
        return profile

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
        scaling_factor = None
        if self.process_option["normalize"]:
            all_profile = self._normalize_fn(all_profile)
            scaling_factor = self.scaling_factor

        assert not self.process_option["pit_transform"], (
            "Not implemented. Deprecated. "
            "Advised to use PIT in the PL model or PL data module."
        )

        if self.process_option["vectorize"]:
            all_profile = self._vectorize_fn(all_profile)

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
            pit=None,
            scaling_factor=scaling_factor,
        )

    # --- Properties ---

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")
