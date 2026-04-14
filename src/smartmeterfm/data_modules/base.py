"""Base classes for time series dataset collections."""

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Annotated

import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from ..utils.configuration import DataConfig
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

    Subclasses must set `base_res_second`, `common_prefix`,
    and implement `create_dataset`.
    """

    original_dict_cond_dim = {}  # to be overwritten by child class
    record_tasks = ["train", "val", "test"]

    def __init__(self, data_config: DataConfig):
        self.all_dict_cond_dim = self.original_dict_cond_dim.copy()
        self.profile_transform: Compose | None = None
        self.root = data_config.root

        self._validate_resolution(data_config.resolution)

        self.process_option = {
            "resolution": data_config.resolution,
            "normalize": data_config.normalize,
            "normalize_method": data_config.normalize_method,
            "pit_transform": data_config.pit,
            "shuffle": data_config.shuffle,
            "vectorize": data_config.vectorize,
            "style_vectorize": data_config.style_vectorize,
            "vectorize_window_size": data_config.vectorize_window_size,
            "segment_type": data_config.segment_type,
            "target_labels": data_config.target_labels,
        }
        hashed_option = self.hash_option(self.process_option)
        processed_filename = self.common_prefix + f"_{hashed_option}.pt"

        self.dataset = None
        if data_config.load:
            self.dataset = self.load_dataset(processed_filename)

        if self.dataset is not None:
            print("All processed data loaded.")
        else:
            print("Process and save data.")
            self.dataset = self.create_dataset()
            self.save_dataset(self.dataset, processed_filename)

        print("Dataset ready.")

    def _validate_resolution(self, resolution: str):
        """Override in subclasses to restrict valid resolutions."""
        if resolution not in RESOLUTION_SECONDS:
            raise ValueError(f"Unknown resolution: {resolution!r}")

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

    def load_dataset(self, processed_filename: str) -> DatasetWithMetadata | None:
        path = os.path.join(self.processed_dir, processed_filename)
        if os.path.exists(path):
            try:
                loaded: DatasetWithMetadata = torch.load(
                    path, map_location="cpu", weights_only=False
                )
                print("Processed data loaded.")
                return loaded
            except Exception as e:
                print(f"Error loading processed data. {e} Recreating...")
        return None

    @abstractmethod
    def create_dataset(self) -> DatasetWithMetadata:
        raise NotImplementedError

    def save_dataset(self, dataset: DatasetWithMetadata, processed_filename: str) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(dataset, os.path.join(self.processed_dir, processed_filename))
        print(f"Saved {processed_filename}")

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

    # --- Monthly loading helper ---

    def _create_dataset_monthly(self) -> DatasetWithMetadata:
        """Shared monthly loading: load NPZs, loop task/month/npz, clean,
        resolution-adjust per-month, pad to 31 days, build month labels."""
        _pool_kernel_size = self._get_pool_kernel_size()
        _max_timesteps = self._max_monthly_timesteps()

        # Load raw NPZ files
        raw_array: dict[str, list] = {}
        for task in self.record_tasks:
            raw_array[task] = []
            for year in self.record_years:
                raw_array[task].append(
                    np.load(
                        os.path.join(
                            self.raw_dir,
                            f"{self.common_prefix}_{year}_{task}.npz",
                        )
                    )
                )

        num_sample_task = []
        all_profile = []
        all_label_month = []
        all_label_year = []
        for task in self.record_tasks:
            _len = 0
            for month in range(1, 13):
                for year, npz in zip(self.record_years, raw_array[task]):
                    if str(month) not in npz:
                        continue
                    data = npz[str(month)]
                    if data.shape[0] == 0:
                        continue

                    _profile = torch.from_numpy(
                        data.astype(np.float32)
                    )  # [N, timesteps_in_month]
                    _month_label = torch.ones(
                        _profile.shape[0], dtype=torch.long
                    ) * (month - 1)
                    _year_label = torch.ones(
                        _profile.shape[0], dtype=torch.long
                    ) * year

                    # clean
                    _profile, indices = self.clean_dataset(_profile)
                    _month_label = _month_label[indices]
                    _year_label = _year_label[indices]

                    # resolution adjustment per-month (before concat,
                    # since different months have different lengths)
                    _profile = rearrange(_profile, "n l -> n () l")
                    if _pool_kernel_size is not None:
                        _profile = torch.nn.functional.avg_pool1d(
                            _profile,
                            kernel_size=_pool_kernel_size,
                            stride=_pool_kernel_size,
                        )

                    # pad to max monthly length (31 days)
                    current_length = _profile.shape[-1]
                    if current_length < _max_timesteps:
                        pad_size = _max_timesteps - current_length
                        _profile = torch.nn.functional.pad(
                            _profile, (0, pad_size), mode="constant", value=0.0
                        )

                    _profile = rearrange(_profile, "n () l -> n l")

                    all_profile.append(_profile)
                    all_label_month.append(_month_label)
                    all_label_year.append(_year_label)
                    _len += len(_profile)

            num_sample_task.append(_len)

        all_profile = torch.cat(all_profile, dim=0)  # [N, seq_length]
        all_profile = rearrange(all_profile, "n l -> n () l")  # [N, 1, seq_length]
        all_label_month = torch.cat(all_label_month, dim=0)
        all_label_month = rearrange(all_label_month, "n -> n ()")
        all_label_year = torch.cat(all_label_year, dim=0)
        all_label_year = rearrange(all_label_year, "n -> n ()")
        all_label = {"month": all_label_month, "year": all_label_year}

        # shared pipeline: normalize, vectorize, split, wrap
        return self._finalize_dataset(all_profile, all_label, num_sample_task)

    # --- Properties ---

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")
