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
from .transforms import (
    ChronoVectorize,
    Compose,
    ConstantScaler,
    MeanStdScaler,
    MinMaxScaler,
    Patchify,
)


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
            "scaling_factor": data_config.scaling_factor,
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
            self._restore_transforms_from_dataset(self.dataset)
        else:
            print("Process and save data.")
            self.dataset = self.create_dataset()
            self.save_dataset(self.dataset, processed_filename)

        print("Dataset ready.")

    def _restore_transforms_from_dataset(self, dataset: DatasetWithMetadata) -> None:
        """Rehydrate ``_scaler`` and ``_vectorizer`` after loading a cached dataset.

        ``_normalize_fn`` / ``_vectorize_fn`` only run on the create_dataset path,
        so loading a cached ``.pt`` otherwise leaves ``profile_inverse_transform``
        as a no-op.
        """
        if self.process_option["normalize"] and dataset.scaling_factor is not None:
            method = self.process_option["normalize_method"]
            sf = dataset.scaling_factor
            if method == "minmax":
                self._scaler = MinMaxScaler(min_val=sf[1], max_val=sf[0])
            elif method == "meanstd":
                self._scaler = MeanStdScaler(mean_val=sf[0], std_val=sf[1])
            elif method == "constant":
                self._scaler = ConstantScaler(scale_val=sf[0])
            else:
                raise ValueError(f"Invalid normalize method: {method}")
            self.scaling_factor = sf

        if self.process_option["vectorize"]:
            style = self.process_option["style_vectorize"]
            window_size = self.process_option["vectorize_window_size"]
            if style in ["chronological", "chrono"]:
                self._vectorizer = ChronoVectorize(window_size)
            elif style == "patchify":
                self._vectorizer = Patchify(window_size)
            elif style == "stft":
                raise NotImplementedError("STFT vectorize not implemented")
            else:
                raise ValueError(f"Invalid vectorize style: {style}")

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

    def save_dataset(
        self, dataset: DatasetWithMetadata, processed_filename: str
    ) -> None:
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
        elif method == "constant":
            # Use config-provided scaling_factor if available, else auto from max
            cfg_sf = self.process_option.get("scaling_factor", None)
            scale_val = cfg_sf[0] if cfg_sf else None
            scaler = ConstantScaler(scale_val=scale_val)
            scaler.fit(data)
            self.scaling_factor = (scaler.scale_val,)
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
        resolution-adjust per-month, pad to 31 days, build month labels.

        The NPZ schema convention is:

            ``"<m>"``           — data array for calendar month *m* (1-12)
            ``"<m>_<name>"``    — optional per-row label array aligned with
                                  ``"<m>"`` (e.g. ``"1_tariff_type"``).

        Per-row labels other than ``month`` / ``year`` are autodetected from
        the NPZ keys; when no ``"<m>_<name>"`` keys are present this method
        produces ``{"month", "year"}`` only — matching the historical
        behaviour bit-for-bit.
        """
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

        num_sample_task: list[int] = []
        all_profile: list[Tensor] = []
        # Per-row label accumulator: name -> list of [N] long tensors.
        # ``month`` and ``year`` are always derived; extra keys come from NPZ.
        all_labels: dict[str, list[Tensor]] = {"month": [], "year": []}

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

                    # Derived per-row labels.
                    chunk_labels: dict[str, Tensor] = {
                        "month": torch.full(
                            (_profile.shape[0],), month - 1, dtype=torch.long
                        ),
                        "year": torch.full(
                            (_profile.shape[0],), year, dtype=torch.long
                        ),
                    }

                    # NPZ-stored per-row labels (e.g. tariff_type, acorn_grouped).
                    prefix = f"{month}_"
                    for npz_key in npz.files:
                        if not npz_key.startswith(prefix):
                            continue
                        name = npz_key[len(prefix) :]
                        # Skip if the suffix is itself another month digit
                        # (would only happen if a label had a numeric name —
                        # not a real risk, but be defensive).
                        if name in chunk_labels:
                            continue
                        arr = npz[npz_key]
                        # Accept either [N] or [N, 1]; squeeze to [N].
                        if arr.ndim == 2 and arr.shape[1] == 1:
                            arr = arr[:, 0]
                        chunk_labels[name] = torch.from_numpy(arr.astype(np.int64))

                    # clean (drops NaN/Inf rows)
                    _profile, indices = self.clean_dataset(_profile)
                    for name, vec in list(chunk_labels.items()):
                        chunk_labels[name] = vec[indices]

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
                    for name, vec in chunk_labels.items():
                        all_labels.setdefault(name, []).append(vec)
                    _len += len(_profile)

            num_sample_task.append(_len)

        all_profile = torch.cat(all_profile, dim=0)  # [N, seq_length]
        all_profile = rearrange(all_profile, "n l -> n () l")  # [N, 1, seq_length]
        n_total = all_profile.shape[0]

        # Concatenate every accumulated label and reshape to [N, 1].  Drop
        # any label whose total length does not match the data length (this
        # only happens when an optional label is present in some year's NPZ
        # but missing from another's — in that case we can't safely align).
        all_label: dict[str, Tensor] = {}
        for name in sorted(all_labels):
            chunks = all_labels[name]
            if not chunks:
                continue
            cat = torch.cat(chunks, dim=0)
            if cat.shape[0] != n_total:
                # Surface the mismatch instead of silently emitting misaligned
                # labels.  Caller should regenerate all NPZs with the same
                # schema, or remove the optional label entirely.
                import warnings

                warnings.warn(
                    f"Label {name!r} length {cat.shape[0]} does not match "
                    f"data length {n_total}; dropping (likely partial-schema "
                    f"NPZs across record_years={self.record_years}).",
                    stacklevel=2,
                )
                continue
            all_label[name] = rearrange(cat, "n -> n ()")

        # shared pipeline: normalize, vectorize, split, wrap
        return self._finalize_dataset(all_profile, all_label, num_sample_task)

    # --- Properties ---

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")
