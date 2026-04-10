"""
WPuQ Household Electricity Consumption (all households, monthly segmentation)
"""

import os
from datetime import datetime

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from .heat_pump import WPuQ, shuffle_array
from .utils import (
    DatasetWithMetadata,
    StaticLabelContainer,
    WPuQHouseholdReader,
)
from .wpuq_pv import get_first_last_moment_of_month


RES_SECOND = 60  # 1-minute resolution


def process_household_monthly(
    list_dataset: list[np.ndarray], year: int
) -> dict[str, list[np.ndarray]]:
    """Process household datasets into monthly samples.

    Each household-month becomes one sample with shape [1, timesteps_in_month].
    """
    out = {str(month): [] for month in range(1, 13)}
    for dataset in tqdm(
        list_dataset, desc="processing households", total=len(list_dataset)
    ):
        dataset = np.sort(dataset, order="index")
        x = dataset["index"].astype(np.float64)
        y = dataset["P_TOT"].astype(np.float64)

        # Filter out NaN/Inf values before interpolation
        valid_mask = np.isfinite(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) == 0:
            continue

        offset = datetime.fromtimestamp(x[0]) - datetime(year, 1, 1, 0, 0, 0)

        for month in range(1, 13):
            first, last = get_first_last_moment_of_month(
                year, month, RES_SECOND, offset
            )
            xp = np.linspace(
                int(first.timestamp()),
                int(last.timestamp()),
                int((last - first).total_seconds()) // RES_SECOND + 1,
            )

            # Skip month if no valid data falls within its range
            month_mask = (x_valid >= xp[0]) & (x_valid <= xp[-1])
            if month_mask.sum() == 0:
                continue

            ds = np.interp(xp, x_valid, y_valid)

            if np.isinf(ds).any() or np.isnan(ds).any():
                continue

            # single sample: [1, timesteps_in_month]
            out[str(month)].append(ds.reshape(1, -1))

    return out


class PreWPuQHousehold:
    """Preprocess WPuQ household electricity data into monthly NPZ files.

    Reads from {NO_PV,WITH_PV}/{SFH}/HOUSEHOLD/table in the 1min HDF5 files.
    Each sample is one household-month at 1-minute resolution.

    train/val/test ratio: 0.5 / 0.25 / 0.25
    """

    hdf5_suffix = "_data_1min.hdf5"
    col_names = [
        "index",
        "P_TOT",
    ]

    def __init__(
        self,
        root: str = "data/wpuq/raw",
        year: int = 2018,
    ):
        self.root = root
        self.year = year
        self.reader = WPuQHouseholdReader(
            os.path.join(self.root, str(self.year) + self.hdf5_suffix), self.col_names
        )

    def load_process_save(self, num_process=1):
        final_dataset_by_month = {str(month): None for month in range(1, 13)}
        train_dataset_by_month, val_dataset_by_month, test_dataset_by_month = {}, {}, {}

        with self.reader as reader:
            all_dataset = list(reader)
            dataset_per_month = process_household_monthly(all_dataset, self.year)

            for month in range(1, 13):
                items = dataset_per_month[str(month)]
                if len(items) == 0:
                    print(f"{self.year}-{month} is empty.")
                    continue
                final_dataset_by_month[str(month)] = shuffle_array(
                    np.concatenate(items, axis=0).astype(np.float32)
                )
                n = len(final_dataset_by_month[str(month)])
                train_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][
                    : int(n * 0.5)
                ]
                val_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][
                    int(n * 0.5) : int(n * 0.75)
                ]
                test_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][
                    int(n * 0.75) :
                ]

        _prefix = "wpuq_household"
        np.savez_compressed(
            os.path.join(self.root, f"{_prefix}_{self.year}_train.npz"),
            **train_dataset_by_month,
        )
        np.savez_compressed(
            os.path.join(self.root, f"{_prefix}_{self.year}_val.npz"),
            **val_dataset_by_month,
        )
        np.savez_compressed(
            os.path.join(self.root, f"{_prefix}_{self.year}_test.npz"),
            **test_dataset_by_month,
        )

        print("complete.")


class WPuQHousehold(WPuQ):
    """Monthly household electricity dataset with optional resolution downsampling."""

    common_prefix = "wpuq_household"
    base_res_second = 60  # preprocessed at 1-minute resolution

    def _get_pool_kernel_size(self) -> int | None:
        resolution = self.process_option["resolution"]
        if resolution == "1min":
            return None
        elif resolution == "15min":
            return 15 * 60 // self.base_res_second
        elif resolution == "30min":
            return 30 * 60 // self.base_res_second
        elif resolution == "1h":
            return 60 * 60 // self.base_res_second
        else:
            raise NotImplementedError(f"Unsupported resolution: {resolution}")

    def _max_monthly_timesteps(self) -> int:
        """Max monthly timesteps after resolution adjustment (31-day month)."""
        base_timesteps = 31 * 24 * 60 * 60 // self.base_res_second
        pool_k = self._get_pool_kernel_size()
        if pool_k is None:
            return base_timesteps
        return base_timesteps // pool_k

    def create_dataset(self) -> DatasetWithMetadata:
        _pool_kernel_size = self._get_pool_kernel_size()
        _max_timesteps = self._max_monthly_timesteps()

        # Load raw NPZ files
        raw_array = {}
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
        for task in self.record_tasks:
            _len = 0
            for month in range(1, 13):
                for npz in raw_array[task]:
                    if str(month) not in npz:
                        continue
                    data = npz[str(month)]
                    if data.shape[0] == 0:
                        continue

                    _profile = torch.from_numpy(
                        data.astype(np.float32)
                    )  # [N, timesteps_in_month]
                    _month_label = torch.ones(_profile.shape[0], dtype=torch.long) * (
                        month - 1
                    )

                    # clean
                    _profile, indices = self.clean_dataset(_profile)
                    _month_label = _month_label[indices]

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
                    _len += len(_profile)

            num_sample_task.append(_len)

        all_profile = torch.cat(all_profile, dim=0)  # [N, seq_length]
        all_profile = rearrange(all_profile, "n l -> n () l")  # [N, 1, seq_length]
        all_label_month = torch.cat(all_label_month, dim=0)
        all_label_month = rearrange(all_label_month, "n -> n ()")
        all_label = {"month": all_label_month}

        # normalize
        scaling_factor = None
        if self.process_option["normalize"]:
            all_profile = self._normalize_fn(all_profile)
            scaling_factor = self.scaling_factor

        # pit
        pit = None
        assert not self.process_option["pit_transform"], (
            "Not implemented. Deprecated. "
            "Advised to use PIT in the PL model or PL data module."
        )

        # vectorize
        if self.process_option["vectorize"]:
            all_profile = self._vectorize_fn(all_profile)

        # split
        task_chunk = list(all_profile.split(num_sample_task, dim=0))
        label_chunk = {}
        for _label_name, _label in all_label.items():
            label_chunk[_label_name] = list(_label.split(num_sample_task, dim=0))
        profile_task_chunk = {}
        label_task_chunk = {}
        for task in self.record_tasks:
            profile_task_chunk[task] = rearrange(task_chunk.pop(0), "n c l -> n l c")
            label_task_chunk[task] = StaticLabelContainer({})
            for _label_name, _label_list in label_chunk.items():
                label_task_chunk[task] += StaticLabelContainer(
                    {_label_name: _label_list.pop(0)}
                )

        dataset = DatasetWithMetadata(
            profile=profile_task_chunk,
            label=label_task_chunk,
            pit=pit,
            scaling_factor=scaling_factor,
        )
        return dataset
