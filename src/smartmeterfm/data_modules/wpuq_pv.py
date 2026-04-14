"""
WPuQ PV Generation at Inverter (multiple households)
"""

import os
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from einops import rearrange

from .containers import DatasetWithMetadata
from .heat_pump import WPuQ
from .preprocessing import NAME_SEASONS, months_of_season, shuffle_array
from .readers import WPuQPVReader


PREPROCESS_RES = "1min"
DIRECTION_CODE = {
    "EAST": 0,
    "SOUTH": 1,
    "WEST": 2,
}
RES_SECOND = {
    "1min": 60,
    "15min": 15 * 60,
    "30min": 30 * 60,
    "1h": 60 * 60,
    "60min": 60 * 60,
}


def get_first_last_moment_of_month(
    year: int,
    month: int,
    res_second: int = 10,
    offset: timedelta = timedelta(seconds=0),
):
    first_day = datetime(year, month, 1, 0, 0, 0) + offset
    last_day = (
        datetime(year, month, 1, 0, 0, 0)
        + relativedelta(months=1)
        - timedelta(seconds=res_second)
        + offset
    )

    return first_day, last_day


def process_pv_dataset(list_dataset, year: int) -> dict[str, list[np.array]]:
    out = {str(month): [] for month in range(1, 13)}
    for full_dataset in list_dataset:  # full == all three directions
        # dataset: custom-dtype array (p * 365,) with many fields
        all_dirs = np.unique(full_dataset["DIRECTION"])
        for dir_u10 in all_dirs:
            dataset = full_dataset[full_dataset["DIRECTION"] == dir_u10]
            dir_int = DIRECTION_CODE[dir_u10]
            x = np.sort(dataset["index"])  # timestamps
            offset = datetime.fromtimestamp(x[0]) - datetime(year, 1, 1, 0, 0, 0)
            # res_second = 24*60*60 // (dataset.shape[0] // 365)
            res_second = RES_SECOND[PREPROCESS_RES]
            y = dataset["P_TOT"]  # power consumption
            # interpolate
            print(f"***{year} {dir_u10}***")
            print(f"start: {datetime.fromtimestamp(x[0])}")
            print(f"end: {datetime.fromtimestamp(x[-1])}")
            # xp = np.linspace(
            #     datetime.fromtimestamp(x[0]).timestamp(),
            #     (datetime.fromtimestamp(x[0])+relativedelta(years=1)-timedelta(seconds=res_second)).timestamp(),
            #     num=365*24*60*60//res_second,
            # )
            # yp = np.interp(xp, x, y)

            for month in range(1, 13):
                first, last = get_first_last_moment_of_month(
                    year, month, res_second, offset
                )
                xp = np.linspace(
                    int(first.timestamp()),
                    int(last.timestamp()),
                    int((last - first).total_seconds()) // res_second + 1,
                )
                ds = np.interp(xp, x, y)
                ds = rearrange(
                    ds,
                    "(days per_day) -> days per_day",
                    per_day=24 * 60 * 60 // res_second,
                )
                _dir = np.full((ds.shape[0], 1), dir_int).astype(np.float64)
                _dtype = np.dtype(
                    [
                        ("P_TOT", "float64", (ds.shape[1],)),
                        ("DIRECTION", "float64", (1,)),
                    ]
                )
                structured_array = np.empty(ds.shape[0], dtype=_dtype)
                structured_array["P_TOT"] = ds
                structured_array["DIRECTION"] = _dir

                out[str(month)].append(structured_array)

    return out


def process_pv_dataset_monthly(list_dataset, year: int) -> dict[str, list[np.array]]:
    """Like process_pv_dataset but keeps each month as a single sample (no daily split)."""
    out = {str(month): [] for month in range(1, 13)}
    for full_dataset in list_dataset:
        all_dirs = np.unique(full_dataset["DIRECTION"])
        for dir_u10 in all_dirs:
            dataset = full_dataset[full_dataset["DIRECTION"] == dir_u10]
            dir_int = DIRECTION_CODE[dir_u10]
            x = np.sort(dataset["index"])
            offset = datetime.fromtimestamp(x[0]) - datetime(year, 1, 1, 0, 0, 0)
            res_second = RES_SECOND[PREPROCESS_RES]
            y = dataset["P_TOT"]

            print(f"***{year} {dir_u10} (monthly)***")
            print(f"start: {datetime.fromtimestamp(x[0])}")
            print(f"end: {datetime.fromtimestamp(x[-1])}")

            for month in range(1, 13):
                first, last = get_first_last_moment_of_month(
                    year, month, res_second, offset
                )
                xp = np.linspace(
                    int(first.timestamp()),
                    int(last.timestamp()),
                    int((last - first).total_seconds()) // res_second + 1,
                )
                ds = np.interp(xp, x, y)
                # Keep as single row: shape [1, timesteps_in_month]
                ds = ds.reshape(1, -1)
                _dir = np.array([[dir_int]], dtype=np.float64)
                _dtype = np.dtype(
                    [
                        ("P_TOT", "float64", (ds.shape[1],)),
                        ("DIRECTION", "float64", (1,)),
                    ]
                )
                structured_array = np.empty(1, dtype=_dtype)
                structured_array["P_TOT"] = ds
                structured_array["DIRECTION"] = _dir

                out[str(month)].append(structured_array)

    return out


class PreWPuQPV:
    """
    __all_fields__ = [('index', '<i8'), ('S_1', '<f8'), ('S_2', '<f8'),
        ('S_3', '<f8'), ('S_TOT', '<f8'), ('I_1', '<f8'),
        ('I_2', '<f8'), ('I_3', '<f8'), ('PF_1', '<f8'),
        ('PF_2', '<f8'), ('PF_3', '<f8'), ('PF_TOT', '<f8'),
        ('P_1', '<f8'), ('P_2', '<f8'), ('P_3', '<f8'),
        ('P_TOT', '<f8'), ('Q_1', '<f8'), ('Q_2', '<f8'),
        ('Q_3', '<f8'), ('Q_TOT', '<f8'), ('U_1', '<f8'),
        ('U_2', '<f8'), ('U_3', '<f8')]

    _data_{res}.hdf5:
        - MISC
            - ES1
                - TRANSFORMER
                    - index
                    - P_TOT

    _data_spatial.hdf5:
        - SUBSTATION
            - {res}
                - -
                    - index
                    - P_TOT

    """

    hdf5_suffix = {
        "10s": "_data_10s.hdf5",
        "1min": "_data_1min.hdf5",
        "15min": "_data_15min.hdf5",
        "60min": "_data_60min.hdf5",
        "1h": "_data_60min.hdf5",
    }[PREPROCESS_RES]
    col_names = [
        "index",
        "PF_TOT",
        "P_TOT",
        "DIRECTION",
    ]

    def __init__(
        self,
        root: str = "data/wpuq/raw",
        year: int = 2018,
        segment_type: str = "daily",
    ):
        self.root = root
        self.year = year
        self.segment_type = segment_type
        self.reader = WPuQPVReader(
            os.path.join(root, f"{year}{self.hdf5_suffix}"), column_names=self.col_names
        )

    def load_process_save(self, num_process=1):
        final_dataset_by_month = {str(month): None for month in range(1, 13)}
        train_dataset_by_month, val_dataset_by_month, test_dataset_by_month = {}, {}, {}
        (
            train_num_sample_per_month,
            val_num_sample_per_month,
            test_num_sample_per_month,
        ) = {}, {}, {}

        with self.reader as reader:
            all_dataset = list(reader)
            num_dataset_per_process = len(all_dataset) // num_process
            list_dataset_per_process = [
                all_dataset[
                    i * num_dataset_per_process : (i + 1) * num_dataset_per_process
                ]
                for i in range(num_process)
            ]
            list_dataset_per_process[-1] = (
                list_dataset_per_process[-1]
                + all_dataset[num_dataset_per_process * num_process :]
            )
            if self.segment_type == "monthly":
                _proc_dataset = partial(process_pv_dataset_monthly, year=self.year)
            else:
                _proc_dataset = partial(process_pv_dataset, year=self.year)
            # pool = mp.Pool(num_process)
            list_dataset_per_process = list(
                map(_proc_dataset, list_dataset_per_process)
            )
            # list_dataset_per_process = [process_dataset(list_dataset_per_process[i], self.year) for i in range(num_process)]
            # pool.close()
            # pool.join()
            for month in range(1, 13):
                # TODO: check if the process results are non empty in each month. also afterwards only save those months that are nonempty
                collected = []
                for idx_process in range(num_process):
                    item = list_dataset_per_process[idx_process][str(month)]
                    if len(item) > 0:
                        collected.append(
                            np.concatenate(item, axis=0)
                        )  # collected: list[np.ndarray[day, seq_length]]
                if len(collected) == 0:
                    print(f"{self.year}-{month} is empty.")
                    continue
                final_dataset_by_month[str(month)] = shuffle_array(
                    np.concatenate(collected, axis=0)
                )
                train_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][
                    : int(len(final_dataset_by_month[str(month)]) * 0.5)
                ]
                val_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][
                    int(len(final_dataset_by_month[str(month)]) * 0.5) : int(
                        len(final_dataset_by_month[str(month)]) * 0.75
                    )
                ]
                test_dataset_by_month[str(month)] = final_dataset_by_month[str(month)][
                    int(len(final_dataset_by_month[str(month)]) * 0.75) :
                ]
                train_num_sample_per_month[str(month)] = len(
                    train_dataset_by_month[str(month)]
                )
                val_num_sample_per_month[str(month)] = len(
                    val_dataset_by_month[str(month)]
                )
                test_num_sample_per_month[str(month)] = len(
                    test_dataset_by_month[str(month)]
                )

        _prefix = "wpuq_pv_monthly" if self.segment_type == "monthly" else "wpuq_pv"
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
        return (
            train_num_sample_per_month,
            val_num_sample_per_month,
            test_num_sample_per_month,
        )


class WPuQPV(WPuQ):
    common_prefix = "wpuq_pv"
    base_res_second = 60  # base resolution = 60s
    original_dict_cond_dim = {"month": 1, "dir": 1}

    # Max monthly timesteps at 15min resolution: 31 days * 96 steps/day
    MAX_MONTHLY_TIMESTEPS_15MIN = 31 * (24 * 60 // 15)  # 2976

    def _get_file_prefix(self) -> str:
        if self.process_option.get("segment_type") == "monthly":
            return "wpuq_pv_monthly"
        return self.common_prefix

    def create_dataset(self) -> DatasetWithMetadata:
        segment_type = self.process_option.get("segment_type", "daily")
        if segment_type == "monthly":
            return self._create_dataset_monthly()
        return self._create_dataset_daily()

    def _create_dataset_monthly(self) -> DatasetWithMetadata:
        _prefix = self._get_file_prefix()
        _pool_kernel_size = self._get_pool_kernel_size()
        _max_timesteps = self._max_monthly_timesteps()

        # Load raw NPZ files (don't pre-concatenate across years to avoid
        # shape mismatch for months with different day counts, e.g. Feb leap year)
        raw_array = {}
        for task in self.record_tasks:
            raw_array[task] = []
            for year in self.record_years:
                raw_array[task].append(
                    np.load(
                        os.path.join(
                            self.raw_dir,
                            f"{_prefix}_{year}_{task}.npz",
                        )
                    )
                )

        num_sample_task = []
        all_profile = []
        all_label_dir = []
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
                        data["P_TOT"].astype(np.float32)
                    )  # [N, timesteps_in_month]
                    _dir = torch.from_numpy(
                        data["DIRECTION"].astype(np.float32)
                    )  # [N, 1]
                    _month_label = torch.ones(_profile.shape[0], dtype=torch.long) * (
                        month - 1
                    )
                    _year_label = torch.ones(_profile.shape[0], dtype=torch.long) * year

                    # clean
                    _profile, indices = self.clean_dataset(_profile)
                    _dir = _dir[indices]
                    _month_label = _month_label[indices]
                    _year_label = _year_label[indices]

                    # resolution adjustment per-month (before concatenation,
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
                    all_label_dir.append(_dir)
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
        all_label_dir = torch.cat(all_label_dir, dim=0)
        all_label = {
            "month": all_label_month,
            "year": all_label_year,
            "dir": all_label_dir,
        }

        # shared pipeline: normalize, vectorize, split, wrap
        return self._finalize_dataset(all_profile, all_label, num_sample_task)

    def _create_dataset_daily(self) -> DatasetWithMetadata:
        raw_array = {}
        for task in self.record_tasks:
            raw_array[task] = []
            for year in self.record_years:
                raw_array[task].append(
                    np.load(
                        os.path.join(
                            self.raw_dir,
                            self.common_prefix + "_" + str(year) + "_" + task + ".npz",
                        )
                    )
                )

        raw_array_collected = {}
        for task in self.record_tasks:
            raw_array_collected[task] = {}
            for month in range(1, 13):
                raw_array_collected[task][str(month)] = []
                for npz in raw_array[task]:
                    if str(month) in npz:
                        raw_array_collected[task][str(month)].append(npz[str(month)])
                if len(raw_array_collected[task][str(month)]) > 0:
                    raw_array_collected[task][str(month)] = np.concatenate(
                        raw_array_collected[task][str(month)], axis=0
                    )

        # before processing, put all tensors together
        num_sample_task = []
        all_profile = []
        all_label_dir = []
        all_label_month = []
        for task in self.record_tasks:
            _len = 0
            for season in NAME_SEASONS:
                season_months = months_of_season(season)
                profiles_per_month = [
                    raw_array_collected[task][str(m)]["P_TOT"] for m in season_months
                ]
                dirs_per_month = [
                    raw_array_collected[task][str(m)]["DIRECTION"]
                    for m in season_months
                ]
                _profile_to_append = torch.from_numpy(
                    np.concatenate(profiles_per_month, axis=0).astype(np.float32)
                )
                _dir_to_append = torch.from_numpy(
                    np.concatenate(dirs_per_month, axis=0).astype(np.float32)
                )
                # Build per-sample month labels matching each month's sample count
                _month_to_append = torch.cat(
                    [
                        torch.full((p.shape[0],), m - 1, dtype=torch.long)
                        for m, p in zip(season_months, profiles_per_month, strict=True)
                    ]
                )
                # clean profile
                _profile_to_append, indices = self.clean_dataset(_profile_to_append)
                # clean - filter labels
                _dir_to_append = _dir_to_append[indices]
                _month_to_append = _month_to_append[indices]
                # !! remove inf and nan
                all_profile.append(
                    _profile_to_append
                )  # shape: [num_sample, seq_length]
                all_label_dir.append(_dir_to_append)  # shape: [num_sample, 1]
                all_label_month.append(_month_to_append)  # shape: [num_sample, 1]
                _len += len(_profile_to_append)
            num_sample_task.append(_len)

        all_profile = torch.cat(all_profile, dim=0)  # shape: [num_sample, seq_length]
        all_profile = rearrange(
            all_profile, "n l -> n () l"
        )  # shape: [num_sample, 1, seq_length]
        all_label_month = torch.cat(all_label_month, dim=0)  # shape: [num_sample,]
        all_label_month = rearrange(
            all_label_month, "n -> n ()"
        )  # shape: [num_sample, 1]
        all_label_dir = torch.cat(all_label_dir, dim=0)  # shape: [num_sample, 1]
        all_label = {
            "month": all_label_month,
            "dir": all_label_dir,
        }  # for all labels, have batch x feature_dim shape

        # resolution adjustment
        all_profile = self._apply_resolution_adjustment(all_profile)

        # shared pipeline: normalize, vectorize, split, wrap
        return self._finalize_dataset(all_profile, all_label, num_sample_task)
