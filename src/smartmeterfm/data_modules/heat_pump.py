import multiprocessing as mp
import os
from datetime import datetime, timedelta
from functools import partial
from typing import Annotated

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from smartmeterfm.data_modules.utils import (
    DatasetWithMetadata,
    StaticLabelContainer,
    TimeSeriesDataCollection,
)

from ..utils.configuration import DataConfig
from .utils import WPuQReader


def get_first_last_second_of_month(year: int, month: int):
    first_day = datetime(year, month, 1, 0, 0, 0)
    last_day = datetime(year + month // 12, month % 12 + 1, 1, 0, 0, 0) - timedelta(
        seconds=1
    )

    return first_day, last_day


def split_to_day(dataset: np.ndarray, first_day: datetime, last_day: datetime):
    startend_days = []
    start_day = first_day  # first second of a day
    while start_day < last_day:  # until the day before last day
        end_day = (
            start_day + timedelta(days=1) - timedelta(seconds=10)
        )  # last 10 seconds of a day
        startend_days.append((start_day, end_day))
        assert (end_day - start_day).total_seconds() == 86400 - 10
        start_day = end_day + timedelta(seconds=10)  # next day

    dataset_per_day = []
    for start, end in startend_days:
        datapoint = dataset[
            (dataset["index"] >= start.timestamp())
            & (dataset["index"] <= end.timestamp())
        ]
        if (
            len(datapoint) == (end - start).total_seconds() // 10 + 1
        ):  # should be const = 8640
            # assert len(datapoint) == 8640 # theoretically can be deleted.
            data = datapoint["P_TOT"]
            # checks inf and nan
            if np.isinf(data).any() or np.isnan(data).any():
                continue
            dataset_per_day.append(data)

    if len(dataset_per_day) == 0:
        return np.array([])
    return np.stack(dataset_per_day, axis=0)  # shape: [num_day, seq_length]


def process_dataset(list_dataset, year: int) -> dict[str, list[np.ndarray]]:
    dataset_per_month = {str(month): [] for month in range(1, 13)}
    if len(list_dataset) == 0:
        return dataset_per_month
    for dataset in tqdm(
        list_dataset, desc="processing households", total=len(list_dataset)
    ):
        dataset = np.sort(dataset, order="index")
        for month in range(1, 13):
            first_day, last_day = get_first_last_second_of_month(year, month)
            ds = dataset[
                (dataset["index"] >= first_day.timestamp())
                & (dataset["index"] <= last_day.timestamp())
            ]
            ds = split_to_day(ds, first_day, last_day)  # shape: [num_day, seq_length]
            if len(ds) > 0:
                dataset_per_month[str(month)].append(ds)

    return dataset_per_month


def shuffle_array(array: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng()
    indices = rng.permutation(len(array))
    return array[indices]


class PreHeatPump:
    """
    Preprocess Germany Heat Pump data. Load from .hdf5 in 'raw' dir
    but also saves to 'raw' dir since this is preprocessing.
    The actuall processed folder is reserved for after normalization, vectorization, etc

    objective of pre-precessing:
        a single year (e.g. 2018) -> 2018_{month}_{resolution}_{task}.pt
        (12 x 3 x 3 = 72 files)

    train val test ratio: 0.5 0.25 0.25
    """

    hdf5_suffix = "_data_10s.hdf5"
    col_names = [
        "index",  # timestamp
        "PF_TOT",  # power factor total
        "P_TOT",  # active power total
    ]

    def __init__(
        self,
        root: str = "data/",
        year: int = 2018,
    ):
        self.root = root
        self.year = year
        self.reader = WPuQReader(
            os.path.join(self.root, str(self.year) + self.hdf5_suffix), self.col_names
        )

    def load_process_save(self, num_process=1):
        final_dataset_by_month = {str(month): None for month in range(1, 13)}
        train_dataset_by_month, val_dataset_by_month, test_dataset_by_month = {}, {}, {}
        (
            train_num_sample_per_month,
            val_num_sample_per_month,
            test_num_sample_per_month,
        ) = ({}, {}, {})

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
            _proc_dataset = partial(process_dataset, year=self.year)
            pool = mp.Pool(num_process)
            list_dataset_per_process = pool.map(_proc_dataset, list_dataset_per_process)
            pool.close()
            pool.join()
            for month in range(1, 13):
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
                    np.concatenate(collected, axis=0).astype(np.float32)
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

        np.savez_compressed(
            os.path.join(self.root, "wpuq_" + str(self.year) + "_train.npz"),
            **train_dataset_by_month,
        )
        np.savez_compressed(
            os.path.join(self.root, "wpuq_" + str(self.year) + "_val.npz"),
            **val_dataset_by_month,
        )
        np.savez_compressed(
            os.path.join(self.root, "wpuq_" + str(self.year) + "_test.npz"),
            **test_dataset_by_month,
        )

        print("complete.")
        return (
            train_num_sample_per_month,
            val_num_sample_per_month,
            test_num_sample_per_month,
        )


def reshape_dataset(
    dataset: np.ndarray, n_days: int
) -> Annotated[np.ndarray, "day, seq_length"]:
    "divide a long time sequence dataset into sequences per day"
    dataset = dataset.reshape(n_days, -1)  # shape: [day, seuqnce]
    return dataset


def season_indices(season: str) -> Annotated[np.ndarray, "day in a year"]:
    "return the day indices of each season in a year,"
    "starting from the last two months of winter. can acceptble >365 days."
    if season == "winter":
        days_foo = np.arange(0, 61)
        days_bar = np.arange(365 - 30, 365)
        days = np.concatenate([days_foo, days_bar])
    elif season == "spring":
        days = np.arange(61, 151)
    elif season == "summer":
        days = np.arange(151, 243)
    elif season == "autumn":
        days = np.arange(243, 335)
    else:
        raise ValueError("Invalid")

    return days


class WPuQ(TimeSeriesDataCollection):
    """
    A class for loading and preprocessing heat pump data.

    Args:
        path (str): The path to the HDF5 file containing the data.
        normalize (bool): Whether to normalize the data.
        pit_transform (bool): Whether to PIT transform the data.
        vectorize (bool): Whether to vectorize the data.
        vectorize_style (str): The vectorization style to use.
            Can be 'chronological' or 'stft'.
        vectorize_window_size (int): The window size to use for vectorization.

    Attributes:
        dict_season_tensor (dict): A dictionary
            containing the dataset for each season in tensors.
        scaling_factors (dict): A dictionary
            containing the scaling factors used for normalization.
        dict_day_tensor (dict): A dictionary
            containing the dataset for each day in tensors.
            'profile': The profile data.
            'condition': The condition data.
                'day': The day of the year.
                'season': The season of the year.
                'annual_consumption': The annual consumption of the year.

    Properties:
        num_channels (int): The number of channels in the data.
        sequence_length (int): The sequence length of the data.
        sample_shape (tuple): The shape of a single sample in the data.

    Methods:
        vectorize_transform(data: Tensor, style: str = 'chronological',
            window_size: int = 3) -> Tensor:
            Transforms (N, 1, 96) to (N, D, 96) by c
                oncatenating temporally adjacent data or by STFT.
        chrono_vectorize(data: Annotated[Tensor, "batch, seq_length"],
            window_size: int = 3) -> A
                nnotated[Tensor, "batch, window_size, seq_length"]:
            Vectorizes the data using the chronological method.
        stft_vectorize(data: Tensor, window_size: int = 5) -> Tensor:
            Vectorizes the data using the STFT method.
        normalize_fn(data: torch.Tensor) ->
            tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
            Normalizes the data.
        clean_dataset(dataset: np.ndarray) -> np.ndarray:
            Removes nan and inf from the dataset.

    """

    common_prefix = "wpuq"
    record_years = [2018, 2019, 2020]
    record_tasks = ["train", "val", "test"]
    base_res_second = 10  # base = 10s resolution
    dict_cond_dim = {
        "season": 1,
    }

    def __init__(
        self,
        data_config: DataConfig,
    ):
        super().__init__()
        self.root = data_config.root
        assert data_config.resolution in [
            "10s",
            "1min",
            "15min",
            "30min",
            "1h",
        ], "Invalid resolution"
        self.process_option = {
            "resolution": data_config.resolution,
            "normalize": data_config.normalize,
            "normalize_method": data_config.normalize_method,
            "pit_transform": data_config.pit,
            "shuffle": data_config.shuffle,
            "vectorize": data_config.vectorize,
            "style_vectorize": data_config.style_vectorize,
            "vectorize_window_size": data_config.vectorize_window_size,
        }
        hashed_option = self.hash_option(self.process_option)
        processed_filename = (
            self.common_prefix + f"_{hashed_option}.pt"
        )  # we saved all processed tasks in one file

        # A: if processed exists and load is True: load
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

    def create_dataset(self) -> DatasetWithMetadata:
        # B: if processed not exists or load is False: load, clean,
        # shuffle, vectorize, save
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
        all_tensor = []
        all_label_month = []
        for task in self.record_tasks:
            _len = 0
            for month in range(1, 13):
                _tensor_to_append = torch.from_numpy(
                    raw_array_collected[task][str(month)].astype(np.float32)
                )
                _tensor_to_append, *_ = self.clean_dataset(_tensor_to_append)
                _label_month_to_append = torch.ones(
                    _tensor_to_append.shape[0], dtype=torch.long
                ) * (month - 1)
                all_tensor.append(_tensor_to_append)  # shape: [num_sample, seq_length]
                all_label_month.append(_label_month_to_append)  # shape: [num_sample,]
                _len += _tensor_to_append.shape[0]
            num_sample_task.append(_len)

        all_tensor = torch.cat(all_tensor, dim=0)  # shape: [num_sample, seq_length]
        all_label_month = torch.cat(all_label_month, dim=0)  # shape: [num_sample,]
        all_tensor = rearrange(
            all_tensor, "n l -> n () l"
        )  # shape: [num_sample, 1, seq_length]
        all_label = {}
        all_label["month"] = rearrange(
            all_label_month, "n -> n ()"
        )  # for all labels, have batch x feature_dim shape

        # resolution adjustment
        resolution = self.process_option["resolution"]
        if resolution == "10s":
            pass
        else:
            if resolution == "1min":
                _pool_kernel_size = 60 // self.base_res_second
            elif resolution == "15min":
                _pool_kernel_size = 15 * 60 // self.base_res_second
            elif resolution == "30min":
                _pool_kernel_size = 30 * 60 // self.base_res_second
            elif resolution == "1h":
                _pool_kernel_size = 60 * 60 // self.base_res_second
            else:
                raise NotImplementedError
            all_tensor = torch.nn.functional.avg_pool1d(
                all_tensor, kernel_size=_pool_kernel_size, stride=_pool_kernel_size
            )

        # normalize
        scaling_factor = None
        if self.process_option["normalize"]:
            all_tensor = self._normalize_fn(all_tensor)
            scaling_factor = self.scaling_factor

        # pit
        pit = None
        assert not self.process_option["pit_transform"], "Not implemented. Deprecated.\
            Advised to use PIT in the PL model or PL data module."

        # shuffle
        pass  # already shuffled in pre-processing

        # vectorize
        if self.process_option["vectorize"]:
            all_tensor = self._vectorize_fn(
                all_tensor,
            )

        # split
        task_chunk = list(
            all_tensor.split(num_sample_task, dim=0)
        )  # shape: [num_sample, 1, seq_length]
        label_chunk = {}  # lable_chunk["label_name"]
        # = [_label_task0, _label_task1, ...]
        for _label_name, _label in all_label.items():
            label_chunk[_label_name] = list(
                _label.split(num_sample_task, dim=0)
            )  # shape: [num_sample, *]
        profile_task_chunk = {}
        label_task_chunk = {}
        for task in self.record_tasks:
            profile_task_chunk[task] = rearrange(task_chunk.pop(0), "n c l -> n l c")
            label_task_chunk[task] = StaticLabelContainer({})
            for _label_name, _label_list in label_chunk.items():
                # label_task_chunk[task][_label_name] = _label_list.pop(0)
                label_task_chunk[task] += StaticLabelContainer(
                    {_label_name: _label_list.pop(0)}
                )
            # data type will be processed later when used.

        dataset = DatasetWithMetadata(
            profile=profile_task_chunk,
            label=label_task_chunk,
            pit=pit,
            scaling_factor=scaling_factor,
        )
        return dataset

    def save_dataset(self, dataset: DatasetWithMetadata, processed_filename) -> None:
        os.makedirs(os.path.join(self.processed_dir), exist_ok=True)
        torch.save(dataset, os.path.join(self.processed_dir, processed_filename))
        print(f"Saved {processed_filename}")

    def load_dataset(self, processed_filename) -> DatasetWithMetadata:
        if os.path.exists(os.path.join(self.processed_dir, processed_filename)):
            try:
                loaded: DatasetWithMetadata = torch.load(
                    os.path.join(self.processed_dir, processed_filename),
                    map_location="cpu",
                    weights_only=False,
                )
                print("Processed data loaded.")
                return loaded
            except Exception as e:
                print(f"Error loading processed data. {e} Recreating...")
        return None


def main(): ...


if __name__ == "__main__":
    main()
