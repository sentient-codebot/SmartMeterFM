"""
WPuQ Household Electricity Consumption (all households, monthly segmentation)

Dataset
-------
Source: WPuQ (Wärmepumpen-Umfeldquartier) dataset from Germany.
HDF5 path: ``{NO_PV,WITH_PV}/{SFH*}/HOUSEHOLD/table`` in ``{year}_data_1min.hdf5``.
Houses: ~38 single-family homes (SFH), from both NO_PV and WITH_PV groups.
Field used: ``P_TOT`` (total active power, in **Watts**).

Units
~~~~~
All values are instantaneous active power in **W** (Watts), read directly
from the HDF5 source.  No unit conversion is applied during preprocessing.

Preprocessing (PreWPuQHousehold)
--------------------------------
1. Read 1-minute resolution household electricity data from all houses.
2. For each household, filter out NaN/Inf values, then interpolate ``P_TOT``
   onto a regular 1-minute grid per month using ``np.interp``.
3. Each household-month becomes one sample: shape ``[1, timesteps_in_month]``.
   Months with no valid data are skipped (e.g. 2018 months 1-4 are all NaN).
4. Shuffle and split 50/25/25 into train/val/test.
5. Save as ``wpuq_household_{year}_{train,val,test}.npz``.

Loading (WPuQHousehold)
-----------------------
- Loads the preprocessed NPZ files for years 2018-2020.
- Optional resolution downsampling via avg_pool1d (e.g. 1min -> 15min).
- Pads all monthly samples to a fixed length (31 days at target resolution).
  At 15min resolution: 31 * 96 = 2976 timesteps.
- Labels: ``month`` (0-11).
- Supports normalization (minmax or meanstd) and vectorization.

Example usage::

    from smartmeterfm.data_modules.wpuq_household import PreWPuQHousehold, WPuQHousehold
    from smartmeterfm.utils.configuration import DataConfig

    # Step 1: Preprocess (once)
    for year in [2018, 2019, 2020]:
        PreWPuQHousehold(root="data/wpuq/raw", year=year).load_process_save()

    # Step 2: Load dataset
    cfg = DataConfig(
        dataset="wpuq_household", root="data/wpuq", resolution="15min",
        load=False, normalize=True, normalize_method="minmax", pit=False,
        shuffle=False, vectorize=False, style_vectorize="chronological",
        vectorize_window_size=3, train_season="all", val_season="all",
        target_labels="month", segment_type="monthly",
    )
    data = WPuQHousehold(cfg)
    # data.dataset.profile["train"].shape -> [N, 2976, 1]
"""

import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

from .containers import DatasetWithMetadata
from .heat_pump import WPuQ
from .preprocessing import shuffle_array
from .readers import WPuQHouseholdReader
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
    original_dict_cond_dim = {"month": 1}

    def _validate_resolution(self, resolution: str):
        if resolution not in ["1min", "15min", "30min", "1h"]:
            raise ValueError(f"Invalid resolution for WPuQHousehold: {resolution!r}")

    def create_dataset(self) -> DatasetWithMetadata:
        return self._create_dataset_monthly()
