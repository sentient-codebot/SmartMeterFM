"""
Low Carbon London (LCL) Smart Meter Electricity Dataset — monthly segmentation

Dataset
-------
Source: UK Power Networks, Low Carbon London trial.
Available from Zenodo (CC-BY 4.0): https://zenodo.org/records/4656091
~5,500 households at half-hourly (30-min) resolution, Nov 2011 – Feb 2014.
CSV columns: ``LCLid``, ``DateTime``, ``KWH/hh`` (energy in kWh per half-hour).

Preprocessing (PreLCLElectricity)
---------------------------------
1. Read CSV files with pandas, filter to a single calendar year.
2. Group by (LCLid, month). Each complete household-month becomes one sample
   of shape ``[1, timesteps_in_month]`` (e.g. 31 × 48 = 1488 for January).
3. Incomplete household-months (missing half-hours) are skipped.
4. Shuffle and split 50/25/25 into train/val/test via ``split_and_save_npz``.
5. Save as ``lcl_electricity_{year}_{train,val,test}.npz``.

Loading (LCLElectricity)
------------------------
- Loads preprocessed NPZ files for years 2012–2013.
- Optional resolution downsampling via avg_pool1d (30min → 1h).
- Pads all monthly samples to a fixed length (31 days at target resolution).
  At 30min resolution: 31 × 48 = 1488 timesteps.
- Labels: ``month`` (0–11).
- Supports normalization (minmax or meanstd) and vectorization.

Example usage::

    from smartmeterfm.data_modules.lcl_electricity import PreLCLElectricity, LCLElectricity
    from smartmeterfm.utils.configuration import DataConfig

    # Step 1: Preprocess (once per year)
    for year in [2012, 2013]:
        PreLCLElectricity(root="data/lcl_electricity/raw", year=year).load_process_save()

    # Step 2: Load dataset
    cfg = DataConfig(
        dataset="lcl_electricity", root="data/lcl_electricity/", resolution="30min",
        load=False, normalize=True, normalize_method="meanstd", pit=False,
        shuffle=False, vectorize=True, style_vectorize="patchify",
        vectorize_window_size=16, train_season="all", val_season="all",
        target_labels="month", segment_type="monthly",
    )
    data = LCLElectricity(cfg)
    # data.dataset.profile["train"].shape -> [N, 93, 16]

Note:
    Only resolutions >= 30min are supported (no upsampling).
    At 1h resolution, max monthly timesteps = 744.  Use vectorize_window_size=8
    or 24 (744 is not divisible by 16).
"""

import calendar
import os

import numpy as np
from tqdm import tqdm

from .preprocessing import split_and_save_npz
from .wpuq_household import WPuQHousehold


RES_SECOND = 1800  # 30-minute resolution
STEPS_PER_DAY = 48


class PreLCLElectricity:
    """Preprocess LCL smart meter CSV data into monthly NPZ files.

    Reads half-hourly CSV files from the Zenodo cleaned dataset, groups by
    household and month, and saves train/val/test splits as compressed NPZ.

    train/val/test ratio: 0.5 / 0.25 / 0.25
    """

    def __init__(
        self,
        root: str = "data/lcl_electricity/raw",
        year: int = 2012,
    ):
        self.root = root
        self.year = year

    def load_process_save(self):
        import glob

        import pandas as pd

        csv_files = sorted(glob.glob(os.path.join(self.root, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.root}. "
                "Download the Zenodo cleaned dataset: "
                "https://zenodo.org/records/4656091"
            )

        # Read and concatenate all CSVs, filtering to target year
        frames = []
        for f in tqdm(csv_files, desc=f"reading CSVs for {self.year}"):
            df = pd.read_csv(f, parse_dates=["DateTime"])
            df = df[df["DateTime"].dt.year == self.year]
            if not df.empty:
                frames.append(df)

        if not frames:
            print(f"No data for year {self.year}.")
            return

        df = pd.concat(frames, ignore_index=True)
        df["month"] = df["DateTime"].dt.month

        # Convert KWH/hh to numeric (some cleaned datasets may have stray strings)
        df["KWH/hh"] = pd.to_numeric(df["KWH/hh"], errors="coerce")
        df = df.dropna(subset=["KWH/hh"])

        dataset_by_month: dict[str, np.ndarray | None] = {
            str(m): None for m in range(1, 13)
        }

        for month in tqdm(range(1, 13), desc="processing months"):
            month_df = df[df["month"] == month]
            if month_df.empty:
                continue

            days_in_month = calendar.monthrange(self.year, month)[1]
            expected_steps = days_in_month * STEPS_PER_DAY

            samples = []
            for _lcl_id, hh_df in month_df.groupby("LCLid"):
                hh_df = hh_df.sort_values("DateTime")
                values = hh_df["KWH/hh"].values.astype(np.float64)

                if len(values) != expected_steps:
                    continue
                if not np.isfinite(values).all():
                    continue

                samples.append(values.reshape(1, -1))

            if samples:
                dataset_by_month[str(month)] = np.concatenate(samples, axis=0).astype(
                    np.float32
                )
                print(
                    f"  {self.year}-{month:02d}: "
                    f"{len(samples)} household-months, "
                    f"{expected_steps} timesteps each"
                )

        split_and_save_npz(dataset_by_month, self.root, "lcl_electricity", self.year)
        print(f"Preprocessing complete for {self.year}.")


class LCLElectricity(WPuQHousehold):
    """Monthly LCL smart meter dataset with optional resolution downsampling.

    Inherits the monthly-segment loading pattern from WPuQHousehold.
    Only the class attributes differ (prefix, base resolution, years).

    Supported resolutions: "30min" (native), "1h".
    """

    common_prefix = "lcl_electricity"
    base_res_second = 1800  # 30-minute native resolution
    record_years = [2012, 2013]

    VALID_RESOLUTIONS = ("30min", "1h")

    def __init__(self, data_config):
        if data_config.resolution not in self.VALID_RESOLUTIONS:
            raise ValueError(
                f"LCL dataset only supports resolutions {self.VALID_RESOLUTIONS} "
                f"(native is 30min, upsampling not supported). "
                f"Got: {data_config.resolution!r}"
            )
        super().__init__(data_config)
