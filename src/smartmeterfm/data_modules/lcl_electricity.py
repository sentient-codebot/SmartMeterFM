"""
Low Carbon London (LCL) Smart Meter Electricity Dataset — monthly segmentation

Dataset
-------
Source: UK Power Networks, Low Carbon London trial.
Available from Zenodo (CC-BY 4.0): https://zenodo.org/records/4656091
~5,560 households at half-hourly (30-min) resolution, Nov 2011 – Feb 2014.
File format: ``.tsf`` (Monash Time Series Forecasting format).
Each line: ``series_name:start_timestamp:val1,val2,...`` — one contiguous
half-hourly sequence per household. Raw values are energy in **kWh per
half-hour**.

Units
~~~~~
The raw ``.tsf`` values are energy (kWh per half-hour interval). During
preprocessing, they are converted to average power in **kW** by dividing by
the interval duration in hours (0.5 h):

    power_kW = energy_kWh / 0.5 h = energy_kWh × 2

All preprocessed NPZ files and downstream data therefore have units of **kW**.
This is consistent with the WPuQ datasets, which also store power values
(albeit in W rather than kW).

Preprocessing (PreLCLElectricity)
---------------------------------
1. Parse ``.tsf`` file: extract per-household start timestamps and value arrays.
2. For each household, compute a half-hourly datetime index from its start
   timestamp and slice values by calendar month for the target year.
3. Convert from kWh/half-hour to kW (multiply by 2).
4. Each complete household-month becomes one sample ``[1, timesteps_in_month]``
   (e.g. 31 × 48 = 1488 for January).  Incomplete months are skipped.
5. Shuffle and split 50/25/25 into train/val/test via ``split_and_save_npz``.
6. Save as ``lcl_electricity_{year}_{train,val,test}.npz``.

Loading (LCLElectricity)
------------------------
- Loads preprocessed NPZ files for years 2012–2013.
- Values are average power in **kW** (converted from kWh/hh during preprocessing).
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
import glob
import os
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

from .base import TimeSeriesDataCollection
from .containers import DatasetWithMetadata
from .preprocessing import split_and_save_npz


RES_SECOND = 1800  # 30-minute resolution
STEPS_PER_DAY = 48


def _parse_tsf(filepath: str):
    """Parse a Monash .tsf file, yielding (series_name, start_dt, values) per household.

    Format::

        @relation ...
        @attribute series_name string
        @attribute start_timestamp date
        @frequency half_hourly
        @data
        T1:2012-10-13 00-00-01:0.263,0.269,...
    """
    in_data = False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "@data":
                in_data = True
                continue
            if not in_data:
                continue

            # Data line: name:timestamp:val1,val2,...
            parts = line.split(":")
            series_name = parts[0]
            # Timestamp uses hyphens for time: "2012-10-13 00-00-01"
            ts_str = parts[1].strip()
            start_dt = datetime.strptime(ts_str, "%Y-%m-%d %H-%M-%S")
            # Snap to nearest half-hour (LCL timestamps are 1s past the hour)
            start_dt = start_dt.replace(second=0, microsecond=0)
            values = np.fromstring(parts[2], sep=",", dtype=np.float64)
            yield series_name, start_dt, values


class PreLCLElectricity:
    """Preprocess LCL smart meter .tsf data into monthly NPZ files.

    Reads the Monash .tsf file from the Zenodo cleaned dataset, slices each
    household's contiguous half-hourly series by calendar month, converts
    from kWh/half-hour to average power in **kW** (×2), and saves
    train/val/test splits as compressed NPZ.

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
        tsf_files = sorted(glob.glob(os.path.join(self.root, "*.tsf")))
        if not tsf_files:
            raise FileNotFoundError(
                f"No .tsf files found in {self.root}. "
                "Download the Zenodo cleaned dataset: "
                "https://zenodo.org/records/4656091"
            )

        dataset_by_month: dict[str, list[np.ndarray]] = {
            str(m): [] for m in range(1, 13)
        }

        for tsf_path in tsf_files:
            print(f"Parsing {os.path.basename(tsf_path)} ...")
            for series_name, start_dt, values in tqdm(
                list(_parse_tsf(tsf_path)), desc="households"
            ):
                self._extract_monthly_samples(start_dt, values, dataset_by_month)

        # Concatenate and save
        final: dict[str, np.ndarray | None] = {}
        for month in range(1, 13):
            samples = dataset_by_month[str(month)]
            if samples:
                final[str(month)] = np.concatenate(samples, axis=0).astype(np.float32)
                days = calendar.monthrange(self.year, month)[1]
                print(
                    f"  {self.year}-{month:02d}: "
                    f"{len(samples)} household-months, "
                    f"{days * STEPS_PER_DAY} timesteps each"
                )
            else:
                final[str(month)] = None

        split_and_save_npz(final, self.root, "lcl_electricity", self.year)
        print(f"Preprocessing complete for {self.year}.")

    def _extract_monthly_samples(
        self,
        start_dt: datetime,
        values: np.ndarray,
        out: dict[str, list[np.ndarray]],
    ):
        """Slice one household's contiguous series into monthly samples for self.year."""
        n = len(values)
        step = timedelta(seconds=RES_SECOND)

        for month in range(1, 13):
            days_in_month = calendar.monthrange(self.year, month)[1]
            expected_steps = days_in_month * STEPS_PER_DAY

            month_start = datetime(self.year, month, 1)
            # Index of first reading in this month
            offset_seconds = (month_start - start_dt).total_seconds()
            idx_start = offset_seconds / RES_SECOND

            # Must be an exact half-hour boundary
            if idx_start != int(idx_start):
                continue
            idx_start = int(idx_start)
            idx_end = idx_start + expected_steps

            if idx_start < 0 or idx_end > n:
                continue

            segment = values[idx_start:idx_end]
            if not np.isfinite(segment).all():
                continue

            # Convert kWh/half-hour → kW: power = energy / time = kWh / 0.5h
            segment = segment * (3600 / RES_SECOND)  # = × 2 for 30-min intervals

            out[str(month)].append(segment.reshape(1, -1))


class LCLElectricity(TimeSeriesDataCollection):
    """Monthly LCL smart meter dataset with optional resolution downsampling.

    Values are average power in **kW** (converted from kWh/half-hour during
    preprocessing).  Supported resolutions: "30min" (native), "1h".
    """

    common_prefix = "lcl_electricity"
    base_res_second = 1800  # 30-minute native resolution
    record_years = [2012, 2013]
    original_dict_cond_dim = {"month": 1}

    VALID_RESOLUTIONS = ("30min", "1h")

    def _validate_resolution(self, resolution: str):
        if resolution not in self.VALID_RESOLUTIONS:
            raise ValueError(
                f"LCL dataset only supports resolutions {self.VALID_RESOLUTIONS} "
                f"(native is 30min, upsampling not supported). "
                f"Got: {resolution!r}"
            )

    def create_dataset(self) -> DatasetWithMetadata:
        return self._create_dataset_monthly()
