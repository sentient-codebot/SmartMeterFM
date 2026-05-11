"""
Low Carbon London (LCL) Smart Meter Electricity Dataset — monthly segmentation

Dataset
-------
Source: UK Power Networks, Low Carbon London trial.
~5,560 households at half-hourly (30-min) resolution, Nov 2011 – Feb 2014.

Two ingestion paths are supported:

* :class:`PreLCLElectricityCSV` (**preferred**) reads the original LCL
  release: ``halfhourly_dataset/block_*.csv.gz`` keyed on ``LCLid`` plus the
  per-household metadata in ``informations_households.csv``.  This is the
  only path that can attach ``tariff_type`` (Std/ToU) and ``acorn_grouped``
  conditions.  Available from
  https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/
  (or the Kaggle mirror ``jeanmidev/smart-meters-in-london``).
* :class:`PreLCLElectricity` (**deprecated**) reads the Monash Time Series
  Forecasting Repository ``.tsf`` file from Zenodo
  (https://zenodo.org/records/4656091).  The TSF anonymises households as
  ``T1, T2, ...`` with no link to ``LCLid``, so per-household metadata
  cannot be joined.  Kept for backward compatibility only.

Units
~~~~~
The raw values are energy (kWh per half-hour interval). During preprocessing,
they are converted to average power in **kW** by dividing by the interval
duration in hours (0.5 h):

    power_kW = energy_kWh / 0.5 h = energy_kWh × 2

All preprocessed NPZ files and downstream data therefore have units of **kW**.
This is consistent with the WPuQ datasets, which also store power values
(albeit in W rather than kW).

NPZ schema
~~~~~~~~~~
For each year/split, ``lcl_electricity_{year}_{train,val,test}.npz`` stores:

* Keys ``"1"`` … ``"12"`` — per-month data arrays ``[N, days_in_month*48]``
* Keys ``"<m>_tariff_type"``     — per-row label ``[N]`` (int64, 0=Std, 1=ToU)
* Keys ``"<m>_acorn_grouped"``   — per-row label ``[N]`` (int64, 0-3)

The label keys are only present in NPZs produced by
:class:`PreLCLElectricityCSV`; the legacy TSF path produces only the data
arrays.

Loading (LCLElectricity)
------------------------
- Loads preprocessed NPZ files for years 2012–2013.
- Values are average power in **kW**.
- Optional resolution downsampling via avg_pool1d (30min → 1h).
- Pads all monthly samples to a fixed length (31 days at target resolution).
  At 30min resolution: 31 × 48 = 1488 timesteps.
- Labels: always ``month``, ``year``; optional ``tariff_type``,
  ``acorn_grouped`` when present in the NPZ.
- Supports normalization (minmax or meanstd) and vectorization.

Example usage::

    from smartmeterfm.data_modules.lcl_electricity import (
        PreLCLElectricityCSV, LCLElectricity
    )
    from smartmeterfm.utils.configuration import DataConfig

    # Step 1: Preprocess (once per year)
    for year in [2012, 2013]:
        PreLCLElectricityCSV(
            root="data/lcl_electricity/raw", year=year
        ).load_process_save()

    # Step 2: Load dataset
    cfg = DataConfig(
        dataset="lcl_electricity", root="data/lcl_electricity/", resolution="30min",
        load=False, normalize=True, normalize_method="meanstd", pit=False,
        shuffle=False, vectorize=True, style_vectorize="patchify",
        vectorize_window_size=16, train_season="all", val_season="all",
        target_labels=["month", "year", "tariff_type", "acorn_grouped"],
        segment_type="monthly",
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
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from .base import TimeSeriesDataCollection
from .containers import DatasetWithMetadata
from .preprocessing import split_and_save_npz


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-source-of-truth encodings for per-household conditions.  These are
# referenced by ``PreLCLElectricityCSV`` and re-exported to keep downstream
# users (notebooks, eval scripts) aligned with the indices stored in the NPZ.
# ---------------------------------------------------------------------------

TARIFF_TO_IDX = {"Std": 0, "ToU": 1}

ACORN_GROUPED_TO_IDX = {
    "Affluent": 0,
    "Comfortable": 1,
    "Adversity": 2,
    "ACORN-U": 3,  # Unclassified
}


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
    """Preprocess LCL smart meter ``.tsf`` data into monthly NPZ files.

    .. deprecated::
        Prefer :class:`PreLCLElectricityCSV`, which reads the original LCL
        release keyed on ``LCLid`` and can therefore attach ``tariff_type``
        and ``acorn_grouped`` per-household labels.  The Monash ``.tsf``
        anonymises households as ``T1, T2, ...`` with no mapping back to
        ``LCLid``, so this preprocessor cannot produce the new conditions.

    Reads the Monash ``.tsf`` file from the Zenodo cleaned dataset, slices
    each household's contiguous half-hourly series by calendar month,
    converts from kWh/half-hour to average power in **kW** (×2), and saves
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


class PreLCLElectricityCSV:
    """Preprocess original LCL halfhourly CSVs into monthly NPZ files.

    Reads the canonical LCL release (London Datastore / Kaggle mirror):

    * ``halfhourly_dataset/block_*.csv.gz`` — columns ``LCLid, tstp,
      energy(kWh/hh)``.  ~112 blocks of compressed CSVs.
    * ``informations_households.csv`` — columns ``LCLid, stdorToU, Acorn,
      Acorn_grouped, file``.

    Each household-month becomes one sample ``[1, days_in_month*48]`` (kW),
    paired with two per-row labels: ``tariff_type`` (Std=0, ToU=1) and
    ``acorn_grouped`` (Affluent=0, Comfortable=1, Adversity=2,
    ACORN-U=3).  Samples that are missing any half-hour slot in the target
    month — frequent in the LCL CSVs — are dropped (the household-month
    must form a complete contiguous half-hour series).

    train/val/test ratio: 0.5 / 0.25 / 0.25 with a single shared
    permutation per (year, month) — see
    :func:`smartmeterfm.data_modules.preprocessing.split_and_save_npz`.

    Args:
        root: directory containing ``halfhourly_dataset/`` and
            ``informations_households.csv``.  NPZ outputs are written here.
        year: calendar year to extract.
    """

    def __init__(
        self,
        root: str = "data/lcl_electricity/raw",
        year: int = 2012,
    ):
        self.root = root
        self.year = year

    # -- helpers --

    def _load_household_meta(self) -> dict[str, tuple[int, int]]:
        """Parse ``informations_households.csv`` → ``LCLid -> (tariff_idx, acorn_idx)``.

        Unknown ``stdorToU`` values are dropped (with a warning).  Unknown
        ``Acorn_grouped`` values are mapped to ``ACORN-U`` (Unclassified).
        """
        info_path = os.path.join(self.root, "informations_households.csv")
        if not os.path.exists(info_path):
            raise FileNotFoundError(
                f"Missing {info_path}. Run scripts/data/download_lcl.sh "
                "or place the LCL release manually."
            )
        df = pd.read_csv(info_path)
        # The canonical column names are LCLid, stdorToU, Acorn, Acorn_grouped, file
        required = {"LCLid", "stdorToU", "Acorn_grouped"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"informations_households.csv missing columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        unknown_acorn: set[str] = set()
        meta: dict[str, tuple[int, int]] = {}
        for _, row in df.iterrows():
            lclid = str(row["LCLid"])
            tariff_str = str(row["stdorToU"]).strip()
            acorn_str = str(row["Acorn_grouped"]).strip()
            if tariff_str not in TARIFF_TO_IDX:
                # e.g. NaN or unexpected — skip household.
                continue
            if acorn_str not in ACORN_GROUPED_TO_IDX:
                unknown_acorn.add(acorn_str)
                acorn_idx = ACORN_GROUPED_TO_IDX["ACORN-U"]
            else:
                acorn_idx = ACORN_GROUPED_TO_IDX[acorn_str]
            meta[lclid] = (TARIFF_TO_IDX[tariff_str], acorn_idx)

        if unknown_acorn:
            logger.warning(
                "Unknown Acorn_grouped values mapped to Unclassified: %s",
                sorted(unknown_acorn),
            )
        logger.info("Loaded metadata for %d households from %s", len(meta), info_path)
        return meta

    def _block_csv_paths(self) -> list[str]:
        """Return sorted list of ``block_*.csv`` / ``block_*.csv.gz`` paths.

        Accepts both formats:

        * ``block_*.csv.gz`` — London Datastore (canonical) ZIP release.
        * ``block_*.csv``    — Kaggle mirror (jeanmidev/smart-meters-in-london)
          ships the blocks uncompressed.

        Tolerates both ``halfhourly_dataset/`` (canonical) and a flat layout
        where blocks sit directly in ``root``.
        """
        patterns = ("block_*.csv.gz", "block_*.csv")
        candidates: list[str] = []
        for pattern in patterns:
            candidates.extend(
                glob.glob(os.path.join(self.root, "halfhourly_dataset", pattern))
            )
        if not candidates:
            for pattern in patterns:
                candidates.extend(glob.glob(os.path.join(self.root, pattern)))
        if not candidates:
            raise FileNotFoundError(
                f"No block_*.csv[.gz] files under {self.root}. Run "
                "scripts/data/download_lcl.sh or place the LCL release manually."
            )
        return sorted(set(candidates))

    def _read_block(self, path: str) -> pd.DataFrame:
        """Read one block CSV, normalise columns, filter to ``self.year``.

        ``compression='infer'`` (the pandas default) handles both ``.csv``
        and ``.csv.gz`` based on the file extension.
        """
        df = pd.read_csv(path)
        # Normalise the energy column name (canonical: "energy(kWh/hh)").
        energy_cols = [
            c for c in df.columns if "energy" in c.lower() or c.lower() == "kwh"
        ]
        if not energy_cols:
            raise ValueError(
                f"Could not find energy column in {path}. Columns: {list(df.columns)}"
            )
        if energy_cols[0] != "energy":
            df = df.rename(columns={energy_cols[0]: "energy"})
        # tstp comes as string; sub-second drift is common — floor to 30 min.
        df["tstp"] = pd.to_datetime(df["tstp"], errors="coerce")
        df = df.dropna(subset=["tstp"])
        df["tstp"] = df["tstp"].dt.floor(f"{RES_SECOND}s")
        df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
        df = df[df["tstp"].dt.year == self.year]
        return df

    def _extract_monthly_samples_from_household(
        self,
        df_house: pd.DataFrame,
        tariff_idx: int,
        acorn_idx: int,
        out_data: dict[str, list[np.ndarray]],
        out_tariff: dict[str, list[int]],
        out_acorn: dict[str, list[int]],
        max_interp_slots: int = 4,
    ) -> None:
        """Slice one household's series into monthly samples for ``self.year``.

        Reindexes onto the canonical half-hour grid for each month so we can
        explicitly detect missing slots.  Months with **more than**
        ``max_interp_slots`` NaN cells are dropped; months with up to that
        many are linearly interpolated.  This recovers household-months from
        small upstream outages (notably the LCL ``2012-12-18 ~15:00`` Null
        outage that affects every household) without inventing data for
        genuine multi-day gaps.
        """
        if df_house.empty:
            return
        # Drop duplicate timestamps (same half-hour reported twice, rare).
        df_house = df_house.drop_duplicates(subset="tstp", keep="last")
        s = df_house.set_index("tstp")["energy"].sort_index()

        for month in range(1, 13):
            days = calendar.monthrange(self.year, month)[1]
            start = pd.Timestamp(self.year, month, 1)
            end = start + pd.Timedelta(days=days) - pd.Timedelta(seconds=RES_SECOND)
            idx = pd.date_range(start=start, end=end, freq=f"{RES_SECOND}s")
            month_series = s.reindex(idx)
            n_nan = int(month_series.isna().sum())
            if n_nan > max_interp_slots:
                continue
            if n_nan > 0:
                month_series = month_series.interpolate(
                    method="linear", limit_direction="both"
                )
                # Any leftover NaN (e.g. trailing edge with no neighbour) →
                # still skip rather than silently filling with garbage.
                if month_series.isna().any():
                    continue
            segment = month_series.to_numpy(dtype=np.float32)
            # kWh/hh -> kW
            segment = segment * (3600.0 / RES_SECOND)
            if not np.isfinite(segment).all():
                continue
            out_data[str(month)].append(segment.reshape(1, -1))
            out_tariff[str(month)].append(tariff_idx)
            out_acorn[str(month)].append(acorn_idx)

    # -- driver --

    def load_process_save(self) -> None:
        meta = self._load_household_meta()
        block_paths = self._block_csv_paths()

        dataset_by_month: dict[str, list[np.ndarray]] = {
            str(m): [] for m in range(1, 13)
        }
        tariff_by_month: dict[str, list[int]] = {str(m): [] for m in range(1, 13)}
        acorn_by_month: dict[str, list[int]] = {str(m): [] for m in range(1, 13)}

        skipped_unknown = 0
        for block_path in tqdm(block_paths, desc="blocks"):
            df = self._read_block(block_path)
            if df.empty:
                continue
            for lclid, df_house in df.groupby("LCLid", sort=False):
                lclid = str(lclid)
                if lclid not in meta:
                    skipped_unknown += 1
                    continue
                tariff_idx, acorn_idx = meta[lclid]
                self._extract_monthly_samples_from_household(
                    df_house,
                    tariff_idx,
                    acorn_idx,
                    dataset_by_month,
                    tariff_by_month,
                    acorn_by_month,
                )

        if skipped_unknown:
            logger.info(
                "Skipped %d households not present in informations_households.csv",
                skipped_unknown,
            )

        # Build final per-month arrays + label arrays.
        final_data: dict[str, np.ndarray | None] = {}
        final_labels: dict[str, dict[str, np.ndarray]] = {}
        for month in range(1, 13):
            samples = dataset_by_month[str(month)]
            if not samples:
                final_data[str(month)] = None
                continue
            final_data[str(month)] = np.concatenate(samples, axis=0).astype(np.float32)
            final_labels[str(month)] = {
                "tariff_type": np.asarray(tariff_by_month[str(month)], dtype=np.int64),
                "acorn_grouped": np.asarray(acorn_by_month[str(month)], dtype=np.int64),
            }
            days = calendar.monthrange(self.year, month)[1]
            print(
                f"  {self.year}-{month:02d}: "
                f"{len(samples)} household-months, "
                f"{days * STEPS_PER_DAY} timesteps each"
            )

        split_and_save_npz(
            final_data,
            self.root,
            "lcl_electricity",
            self.year,
            labels_by_month=final_labels,
        )
        print(f"Preprocessing complete for {self.year}.")


class LCLElectricity(TimeSeriesDataCollection):
    """Monthly LCL smart meter dataset with optional resolution downsampling.

    Values are average power in **kW** (converted from kWh/half-hour during
    preprocessing).  Supported resolutions: "30min" (native), "1h".

    Available conditions (filtered by ``DataConfig.target_labels``):

    * ``month`` (0-11), ``year``               — derived from NPZ structure
    * ``first_day_of_week`` (0-6),
      ``month_length`` (0-3)                   — derived in collate from
                                                 ``(year, month)``
    * ``tariff_type`` (0-1, Std/ToU),
      ``acorn_grouped`` (0-3)                  — per-row labels in NPZ
                                                 (only when produced by
                                                 :class:`PreLCLElectricityCSV`)
    """

    common_prefix = "lcl_electricity"
    base_res_second = 1800  # 30-minute native resolution
    record_years = [2012, 2013]
    original_dict_cond_dim = {
        "month": 1,
        "year": 1,
        "first_day_of_week": 1,
        "month_length": 1,
        "tariff_type": 1,
        "acorn_grouped": 1,
    }

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
