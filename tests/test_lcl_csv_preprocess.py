"""Tests for the LCL CSV preprocessing path and the new
``tariff_type`` / ``acorn_grouped`` conditions.

Covers:
- Encoding dicts for tariff_type and acorn_grouped (incl. ACORN-U)
- ``PreLCLElectricityCSV._load_household_meta`` against a synthetic
  ``informations_households.csv``
- End-to-end CSV → NPZ extraction with synthetic block CSV fixture
- NPZ schema (``"<m>"`` / ``"<m>_<label>"`` keys, aligned lengths)
- ``split_and_save_npz`` permutation sharing across data + labels
- ``LCLCondition`` field validation + ``to_tensor_dict`` round-trip
- ``LCLLabelEmbedder`` backward compat (state_dict keys unchanged when
  the new optional kwargs are omitted) and forward-shape check when they
  are enabled
"""

from __future__ import annotations

import calendar
import gzip
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from smartmeterfm.conditions import LCLCondition
from smartmeterfm.data_modules.lcl_electricity import (
    ACORN_GROUPED_TO_IDX,
    TARIFF_TO_IDX,
    PreLCLElectricityCSV,
)
from smartmeterfm.data_modules.preprocessing import split_and_save_npz
from smartmeterfm.models.embedders.lcl import LCLLabelEmbedder


# ---------------------------------------------------------------------------
# Helpers — build a tiny synthetic LCL release on disk
# ---------------------------------------------------------------------------


STEPS_PER_DAY = 48
RES_SECOND = 1800


def _write_info_csv(
    path: str,
    rows: list[tuple[str, str, str, str]],
) -> None:
    """Each row: (LCLid, stdorToU, Acorn, Acorn_grouped)."""
    df = pd.DataFrame(rows, columns=["LCLid", "stdorToU", "Acorn", "Acorn_grouped"])
    df["file"] = "block_0"
    df.to_csv(path, index=False)


def _write_block_csv_gz(
    path: str,
    rows: list[tuple[str, datetime, float]],
) -> None:
    """Each row: (LCLid, tstp, energy_kwh_per_hh).

    Writes a gzipped CSV with the canonical LCL column header
    ``LCLid,tstp,energy(kWh/hh)``.
    """
    with gzip.open(path, "wt") as f:
        f.write("LCLid,tstp,energy(kWh/hh)\n")
        for lclid, tstp, e in rows:
            f.write(f"{lclid},{tstp.isoformat(sep=' ')},{e}\n")


def _household_january_rows(
    lclid: str, year: int, energy_value: float
) -> list[tuple[str, datetime, float]]:
    """Build a complete contiguous half-hourly January for one household."""
    rows = []
    days = calendar.monthrange(year, 1)[1]
    n = days * STEPS_PER_DAY
    start = datetime(year, 1, 1)
    for i in range(n):
        rows.append((lclid, start + timedelta(seconds=RES_SECOND * i), energy_value))
    return rows


# ---------------------------------------------------------------------------
# Encoding dicts
# ---------------------------------------------------------------------------


class TestEncodings:
    def test_tariff(self):
        assert TARIFF_TO_IDX == {"Std": 0, "ToU": 1}

    def test_acorn_grouped_includes_unclassified(self):
        assert ACORN_GROUPED_TO_IDX["Affluent"] == 0
        assert ACORN_GROUPED_TO_IDX["Comfortable"] == 1
        assert ACORN_GROUPED_TO_IDX["Adversity"] == 2
        assert ACORN_GROUPED_TO_IDX["ACORN-U"] == 3
        assert len(ACORN_GROUPED_TO_IDX) == 4


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------


class TestLoadHouseholdMeta:
    def test_all_four_acorn_classes(self, tmp_path: Path):
        info = tmp_path / "informations_households.csv"
        _write_info_csv(
            str(info),
            [
                ("MAC000001", "Std", "ACORN-A", "Affluent"),
                ("MAC000002", "ToU", "ACORN-F", "Comfortable"),
                ("MAC000003", "Std", "ACORN-K", "Adversity"),
                ("MAC000004", "ToU", "ACORN-U", "ACORN-U"),
            ],
        )
        pre = PreLCLElectricityCSV(root=str(tmp_path), year=2012)
        meta = pre._load_household_meta()
        assert meta == {
            "MAC000001": (0, 0),
            "MAC000002": (1, 1),
            "MAC000003": (0, 2),
            "MAC000004": (1, 3),
        }

    def test_unknown_acorn_falls_back_to_unclassified(self, tmp_path: Path):
        info = tmp_path / "informations_households.csv"
        _write_info_csv(
            str(info),
            [
                ("MAC000001", "Std", "ACORN-X", "WeirdClass"),
            ],
        )
        pre = PreLCLElectricityCSV(root=str(tmp_path), year=2012)
        meta = pre._load_household_meta()
        # WeirdClass is unknown → ACORN-U (index 3)
        assert meta["MAC000001"] == (0, 3)

    def test_unknown_tariff_household_dropped(self, tmp_path: Path):
        info = tmp_path / "informations_households.csv"
        _write_info_csv(
            str(info),
            [
                ("MAC000001", "WeirdTariff", "ACORN-A", "Affluent"),
                ("MAC000002", "Std", "ACORN-A", "Affluent"),
            ],
        )
        pre = PreLCLElectricityCSV(root=str(tmp_path), year=2012)
        meta = pre._load_household_meta()
        assert "MAC000001" not in meta
        assert meta["MAC000002"] == (0, 0)

    def test_missing_csv_raises(self, tmp_path: Path):
        pre = PreLCLElectricityCSV(root=str(tmp_path), year=2012)
        with pytest.raises(FileNotFoundError):
            pre._load_household_meta()


# ---------------------------------------------------------------------------
# End-to-end CSV → NPZ
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_lcl_release(tmp_path: Path) -> Path:
    """Write a minimal LCL release: 2 households, full January 2012 data."""
    root = tmp_path / "raw"
    root.mkdir()

    _write_info_csv(
        str(root / "informations_households.csv"),
        [
            ("MAC000001", "Std", "ACORN-A", "Affluent"),
            ("MAC000002", "ToU", "ACORN-K", "Adversity"),
        ],
    )

    hh_dir = root / "halfhourly_dataset"
    hh_dir.mkdir()
    rows = _household_january_rows("MAC000001", 2012, energy_value=0.5)
    rows += _household_january_rows("MAC000002", 2012, energy_value=1.0)
    _write_block_csv_gz(str(hh_dir / "block_0.csv.gz"), rows)
    return root


class TestPreLCLElectricityCSVEndToEnd:
    def test_january_only_one_complete_month(self, tiny_lcl_release: Path):
        pre = PreLCLElectricityCSV(root=str(tiny_lcl_release), year=2012)
        pre.load_process_save()

        # NPZ files for all three splits should exist.
        for split in ("train", "val", "test"):
            assert (tiny_lcl_release / f"lcl_electricity_2012_{split}.npz").exists()

        # Concatenate splits and verify total = 2 (one per household, January).
        total = 0
        for split in ("train", "val", "test"):
            z = np.load(tiny_lcl_release / f"lcl_electricity_2012_{split}.npz")
            if "1" in z:
                total += z["1"].shape[0]
        assert total == 2

    def test_npz_schema_has_label_keys(self, tiny_lcl_release: Path):
        pre = PreLCLElectricityCSV(root=str(tiny_lcl_release), year=2012)
        pre.load_process_save()

        # Find whichever split contains January data and assert per-row labels.
        days = calendar.monthrange(2012, 1)[1]
        expected_len = days * STEPS_PER_DAY  # 31 * 48 = 1488

        seen_label_rows = 0
        for split in ("train", "val", "test"):
            z = np.load(tiny_lcl_release / f"lcl_electricity_2012_{split}.npz")
            if "1" not in z:
                continue
            assert "1_tariff_type" in z.files
            assert "1_acorn_grouped" in z.files
            n = z["1"].shape[0]
            assert z["1"].shape[1] == expected_len
            assert z["1_tariff_type"].shape[0] == n
            assert z["1_acorn_grouped"].shape[0] == n
            seen_label_rows += n
        assert seen_label_rows == 2

    def test_kwh_to_kw_conversion(self, tiny_lcl_release: Path):
        pre = PreLCLElectricityCSV(root=str(tiny_lcl_release), year=2012)
        pre.load_process_save()
        # Energy values in fixture were 0.5 and 1.0 kWh/hh →
        # power values should be 1.0 and 2.0 kW (×2).
        all_values: list[float] = []
        for split in ("train", "val", "test"):
            z = np.load(tiny_lcl_release / f"lcl_electricity_2012_{split}.npz")
            if "1" in z and z["1"].shape[0] > 0:
                all_values.extend(np.unique(z["1"]).tolist())
        # Two unique values across the two households.
        unique_sorted = sorted({round(v, 6) for v in all_values})
        assert unique_sorted == [pytest.approx(1.0), pytest.approx(2.0)]

    def test_label_values_are_in_range(self, tiny_lcl_release: Path):
        pre = PreLCLElectricityCSV(root=str(tiny_lcl_release), year=2012)
        pre.load_process_save()
        for split in ("train", "val", "test"):
            z = np.load(tiny_lcl_release / f"lcl_electricity_2012_{split}.npz")
            if "1_tariff_type" in z.files:
                assert set(np.unique(z["1_tariff_type"]).tolist()) <= {0, 1}
            if "1_acorn_grouped" in z.files:
                assert set(np.unique(z["1_acorn_grouped"]).tolist()) <= {0, 1, 2, 3}

    def test_incomplete_month_dropped(self, tmp_path: Path):
        """A household with a *large* gap (> max_interp_slots) is dropped.

        Single-slot gaps are interpolated (see ``test_short_gap_interpolated``).
        Here we drop 10 consecutive rows, which exceeds the 4-slot interp
        budget, so the household-month must be discarded.
        """
        root = tmp_path / "raw"
        root.mkdir()
        _write_info_csv(
            str(root / "informations_households.csv"),
            [("MAC000001", "Std", "ACORN-A", "Affluent")],
        )
        hh_dir = root / "halfhourly_dataset"
        hh_dir.mkdir()
        rows = _household_january_rows("MAC000001", 2012, energy_value=0.5)
        # Drop 10 consecutive rows — well above the 4-slot interpolation budget.
        del rows[100:110]
        _write_block_csv_gz(str(hh_dir / "block_0.csv.gz"), rows)

        pre = PreLCLElectricityCSV(root=str(root), year=2012)
        pre.load_process_save()
        total_jan = 0
        for split in ("train", "val", "test"):
            z = np.load(root / f"lcl_electricity_2012_{split}.npz")
            if "1" in z:
                total_jan += z["1"].shape[0]
        assert total_jan == 0

    def test_short_gap_interpolated(self, tmp_path: Path):
        """A household with a single missing slot is recovered via
        linear interpolation (handles the LCL 2012-12-18 outage pattern).
        """
        root = tmp_path / "raw"
        root.mkdir()
        _write_info_csv(
            str(root / "informations_households.csv"),
            [("MAC000001", "Std", "ACORN-A", "Affluent")],
        )
        hh_dir = root / "halfhourly_dataset"
        hh_dir.mkdir()
        rows = _household_january_rows("MAC000001", 2012, energy_value=0.5)
        # Drop a single row → 1 NaN slot, within the interp budget.
        rows.pop(100)
        _write_block_csv_gz(str(hh_dir / "block_0.csv.gz"), rows)

        pre = PreLCLElectricityCSV(root=str(root), year=2012)
        pre.load_process_save()
        total_jan = 0
        for split in ("train", "val", "test"):
            z = np.load(root / f"lcl_electricity_2012_{split}.npz")
            if "1" in z:
                total_jan += z["1"].shape[0]
        # Single sample recovered (constant 0.5 kWh/hh → 1.0 kW everywhere).
        assert total_jan == 1


# ---------------------------------------------------------------------------
# split_and_save_npz alignment
# ---------------------------------------------------------------------------


class TestSplitAndSaveNpzAlignment:
    def test_data_and_labels_share_permutation(self, tmp_path: Path):
        """Sentinel test: a row whose data is uniquely identifiable should
        always be paired with its corresponding label across all splits.
        """
        n = 20
        data = np.zeros((n, 4), dtype=np.float32)
        # Each row carries its own row-id encoded in column 0 so we can
        # find it after shuffling/splitting.
        data[:, 0] = np.arange(n)
        labels = np.arange(n, dtype=np.int64) * 10  # row i -> 10*i

        split_and_save_npz(
            dataset_by_month={"1": data},
            root=str(tmp_path),
            prefix="test_split",
            year=2099,
            ratios=(0.5, 0.25, 0.25),
            labels_by_month={"1": {"sentinel": labels}},
        )

        seen = 0
        for split in ("train", "val", "test"):
            z = np.load(tmp_path / f"test_split_2099_{split}.npz")
            if "1" not in z:
                continue
            d = z["1"]
            lab = z["1_sentinel"]
            for i in range(d.shape[0]):
                row_id = int(d[i, 0])
                assert lab[i] == row_id * 10, (
                    f"split={split} i={i} row_id={row_id} label={lab[i]}"
                )
                seen += 1
        assert seen == n

    def test_backward_compat_when_labels_none(self, tmp_path: Path):
        """Calling without labels_by_month yields the legacy NPZ schema."""
        data = np.arange(40, dtype=np.float32).reshape(8, 5)
        split_and_save_npz(
            dataset_by_month={"1": data},
            root=str(tmp_path),
            prefix="legacy",
            year=2099,
        )
        # Only the data key should be present.
        for split in ("train", "val", "test"):
            z = np.load(tmp_path / f"legacy_2099_{split}.npz")
            assert set(z.files) <= {"1"}


# ---------------------------------------------------------------------------
# LCLCondition validation
# ---------------------------------------------------------------------------


class TestLCLCondition:
    def test_new_fields_in_range(self):
        c = LCLCondition(month=0, year=2012, tariff_type=1, acorn_grouped=2)
        assert c.tariff_type == 1
        assert c.acorn_grouped == 2

    def test_tariff_out_of_range(self):
        with pytest.raises(ValidationError, match="tariff_type"):
            LCLCondition(month=0, year=2012, tariff_type=2)

    def test_acorn_out_of_range(self):
        with pytest.raises(ValidationError, match="acorn_grouped"):
            LCLCondition(month=0, year=2012, acorn_grouped=4)

    def test_to_tensor_dict_includes_new_fields(self):
        c = LCLCondition(month=3, year=2012, tariff_type=1, acorn_grouped=3)
        td = c.to_tensor_dict(batch_size=4)
        assert set(td) >= {"month", "year", "tariff_type", "acorn_grouped"}
        assert torch.equal(td["tariff_type"], torch.full((4, 1), 1))
        assert torch.equal(td["acorn_grouped"], torch.full((4, 1), 3))

    def test_to_tensor_dict_omits_unset_fields(self):
        c = LCLCondition(month=3, year=2012)
        td = c.to_tensor_dict(batch_size=2)
        # Calendar fields are auto-derived; tariff/acorn remain None and are
        # not emitted.
        assert "tariff_type" not in td
        assert "acorn_grouped" not in td
        assert "month" in td and "year" in td

    def test_force_drop_ids(self):
        c = LCLCondition(month=0, year=2012, tariff_type=0)
        ids = c.to_force_drop_ids(batch_size=3)
        assert ids["tariff_type"].tolist() == [0, 0, 0]
        assert ids["acorn_grouped"].tolist() == [1, 1, 1]


# ---------------------------------------------------------------------------
# Embedder backward compatibility + forward shape
# ---------------------------------------------------------------------------


def _legacy_kwargs(dim_base: int = 32) -> dict:
    """Embedder kwargs matching the pre-tariff/acorn LCL embedder constructor."""
    return {
        "month": {"num_embedding": 12, "dim_embedding": dim_base, "dropout": 0.1},
        "year": {"dim_embedding": dim_base},
        "first_day_of_week": {
            "num_embedding": 7,
            "dim_embedding": dim_base,
            "dropout": 0.1,
        },
        "month_length": {
            "num_embedding": 4,
            "dim_embedding": dim_base,
            "dropout": 0.1,
        },
    }


class TestLCLEmbedderBackwardCompat:
    def test_state_dict_keys_unchanged_when_optionals_absent(self):
        """state_dict keys must NOT include tariff/acorn entries when the
        new kwargs are omitted — otherwise existing checkpoints would fail
        to load.
        """
        emb = LCLLabelEmbedder(**_legacy_kwargs())
        keys = set(emb.state_dict().keys())
        # No keys should contain "tariff_type" or "acorn_grouped"
        assert not any("tariff_type" in k for k in keys)
        assert not any("acorn_grouped" in k for k in keys)
        # Sanity: month / year / first_day_of_week / month_length keys exist
        assert any("month" in k for k in keys)
        assert any("year" in k for k in keys)

    def test_forward_with_new_fields(self):
        kw = _legacy_kwargs()
        kw["tariff_type"] = {
            "num_embedding": 2,
            "dim_embedding": 32,
            "dropout": 0.1,
        }
        kw["acorn_grouped"] = {
            "num_embedding": 4,
            "dim_embedding": 32,
            "dropout": 0.1,
        }
        emb = LCLLabelEmbedder(**kw)
        # state_dict now includes tariff/acorn modules
        keys = set(emb.state_dict().keys())
        assert any("tariff_type" in k for k in keys)
        assert any("acorn_grouped" in k for k in keys)

        labels = {
            "month": torch.zeros(2, 1, dtype=torch.long),
            "year": torch.full((2, 1), 2012, dtype=torch.long),
            "first_day_of_week": torch.zeros(2, 1, dtype=torch.long),
            "month_length": torch.full((2, 1), 3, dtype=torch.long),
            "tariff_type": torch.zeros(2, 1, dtype=torch.long),
            "acorn_grouped": torch.full((2, 1), 2, dtype=torch.long),
        }
        out = emb(labels, batch_size=2)
        assert out.shape == (2, 32)
