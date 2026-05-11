"""Season utilities and shared preprocessing helpers."""

import os

import numpy as np


NAME_SEASONS = ["winter", "spring", "summer", "autumn"]


def months_of_season(season: str) -> list[int]:
    season = season.lower()
    if season == "winter":
        months = [12, 1, 2]
    elif season == "spring":
        months = [3, 4, 5]
    elif season == "summer":
        months = [6, 7, 8]
    elif season in ["autumn", "fall"]:
        months = [9, 10, 11]
    else:
        raise ValueError("Invalid")
    return months


def season_of_month(month: int) -> str:
    assert month in range(1, 13)
    if month in [12, 1, 2]:
        season = "winter"
    elif month in [3, 4, 5]:
        season = "spring"
    elif month in [6, 7, 8]:
        season = "summer"
    elif month in [9, 10, 11]:
        season = "autumn"
    else:
        raise ValueError("Invalid")
    return season


def shuffle_array(array: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng()
    indices = rng.permutation(len(array))
    return array[indices]


def split_and_save_npz(
    dataset_by_month: dict[str, np.ndarray | None],
    root: str,
    prefix: str,
    year: int,
    ratios: tuple[float, float, float] = (0.5, 0.25, 0.25),
    labels_by_month: dict[str, dict[str, np.ndarray]] | None = None,
):
    """Shuffle, split into train/val/test, and save as compressed NPZ.

    Args:
        dataset_by_month: dict mapping month string -> array or None
        root: directory to save into
        prefix: filename prefix (e.g., "wpuq", "wpuq_household", "wpuq_pv")
        year: recording year
        ratios: (train, val, test) split ratios
        labels_by_month: optional dict mapping ``month_str -> {label_name: array}``
            of per-row labels aligned with ``dataset_by_month[month_str]``.  When
            provided, each label array is shuffled with the SAME permutation as
            the data array so row alignment is preserved through the split, and
            each split's NPZ stores label arrays under key ``"{month}_{label}"``
            alongside the data array at key ``"{month}"``.  Defaults to ``None``
            (no per-row labels — backward compatible with existing pipelines).
    """
    train_r, val_r, _ = ratios
    splits: dict[str, dict[str, np.ndarray]] = {"train": {}, "val": {}, "test": {}}
    rng = np.random.default_rng()

    for month_str, data in dataset_by_month.items():
        if data is None:
            continue
        data = data.astype(np.float32)
        n = len(data)

        # Validate aligned labels (if any) before we shuffle so a mismatch
        # surfaces as an AssertionError rather than corrupted alignment.
        month_labels = (
            labels_by_month.get(month_str, {}) if labels_by_month is not None else {}
        )
        for label_name, label_arr in month_labels.items():
            assert len(label_arr) == n, (
                f"label {label_name!r} for month {month_str!r} has length "
                f"{len(label_arr)}, expected {n} (data length)"
            )

        # Single permutation shared across data + every label.
        perm = rng.permutation(n)
        data = data[perm]
        month_labels = {name: arr[perm] for name, arr in month_labels.items()}

        i_train = int(n * train_r)
        i_val = int(n * (train_r + val_r))
        bounds = {
            "train": (0, i_train),
            "val": (i_train, i_val),
            "test": (i_val, n),
        }
        for task_name, (lo, hi) in bounds.items():
            splits[task_name][month_str] = data[lo:hi]
            for label_name, label_arr in month_labels.items():
                splits[task_name][f"{month_str}_{label_name}"] = label_arr[lo:hi]

    for task_name, task_data in splits.items():
        np.savez_compressed(
            os.path.join(root, f"{prefix}_{year}_{task_name}.npz"),
            **task_data,
        )
