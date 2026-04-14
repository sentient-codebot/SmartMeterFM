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
):
    """Shuffle, split into train/val/test, and save as compressed NPZ.

    Args:
        dataset_by_month: dict mapping month string -> array or None
        root: directory to save into
        prefix: filename prefix (e.g., "wpuq", "wpuq_household", "wpuq_pv")
        year: recording year
        ratios: (train, val, test) split ratios
    """
    train_r, val_r, _ = ratios
    splits = {"train": {}, "val": {}, "test": {}}

    for month_str, data in dataset_by_month.items():
        if data is None:
            continue
        data = shuffle_array(data.astype(np.float32))
        n = len(data)
        i_train = int(n * train_r)
        i_val = int(n * (train_r + val_r))
        splits["train"][month_str] = data[:i_train]
        splits["val"][month_str] = data[i_train:i_val]
        splits["test"][month_str] = data[i_val:]

    for task_name, task_data in splits.items():
        np.savez_compressed(
            os.path.join(root, f"{prefix}_{year}_{task_name}.npz"),
            **task_data,
        )
