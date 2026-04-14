"""Verify processed LCL smart meter data with visual plots and statistics.

Loads the LCL dataset in raw (un-normalized, un-vectorized) form, produces
four diagnostic figures and a statistics table to stdout.

Note: LCL values are average power in **kW** (converted from kWh/half-hour
during preprocessing).  To recover energy in kWh, multiply by the interval
duration (0.5 h).

Usage:
    uv run python scripts/showcase/verify_lcl_data.py
"""

import calendar
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from smartmeterfm.data_modules.lcl_electricity import LCLElectricity
from smartmeterfm.utils.configuration import DataConfig


OUTDIR = "results/lcl_verification"
STEPS_PER_DAY = 48
MONTH_NAMES = [calendar.month_abbr[m] for m in range(1, 13)]
SEASON_COLORS = {
    "winter": "#3b82f6",
    "spring": "#22c55e",
    "summer": "#f59e0b",
    "autumn": "#ef4444",
}
MONTH_TO_SEASON = [
    "winter",
    "winter",
    "spring",
    "spring",
    "spring",
    "summer",
    "summer",
    "summer",
    "autumn",
    "autumn",
    "autumn",
    "winter",
]


def load_raw_data():
    """Load LCL data without normalization or vectorization."""
    cfg = DataConfig(
        dataset="lcl_electricity",
        root="data/lcl_electricity/",
        resolution="30min",
        load=False,
        normalize=False,
        normalize_method="meanstd",
        pit=False,
        shuffle=False,
        vectorize=False,
        style_vectorize="patchify",
        vectorize_window_size=16,
        train_season="whole_year",
        val_season="whole_year",
        target_labels="month",
        segment_type="monthly",
    )
    return LCLElectricity(cfg)


def get_all_profiles_and_labels(data):
    """Concatenate train/val/test into single arrays."""
    profiles = torch.cat(
        [data.dataset.profile[t] for t in ["train", "val", "test"]], dim=0
    )
    labels = torch.cat(
        [data.dataset.label[t]["month"] for t in ["train", "val", "test"]], dim=0
    )
    # profiles: [N, 1488, 1] (un-vectorized) -> squeeze to [N, 1488]
    profiles = profiles.squeeze(-1).numpy()
    labels = labels.squeeze(-1).numpy()
    return profiles, labels


def print_statistics(profiles, labels):
    """Print per-month and overall statistics."""
    print("=" * 80)
    print("LCL Dataset Statistics")
    print("=" * 80)
    print(
        f"{'Month':<8} {'Count':>7} {'Mean kWh':>10} {'Std':>10} "
        f"{'Median':>10} {'Min':>10} {'Max':>10}"
    )
    print("-" * 80)

    for m in range(12):
        mask = labels == m
        if mask.sum() == 0:
            continue
        month_profiles = profiles[mask]
        # Total consumption per household-month in kWh.
        # Values are in kW; multiply by interval duration (0.5 h) to get kWh.
        totals = month_profiles.sum(axis=1) * 0.5
        print(
            f"{MONTH_NAMES[m]:<8} {mask.sum():>7d} {totals.mean():>10.1f} "
            f"{totals.std():>10.1f} {np.median(totals):>10.1f} "
            f"{totals.min():>10.1f} {totals.max():>10.1f}"
        )

    print("-" * 80)
    totals_all = profiles.sum(axis=1) * 0.5  # kW → kWh (×0.5 h interval)
    print(
        f"{'All':<8} {len(profiles):>7d} {totals_all.mean():>10.1f} "
        f"{totals_all.std():>10.1f} {np.median(totals_all):>10.1f} "
        f"{totals_all.min():>10.1f} {totals_all.max():>10.1f}"
    )
    print()
    print(f"Value range (half-hourly): [{profiles.min():.4f}, {profiles.max():.4f}] kW")
    print(f"Fraction of exact zeros:   {(profiles == 0).mean():.4f}")
    print("Fraction of padding zeros: see month-end padding below")

    # Check padding: for months < 31 days, trailing values should be 0
    padding_counts = {}
    for m in range(12):
        days = calendar.monthrange(2012, m + 1)[1]  # representative year
        actual_steps = days * STEPS_PER_DAY
        pad_steps = 31 * STEPS_PER_DAY - actual_steps
        if pad_steps > 0:
            mask = labels == m
            month_profiles = profiles[mask]
            padding_region = month_profiles[:, actual_steps:]
            padding_counts[MONTH_NAMES[m]] = {
                "pad_steps": pad_steps,
                "all_zero": (padding_region == 0).all(),
            }

    if padding_counts:
        print("\nPadding verification (months < 31 days):")
        for name, info in padding_counts.items():
            status = "OK" if info["all_zero"] else "UNEXPECTED NON-ZERO"
            print(f"  {name}: {info['pad_steps']} padded steps — {status}")
    print()


def fig1_sample_counts(data):
    """Bar chart of sample counts per month, split by train/val/test."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(12)
    width = 0.25

    for i, (split, color) in enumerate(
        [("train", "#3b82f6"), ("val", "#22c55e"), ("test", "#ef4444")]
    ):
        counts = []
        for m in range(12):
            mask = data.dataset.label[split]["month"].squeeze(-1) == m
            counts.append(mask.sum().item())
        ax.bar(x + i * width, counts, width, label=split, color=color, alpha=0.85)

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of household-months")
    ax.set_title("LCL Dataset: Sample Count per Month")
    ax.set_xticks(x + width)
    ax.set_xticklabels(MONTH_NAMES)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig2_mean_daily_profiles(profiles, labels):
    """Mean daily profile (averaged across days and households) per month."""
    fig, ax = plt.subplots(figsize=(10, 6))
    hours = np.arange(STEPS_PER_DAY) * 0.5  # 0, 0.5, 1.0, ..., 23.5

    for m in range(12):
        mask = labels == m
        if mask.sum() == 0:
            continue
        month_profiles = profiles[mask]
        days = calendar.monthrange(2012, m + 1)[1]
        actual_steps = days * STEPS_PER_DAY
        # Trim padding, reshape to [N * days, 48], take mean
        trimmed = month_profiles[:, :actual_steps]
        daily = trimmed.reshape(-1, STEPS_PER_DAY)
        mean_daily = daily.mean(axis=0)

        color = SEASON_COLORS[MONTH_TO_SEASON[m]]
        ax.plot(hours, mean_daily, label=MONTH_NAMES[m], color=color, alpha=0.8)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean power [kW]")
    ax.set_title("LCL: Mean Daily Profile by Month")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig3_monthly_distributions(profiles, labels):
    """Box plot of total monthly consumption per household-month."""
    fig, ax = plt.subplots(figsize=(10, 5))

    data_per_month = []
    colors = []
    for m in range(12):
        mask = labels == m
        if mask.sum() == 0:
            data_per_month.append([0])
            colors.append("#999999")
            continue
        # kW × 0.5 h = kWh per interval; sum gives total monthly kWh
        totals = profiles[mask].sum(axis=1) * 0.5
        data_per_month.append(totals)
        colors.append(SEASON_COLORS[MONTH_TO_SEASON[m]])

    bp = ax.boxplot(
        data_per_month,
        tick_labels=MONTH_NAMES,
        patch_artist=True,
        showfliers=False,  # hide outliers for cleaner plot
        medianprops={"color": "black"},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Month")
    ax.set_ylabel("Total monthly consumption [kWh]")
    ax.set_title("LCL: Monthly Consumption Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig4_sample_gallery(profiles, labels):
    """Grid of 12 subplots (one per month), each showing 5 random profiles."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    rng = np.random.default_rng(42)

    for m, ax in enumerate(axes.flat):
        mask = labels == m
        if mask.sum() == 0:
            ax.set_title(MONTH_NAMES[m])
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        month_profiles = profiles[mask]
        days = calendar.monthrange(2012, m + 1)[1]
        actual_steps = days * STEPS_PER_DAY
        hours = np.arange(actual_steps) / STEPS_PER_DAY  # in days

        indices = rng.choice(
            len(month_profiles), size=min(5, len(month_profiles)), replace=False
        )
        for idx in indices:
            ax.plot(hours, month_profiles[idx, :actual_steps], alpha=0.6, linewidth=0.5)

        ax.set_title(MONTH_NAMES[m], fontsize=10)
        ax.set_xlim(0, days)
        ax.set_xlabel("Day", fontsize=7)
        ax.set_ylabel("kW", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    fig.suptitle("LCL: Random Household-Month Profiles (5 per month)", fontsize=12)
    fig.tight_layout()
    return fig


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Loading raw LCL data (no normalization, no vectorization)...")
    data = load_raw_data()
    profiles, labels = get_all_profiles_and_labels(data)
    print(f"Loaded {len(profiles)} samples, shape per sample: {profiles.shape[1:]}")
    print()

    print_statistics(profiles, labels)

    print("Generating figures...")
    for name, fig_fn in [
        ("sample_counts", lambda: fig1_sample_counts(data)),
        ("mean_daily_profiles", lambda: fig2_mean_daily_profiles(profiles, labels)),
        ("monthly_distributions", lambda: fig3_monthly_distributions(profiles, labels)),
        ("sample_gallery", lambda: fig4_sample_gallery(profiles, labels)),
    ]:
        fig = fig_fn()
        path = os.path.join(OUTDIR, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
