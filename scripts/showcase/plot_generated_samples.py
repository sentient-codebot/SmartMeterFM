"""
Plot generated samples vs real test data for WPuQ Household.

Creates per-month comparison figures and a summary metrics bar chart.

Usage:
    uv run python scripts/showcase/plot_generated_samples.py \
        --samples_dir samples/wpuq_household \
        --config configs/showcase/wpuq_household_flow_monthly.toml \
        --output_dir results/figures/wpuq_household_50k
"""

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange

from smartmeterfm.data_modules.heat_pump import WPuQ
from smartmeterfm.data_modules.lcl_electricity import LCLElectricity
from smartmeterfm.data_modules.wpuq_household import WPuQHousehold
from smartmeterfm.utils.configuration import ExperimentConfig
from smartmeterfm.utils.plot import plot_time_series_comparison_advanced


MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - plot - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_generated_samples(samples_dir: str) -> dict[int, torch.Tensor]:
    all_samples_path = os.path.join(samples_dir, "generated_samples.pt")
    if os.path.exists(all_samples_path):
        return torch.load(all_samples_path, map_location="cpu", weights_only=True)
    samples_by_month = {}
    for month in range(12):
        month_path = os.path.join(samples_dir, f"month_{month:02d}_samples.pt")
        if os.path.exists(month_path):
            samples_by_month[month] = torch.load(
                month_path, map_location="cpu", weights_only=True
            )
    return samples_by_month


def load_test_data(config: ExperimentConfig):
    dataset_name = getattr(config.data, "dataset", "wpuq")
    if dataset_name == "lcl_electricity":
        data_collection = LCLElectricity(config.data)
    elif dataset_name == "wpuq_household":
        data_collection = WPuQHousehold(config.data)
    else:
        data_collection = WPuQ(config.data)
    test_profiles = data_collection.dataset.profile["test"]
    test_labels = data_collection.dataset.label["test"]
    year_labels = test_labels["year"] if "year" in test_labels else None
    return test_profiles, test_labels["month"], year_labels


def plot_per_month_comparison(
    generated: dict[int, torch.Tensor],
    real_profiles: torch.Tensor,
    real_labels: torch.Tensor,
    output_dir: str,
    real_year_labels: torch.Tensor | None = None,
    year: int | None = None,
):
    """Plot generated vs real comparison for each month."""
    real_labels_flat = real_labels.squeeze(-1)
    real_year_flat = (
        real_year_labels.squeeze(-1) if real_year_labels is not None else None
    )
    months_dir = os.path.join(output_dir, "per_month")
    os.makedirs(months_dir, exist_ok=True)

    for month in sorted(generated.keys()):
        gen_samples = generated[month]
        month_mask = real_labels_flat == month
        if year is not None and real_year_flat is not None:
            month_mask = month_mask & (real_year_flat == year)
        real_month = real_profiles[month_mask]

        if real_month.shape[0] == 0:
            logging.warning(f"No real samples for month {month}, skipping.")
            continue

        output_path = os.path.join(months_dir, f"month_{month + 1:02d}.png")
        plot_time_series_comparison_advanced(
            generated=gen_samples,
            real=real_month,
            output_path=output_path,
            title=f"{MONTH_NAMES[month]} — Generated vs Real",
            ymin=-4,
            ymax=6,
        )
        logging.info(
            f"Month {month + 1}: {gen_samples.shape[0]} gen, "
            f"{real_month.shape[0]} real → {output_path}"
        )


def plot_mean_std_comparison(
    generated: dict[int, torch.Tensor],
    real_profiles: torch.Tensor,
    real_labels: torch.Tensor,
    output_dir: str,
    real_year_labels: torch.Tensor | None = None,
    year: int | None = None,
):
    """Plot mean and std profiles for generated vs real, per month (3x4 grid)."""
    real_labels_flat = real_labels.squeeze(-1)
    real_year_flat = (
        real_year_labels.squeeze(-1) if real_year_labels is not None else None
    )

    with plt.style.context("smartmeterfm.utils.article_compatible"):
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for month in range(12):
            ax = axes[month]

            # Flatten patchified data to 1D time series
            if month in generated:
                gen = generated[month]
                if gen.dim() == 3:
                    gen = rearrange(gen, "b s c -> b (s c)")
                gen_mean = gen.mean(dim=0).numpy()
                gen_std = gen.std(dim=0).numpy()
            else:
                continue

            month_mask = real_labels_flat == month
            if year is not None and real_year_flat is not None:
                month_mask = month_mask & (real_year_flat == year)
            real_month = real_profiles[month_mask]
            if real_month.dim() == 3:
                real_month = rearrange(real_month, "b s c -> b (s c)")
            real_mean = real_month.mean(dim=0).numpy()
            real_std = real_month.std(dim=0).numpy()

            t = np.arange(len(gen_mean))

            ax.plot(t, gen_mean, color="tab:orange", label="Generated", linewidth=1.2)
            ax.fill_between(
                t,
                gen_mean - gen_std,
                gen_mean + gen_std,
                color="tab:orange",
                alpha=0.2,
            )
            ax.plot(t, real_mean, color="tab:blue", label="Real", linewidth=1.2)
            ax.fill_between(
                t,
                real_mean - real_std,
                real_mean + real_std,
                color="tab:blue",
                alpha=0.2,
            )

            ax.set_title(MONTH_NAMES[month])
            ax.set_xlim(0, len(t) - 1)
            ax.set_ylim(-3, 5)
            ax.grid(True, alpha=0.3)

            if month == 0:
                ax.legend(fontsize=8)
            if month >= 9:
                ax.set_xlabel("Time Step")
            if month % 3 == 0:
                ax.set_ylabel("Normalized Power")

        fig.suptitle(
            "Mean ± Std: Generated vs Real Profiles",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        output_path = os.path.join(output_dir, "mean_std_comparison.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved mean/std comparison to {output_path}")


def plot_metrics_summary(eval_metrics_path: str, output_dir: str):
    """Plot bar charts of evaluation metrics per month."""
    if not os.path.exists(eval_metrics_path):
        logging.warning(f"Eval metrics not found at {eval_metrics_path}, skipping.")
        return

    with open(eval_metrics_path) as f:
        results = json.load(f)

    per_month = results.get("per_month", {})
    if not per_month:
        return

    months = sorted(per_month.keys())
    month_labels = [MONTH_NAMES[int(m.split("_")[1]) - 1][:3] for m in months]

    metrics_to_plot = ["DirectFD", "kl_divergence", "ws_distance", "MkMMD"]
    metric_labels = [
        "Fréchet Distance",
        "KL Divergence",
        "Wasserstein Distance",
        "MkMMD",
    ]

    with plt.style.context("smartmeterfm.utils.article_compatible"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()

        for idx, (metric_key, metric_label) in enumerate(
            zip(metrics_to_plot, metric_labels)
        ):
            ax = axes[idx]
            values = [per_month[m].get(metric_key, 0) for m in months]
            overall = results.get("overall", {}).get(metric_key, None)

            bars = ax.bar(month_labels, values, color="tab:blue", alpha=0.7)
            if overall is not None and not np.isnan(overall):
                ax.axhline(
                    y=overall,
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Overall: {overall:.4f}",
                )
                ax.legend(fontsize=8)

            ax.set_title(metric_label)
            ax.set_ylabel(metric_key)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "Evaluation Metrics per Month",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        output_path = os.path.join(output_dir, "metrics_summary.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved metrics summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot generated samples vs real test data"
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        required=True,
        help="Directory containing generated samples",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config (TOML)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures/wpuq_household",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--eval_metrics",
        type=str,
        default=None,
        help="Path to eval_metrics.json (optional, for metrics bar chart)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Filter real test data to this year only (e.g. 2013). "
        "Should match the year used during generation.",
    )
    args = parser.parse_args()

    setup_logging()

    config = ExperimentConfig.from_toml(args.config)
    generated = load_generated_samples(args.samples_dir)
    if not generated:
        logging.error("No generated samples found.")
        return

    real_profiles, real_labels, real_year_labels = load_test_data(config)

    if args.year is not None:
        logging.info(f"Filtering real data to year {args.year}")

    # 1. Per-month overlap comparison
    logging.info("Plotting per-month comparisons...")
    plot_per_month_comparison(
        generated,
        real_profiles,
        real_labels,
        args.output_dir,
        real_year_labels=real_year_labels,
        year=args.year,
    )

    # 2. Mean ± std comparison grid
    logging.info("Plotting mean/std comparison...")
    plot_mean_std_comparison(
        generated,
        real_profiles,
        real_labels,
        args.output_dir,
        real_year_labels=real_year_labels,
        year=args.year,
    )

    # 3. Metrics summary bar chart
    eval_path = args.eval_metrics or os.path.join(
        args.output_dir.replace("figures", "eval"), "eval_metrics.json"
    )
    logging.info("Plotting metrics summary...")
    plot_metrics_summary(eval_path, args.output_dir)

    logging.info(f"All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
