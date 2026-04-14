"""
Evaluate generated samples against real test data using distribution metrics.

This script loads generated samples (from generate_samples.py) and real test
data, then computes evaluation metrics including MMD, KL divergence,
Wasserstein distance, Frechet distance, and KS test statistics.

Usage:
    uv run python scripts/showcase/evaluate_samples.py \
        --samples_dir samples/wpuq_household \
        --config configs/showcase/wpuq_household_flow_monthly.toml \
        --output_dir results/eval/wpuq_household

Example:
    uv run python scripts/showcase/evaluate_samples.py \
        --samples_dir samples/wpuq_household \
        --config configs/showcase/wpuq_household_flow_monthly.toml \
        --output_dir results/eval/wpuq_household \
        --num_permutations 500
"""

import argparse
import json
import logging
import os

import torch

from smartmeterfm.data_modules.wpuq_household import WPuQHousehold
from smartmeterfm.utils.configuration import ExperimentConfig
from smartmeterfm.utils.eval import (
    MkMMD,
    MultiMetric,
    calculate_frechet,
    kl_divergence,
    ks_test_d,
    ks_test_p,
    source_mean,
    source_std,
    target_mean,
    target_std,
    ws_distance,
)


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - evaluate - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_generated_samples(samples_dir: str) -> dict[int, torch.Tensor]:
    """Load generated samples from directory.

    Expects either a single generated_samples.pt file (dict mapping month -> tensor)
    or individual month_XX_samples.pt files.
    """
    all_samples_path = os.path.join(samples_dir, "generated_samples.pt")
    if os.path.exists(all_samples_path):
        logging.info(f"Loading all samples from {all_samples_path}")
        return torch.load(all_samples_path, map_location="cpu", weights_only=True)

    # Fall back to per-month files
    samples_by_month = {}
    for month in range(12):
        month_path = os.path.join(samples_dir, f"month_{month:02d}_samples.pt")
        if os.path.exists(month_path):
            samples_by_month[month] = torch.load(
                month_path, map_location="cpu", weights_only=True
            )
            logging.info(f"Loaded month {month}: {samples_by_month[month].shape}")
    return samples_by_month


def load_test_data(config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Load test data using the WPuQHousehold data module.

    Returns:
        Tuple of (test_profiles, test_month_labels)
    """
    logging.info("Loading test data from WPuQHousehold data module...")
    wpuq_data = WPuQHousehold(config.data)

    test_profiles = wpuq_data.dataset.profile["test"]  # [N, seq_len, channels]
    test_labels = wpuq_data.dataset.label["test"]

    logging.info(f"Test data shape: {test_profiles.shape}")
    logging.info(f"Test labels: {list(test_labels.dict_labels.keys())}")

    return test_profiles, test_labels["month"]


def compute_metrics_per_month(
    generated: dict[int, torch.Tensor],
    real_profiles: torch.Tensor,
    real_labels: torch.Tensor,
    device: str = "cpu",
) -> dict[str, dict]:
    """Compute evaluation metrics per month and overall.

    Args:
        generated: Dict mapping month index to generated samples.
        real_profiles: Real test profiles [N, seq_len, channels].
        real_labels: Month labels [N, 1].
        device: Device for computation.

    Returns:
        Dictionary of metric results.
    """
    mkmmd = MkMMD(kernel_type="rbf", num_kernel=1, kernel_mul=2.0, coefficient="auto")
    metric_fns = {
        "MkMMD": mkmmd,
        "DirectFD": calculate_frechet,
        "kl_divergence": kl_divergence,
        "ws_distance": ws_distance,
        "ks_test_d": ks_test_d,
        "ks_test_p": ks_test_p,
        "source_mean": source_mean,
        "source_std": source_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    multi_metric = MultiMetric(metric_fns, compute_on_cpu=True)

    results = {"per_month": {}, "overall": {}}

    # Flatten labels for comparison
    real_labels_flat = real_labels.squeeze(-1)  # [N]

    # Per-month evaluation
    all_gen_samples = []
    all_real_samples = []

    for month in sorted(generated.keys()):
        gen_samples = generated[month].to(device)
        # Get real samples for this month
        month_mask = real_labels_flat == month
        real_month = real_profiles[month_mask].to(device)

        if real_month.shape[0] == 0:
            logging.warning(f"No real samples for month {month}, skipping.")
            continue

        logging.info(
            f"Month {month + 1}: {gen_samples.shape[0]} generated, "
            f"{real_month.shape[0]} real samples"
        )

        # Compute metrics for this month
        multi_metric.reset()
        month_results = multi_metric(gen_samples, real_month)

        # Convert to serializable format
        results["per_month"][f"month_{month + 1:02d}"] = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in month_results.items()
        }

        all_gen_samples.append(gen_samples)
        all_real_samples.append(real_month)

    # Overall evaluation (all months combined)
    if all_gen_samples and all_real_samples:
        all_gen = torch.cat(all_gen_samples, dim=0)
        all_real = torch.cat(all_real_samples, dim=0)

        logging.info(
            f"Overall: {all_gen.shape[0]} generated, {all_real.shape[0]} real samples"
        )

        multi_metric.reset()
        overall_results = multi_metric(all_gen, all_real)
        results["overall"] = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in overall_results.items()
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated samples against real test data"
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        required=True,
        help="Directory containing generated samples (from generate_samples.py)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config (TOML) used during training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/eval",
        help="Directory to save evaluation results (default: results/eval)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for metric computation (default: cuda if available)",
    )
    args = parser.parse_args()

    setup_logging()

    # Load config
    logging.info(f"Loading config from {args.config}")
    config = ExperimentConfig.from_toml(args.config)

    # Load generated samples
    logging.info(f"Loading generated samples from {args.samples_dir}")
    generated = load_generated_samples(args.samples_dir)
    if not generated:
        logging.error("No generated samples found. Exiting.")
        return

    logging.info(f"Loaded samples for months: {sorted(generated.keys())}")
    for month, samples in sorted(generated.items()):
        logging.info(f"  Month {month + 1}: {samples.shape}")

    # Load real test data
    real_profiles, real_labels = load_test_data(config)

    # Compute metrics
    logging.info("Computing evaluation metrics...")
    results = compute_metrics_per_month(
        generated, real_profiles, real_labels, device=args.device
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved evaluation results to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    if results["overall"]:
        print("\nOverall Metrics:")
        for metric_name, value in sorted(results["overall"].items()):
            print(f"  {metric_name:20s}: {value:.6f}")

    print("\nPer-Month Metrics:")
    for month_key in sorted(results["per_month"].keys()):
        month_metrics = results["per_month"][month_key]
        print(f"\n  {month_key}:")
        for metric_name, value in sorted(month_metrics.items()):
            print(f"    {metric_name:20s}: {value:.6f}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
