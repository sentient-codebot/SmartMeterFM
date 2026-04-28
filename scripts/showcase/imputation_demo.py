"""
Demonstrate imputation using Flow Matching model on WPuQ Heat Pump data.

This showcase script demonstrates how to use a trained Flow Matching model
for time series imputation (filling in missing values) using the shared
``SmartMeterFMModel.impute`` interface, which wraps the model with a
``PosteriorVelocityModelWrapper`` in PROJECT mode.

The script supports different missing data patterns:
- MCAR (Missing Completely At Random): Random positions are missing
- MNAR Consecutive: Missing values occur in consecutive blocks

Usage:
    uv run python scripts/showcase/imputation_demo.py --checkpoint path/to/checkpoint.ckpt --imputation_type mcar --missing_rate 0.2

Example:
    # MCAR imputation with 20% missing data
    uv run python scripts/showcase/imputation_demo.py --checkpoint checkpoints/flow_001/last.ckpt --imputation_type mcar --missing_rate 0.2

    # Block-wise consecutive missing imputation
    uv run python scripts/showcase/imputation_demo.py --checkpoint checkpoints/flow_001/last.ckpt --imputation_type mnar_consecutive --missing_rate 0.3 --min_block_size 5
"""

import argparse
import json
import os

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from smartmeterfm.conditions import LCLCondition, WPuQCondition
from smartmeterfm.data_modules.heat_pump import WPuQ
from smartmeterfm.data_modules.lcl_electricity import LCLElectricity
from smartmeterfm.data_modules.wpuq_household import WPuQHousehold
from smartmeterfm.interfaces.smartmeter import SmartMeterFMModel
from smartmeterfm.models.baselines.interpolation import InterpolationBaseline
from smartmeterfm.utils.configuration import DataConfig
from smartmeterfm.utils.eval import (
    calculate_pearsonr_per_sample,
    crps_empirical_dimwise,
    peak_load_error,
    sym_quantile_error,
)
from smartmeterfm.utils.plot import plot_time_series_comparison_advanced


_CONDITION_CLASS = {
    "wpuq": WPuQCondition,
    "wpuq_household": WPuQCondition,
    "lcl_electricity": LCLCondition,
}


def generate_mcar_mask(
    length: int, missing_rate: float, seed: int = None
) -> torch.Tensor:
    """Generate Missing Completely At Random (MCAR) mask.

    Args:
        length: Length of the time series.
        missing_rate: Fraction of values to be missing.
        seed: Random seed for reproducibility.

    Returns:
        Boolean mask tensor where 1 = observed, 0 = missing.
    """
    if seed is not None:
        torch.manual_seed(seed)

    mask = torch.ones(length)
    num_missing = int(length * missing_rate)
    missing_indices = torch.randperm(length)[:num_missing]
    mask[missing_indices] = 0.0
    return mask


def generate_mnar_consecutive_mask(
    length: int, missing_rate: float, min_block_size: int = 5, seed: int = None
) -> torch.Tensor:
    """Generate Missing Not At Random (MNAR) mask with consecutive block pattern.

    Args:
        length: Length of the time series.
        missing_rate: Target fraction of values to be missing.
        min_block_size: Minimum size of consecutive missing blocks.
        seed: Random seed for reproducibility.

    Returns:
        Boolean mask tensor where 1 = observed, 0 = missing.
    """
    if seed is not None:
        torch.manual_seed(seed)

    mask = torch.ones(length)
    target_missing = int(length * missing_rate)
    current_missing = 0
    max_attempts = 100
    attempts = 0

    while current_missing < target_missing and attempts < max_attempts:
        remaining_needed = target_missing - current_missing
        max_block_size = min(remaining_needed, length // 4)
        block_size = torch.randint(
            min_block_size, max(min_block_size + 1, max_block_size + 1), (1,)
        ).item()

        max_start = length - block_size
        if max_start <= 0:
            break

        start_pos = torch.randint(0, max_start + 1, (1,)).item()
        end_pos = start_pos + block_size

        mask[start_pos:end_pos] = 0.0
        current_missing += block_size
        attempts += 1

    return mask


def evaluate_imputation(
    original: torch.Tensor,
    imputed: torch.Tensor,
    mask: torch.Tensor,
    baseline: torch.Tensor,
) -> dict:
    """Evaluate imputation quality against a linear-interpolation baseline.

    Args:
        original: Original complete time series.
        imputed: Imputed samples [num_samples, *shape].
        mask: Binary mask (1 = observed, 0 = missing).
        baseline: Linear-interpolation imputation, same shape as original.

    Returns:
        Dictionary of evaluation metrics. Flow-model metrics are unsuffixed;
        baseline counterparts carry the ``_baseline`` suffix.
    """
    # Get missing positions
    missing_mask = (1 - mask).bool()

    # Mean imputation (average of samples)
    mean_imputed = imputed.mean(dim=0)

    # Extract values at missing positions
    original_flat = original.flatten()
    missing_flat = missing_mask.flatten()
    original_missing = original_flat[missing_flat]
    mean_imputed_missing = mean_imputed.flatten()[missing_flat]
    baseline_missing = baseline.flatten()[missing_flat]

    # Flow metrics at missing positions
    mse = ((original_missing - mean_imputed_missing) ** 2).mean().item()
    mae = (original_missing - mean_imputed_missing).abs().mean().item()
    rmse = mse**0.5

    # Baseline metrics at missing positions
    mse_baseline = ((original_missing - baseline_missing) ** 2).mean().item()
    mae_baseline = (original_missing - baseline_missing).abs().mean().item()
    rmse_baseline = mse_baseline**0.5

    # Uncertainty (std of imputed values at missing positions)
    imputed_at_missing = imputed[:, missing_mask.expand_as(imputed[0])].view(
        imputed.shape[0], -1
    )
    uncertainty = imputed_at_missing.std(dim=0).mean().item()

    # CRPS at missing positions (baseline treated as 1-sample distribution)
    baseline_batch = baseline.flatten().unsqueeze(0)
    crps_per_dim = crps_empirical_dimwise(imputed, original_flat)
    crps_missing = crps_per_dim[missing_flat].mean().item()
    crps_baseline_per_dim = crps_empirical_dimwise(baseline_batch, original_flat)
    crps_baseline = crps_baseline_per_dim[missing_flat].mean().item()

    # Peak Load Error and Symmetric Quantile Error (full sequence)
    ple = peak_load_error(imputed, original_flat).item()
    sqe = sym_quantile_error(imputed, original_flat, quantile=0.99).item()
    ple_baseline = peak_load_error(baseline_batch, original_flat).item()
    sqe_baseline = sym_quantile_error(
        baseline_batch, original_flat, quantile=0.99
    ).item()

    # Autocorrelation comparison (Pearson R with shift=1)
    source_acorr = calculate_pearsonr_per_sample(imputed, shift=1).mean().item()
    target_acorr = calculate_pearsonr_per_sample(
        original_flat.unsqueeze(0), shift=1
    ).item()
    source_acorr_baseline = (
        calculate_pearsonr_per_sample(baseline_batch, shift=1).mean().item()
    )
    pearsonr_diff = abs(source_acorr - target_acorr)
    pearsonr_diff_baseline = abs(source_acorr_baseline - target_acorr)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mse_baseline": mse_baseline,
        "mae_baseline": mae_baseline,
        "rmse_baseline": rmse_baseline,
        "uncertainty": uncertainty,
        "crps": crps_missing,
        "crps_baseline": crps_baseline,
        "peak_load_error": ple,
        "peak_load_error_baseline": ple_baseline,
        "sym_quantile_error_99": sqe,
        "sym_quantile_error_99_baseline": sqe_baseline,
        "pearsonr_diff": pearsonr_diff,
        "pearsonr_diff_baseline": pearsonr_diff_baseline,
        "num_missing": missing_mask.sum().item(),
        "missing_rate_actual": missing_mask.float().mean().item(),
    }


# hard copied from sr demo
def plot_imp_example(
    original_real: torch.Tensor,
    imputed_samples_real: torch.Tensor,
    baseline_real: torch.Tensor,
    output_path: str,
    title: str = "Imputation Example",
    ymin: float = -0.25,
    ymax: float = 0.25,
):
    """Plot a single imputation example with original, imputed, and baseline.

    Args:
        original_real: Original HR time series [seq_length].
        imputed_samples_real: Imputed samples [num_samples, seq_length].
        baseline_real: Baseline (linear interp) HR series [seq_length].
        output_path: Path to save the figure.
        title: Figure title.
    """
    with plt.style.context("smartmeterfm.utils.article_compatible"):
        fig, ax = plt.subplots(figsize=(12, 4))

        t = np.arange(original_real.shape[0])
        imputed_mean = imputed_samples_real.mean(dim=0).numpy()
        imputed_std = imputed_samples_real.std(dim=0).numpy()

        ax.plot(
            t,
            original_real.numpy(),
            label="Original HR",
            color="tab:blue",
            linewidth=1.5,
        )
        ax.plot(
            t,
            imputed_mean,
            label="Imputed Mean",
            color="tab:orange",
            linewidth=1.5,
        )
        ax.fill_between(
            t,
            imputed_mean - imputed_std,
            imputed_mean + imputed_std,
            color="tab:orange",
            alpha=0.2,
            label="Imputed \u00b11\u03c3",
        )
        ax.plot(
            t,
            baseline_real.numpy(),
            label="Baseline (linear)",
            color="tab:green",
            linewidth=1.0,
            linestyle="--",
        )

        ax.set_xlabel("Time Step [-]")
        ax.set_ylabel("Normalized Power [-]")
        ax.set_xlim(0, len(t) - 1)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate imputation using Flow Matching model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained Flow model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wpuq", "wpuq_household", "lcl_electricity"],
        default="wpuq",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/wpuq/",
        help="Root directory for data",
    )
    parser.add_argument(
        "--imputation_type",
        type=str,
        choices=["mcar", "mnar_consecutive"],
        default="mnar_consecutive",
        help="Type of missing data pattern",
    )
    parser.add_argument(
        "--missing_rate",
        type=float,
        default=0.2,
        help="Fraction of values to be missing (default: 0.2)",
    )
    parser.add_argument(
        "--min_block_size",
        type=int,
        default=5,
        help="Minimum block size for consecutive missing (default: 5)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of imputation samples per time series (default: 10)",
    )
    parser.add_argument(
        "--num_test_series",
        type=int,
        default=100,
        help="Number of test time series to impute (default: 100)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of ODE integration steps (default: 100)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (default: 1.0)",
    )
    parser.add_argument(
        "--resample_steps",
        type=int,
        default=0,
        help="RePaint-style inner refine iterations at t<threshold (default: 0 = off)",
    )
    parser.add_argument(
        "--resample_t_threshold",
        type=float,
        default=0.4,
        help="Apply resampling only at t<this (default: 0.4)",
    )
    parser.add_argument(
        "--time_grid_mode",
        type=str,
        default="uniform",
        choices=["uniform", "geometric"],
        help="ODE time grid for posterior sampling (default: uniform)",
    )
    parser.add_argument(
        "--time_grid_gamma",
        type=float,
        default=2.0,
        help="Gamma for geometric time grid (>1 concentrates near t=0; default: 2.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--num_example_figures",
        type=int,
        default=3,
        help="Number of example figures to save (default: 3)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/imputation",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Imputation type: {args.imputation_type}")
    print(f"Missing rate: {args.missing_rate}")
    print(f"Number of samples per series: {args.num_samples}")

    # Load model via the unified interface
    print(f"\nLoading model from {args.checkpoint}...")
    model = SmartMeterFMModel.from_checkpoint(args.checkpoint, device=args.device)

    use_bf16 = model.device.type == "cuda"
    if use_bf16:
        model.pl_model.ema.ema_model = torch.compile(model.pl_model.ema.ema_model)

    # Load test data
    dataset_name = args.dataset
    resolution_map = {
        "wpuq": "15min",
        "wpuq_household": "15min",
        "lcl_electricity": "30min",
    }
    # Must match the training config for each dataset, otherwise the loaded
    # test profiles are on a different scale than the model was trained on.
    normalize_method_map = {
        "wpuq": "meanstd",
        "wpuq_household": "meanstd",
        "lcl_electricity": "constant",
    }
    print(f"\nLoading {dataset_name} test data...")
    data_config = DataConfig(
        dataset=dataset_name,
        root=args.data_root,
        load=True,
        normalize=True,
        normalize_method=normalize_method_map[dataset_name],
        pit=False,
        resolution=resolution_map[dataset_name],
        shuffle=False,
        vectorize=True,
        style_vectorize="patchify",
        vectorize_window_size=16,
        target_labels=["month"],
        train_season="whole_year",
        val_season="whole_year",
        segment_type="monthly",
    )
    if dataset_name == "lcl_electricity":
        data_collection = LCLElectricity(data_config)
    elif dataset_name == "wpuq_household":
        data_collection = WPuQHousehold(data_config)
    else:
        data_collection = WPuQ(data_config)

    test_profiles = data_collection.dataset.profile["test"][: args.num_test_series]
    test_labels = data_collection.dataset.label["test"]

    # Run imputation
    all_metrics = []

    print(f"\nRunning imputation on {len(test_profiles)} time series...")
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    example_indices = set(range(min(args.num_example_figures, len(test_profiles))))
    interp_baseline = InterpolationBaseline(method="linear")

    for idx in tqdm(range(len(test_profiles)), desc="Imputing"):
        original_patched = test_profiles[idx]  # [num_patches, patch_size]
        month = test_labels["month"][idx].item()

        # Unpatchify to real temporal space
        original_real = rearrange(original_patched, "l c -> (l c)")
        seq_length = original_real.shape[0]

        # Generate mask in real temporal space
        if args.imputation_type == "mcar":
            mask = generate_mcar_mask(
                seq_length, args.missing_rate, seed=args.seed + idx
            )
        else:
            mask = generate_mnar_consecutive_mask(
                seq_length,
                args.missing_rate,
                args.min_block_size,
                seed=args.seed + idx,
            )

        # Linear-interpolation baseline over observed positions
        baseline_real = interp_baseline.impute(original_real, mask)

        # Build condition via the typed condition class; calendar fields
        # (first_day_of_week, month_length) are auto-derived when year is set.
        CondClass = _CONDITION_CLASS[args.dataset]
        year = test_labels["year"][idx].item() if "year" in test_labels else None
        cond = CondClass(month=month, year=year)
        condition = cond.to_tensor_dict(batch_size=1, device=args.device)

        # Impute via the unified interface (uses PosteriorVelocityModelWrapper
        # in PROJECT mode + InpaintingOperator internally).
        with torch.autocast(
            device_type=model.device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            imputed_real = model.impute(
                observed_real=original_real,
                mask_real=mask,
                condition=condition,
                num_samples=args.num_samples,
                num_step=args.num_steps,
                cfg_scale=args.cfg_scale,
                resample_steps=args.resample_steps,
                resample_t_threshold=args.resample_t_threshold,
                time_grid_mode=args.time_grid_mode,
                time_grid_gamma=args.time_grid_gamma,
            )
        imputed_real = imputed_real.float().cpu()

        # Evaluate in real temporal space
        metrics = evaluate_imputation(original_real, imputed_real, mask, baseline_real)
        metrics["sample_idx"] = idx
        metrics["month"] = month
        all_metrics.append(metrics)

        # Save example figure for selected indices
        if idx in example_indices:
            fig = plot_time_series_comparison_advanced(
                generated=imputed_real,
                real=original_real.unsqueeze(0),
                output_path=os.path.join(
                    figures_dir, f"imputation_example_{idx:03d}.png"
                ),
                title=(
                    f"Imputation Example {idx} "
                    f"(month={month}, {args.imputation_type}, "
                    f"missing={args.missing_rate:.0%})"
                ),
                mask=mask,
                overlap_generated_label="Imputed Samples",
                ymin=-0.25,
                ymax=0.25,
            )
            plt.close(fig)
            plot_imp_example(
                original_real=original_real,
                imputed_samples_real=imputed_real,
                baseline_real=baseline_real,
                output_path=os.path.join(
                    figures_dir, f"imputation_example_advanced_{idx:03d}.png"
                ),
                title=(
                    f"Imputation Example {idx} "
                    f"(month={month}, {args.imputation_type}, "
                    f"missing={args.missing_rate:.0%})"
                ),
                ymin=-0.25,
                ymax=0.25,
            )

    # Aggregate results
    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in [
            "mse",
            "mae",
            "rmse",
            "mse_baseline",
            "mae_baseline",
            "rmse_baseline",
            "uncertainty",
            "crps",
            "crps_baseline",
            "peak_load_error",
            "peak_load_error_baseline",
            "sym_quantile_error_99",
            "sym_quantile_error_99_baseline",
            "pearsonr_diff",
            "pearsonr_diff_baseline",
        ]
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "imputation_results.pt")
    torch.save(
        {
            "metrics": all_metrics,
            "avg_metrics": avg_metrics,
            "args": vars(args),
        },
        results_path,
    )

    # Save metrics as JSON
    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "avg_metrics": avg_metrics,
                "args": vars(args),
                "per_series_metrics": all_metrics,
            },
            f,
            indent=2,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("Imputation Results Summary")
    print("=" * 60)
    print(f"Imputation type: {args.imputation_type}")
    print(f"Missing rate: {args.missing_rate:.1%}")
    print(f"Number of test series: {len(test_profiles)}")
    print(f"Samples per series: {args.num_samples}")
    print("-" * 60)
    print(f"{'Metric':<25} {'Flow Imp':<15} {'Baseline':<15}")
    print("-" * 60)
    print(
        f"{'MSE':<25} {avg_metrics['mse']:<15.6f} {avg_metrics['mse_baseline']:<15.6f}"
    )
    print(
        f"{'MAE':<25} {avg_metrics['mae']:<15.6f} {avg_metrics['mae_baseline']:<15.6f}"
    )
    print(
        f"{'RMSE':<25} {avg_metrics['rmse']:<15.6f} {avg_metrics['rmse_baseline']:<15.6f}"
    )
    print(
        f"{'CRPS':<25} {avg_metrics['crps']:<15.6f} {avg_metrics['crps_baseline']:<15.6f}"
    )
    print(
        f"{'PeakLoadError':<25} {avg_metrics['peak_load_error']:<15.6f} "
        f"{avg_metrics['peak_load_error_baseline']:<15.6f}"
    )
    print(
        f"{'SymQuantileErr_99':<25} {avg_metrics['sym_quantile_error_99']:<15.6f} "
        f"{avg_metrics['sym_quantile_error_99_baseline']:<15.6f}"
    )
    print(
        f"{'PearsonR Diff':<25} {avg_metrics['pearsonr_diff']:<15.6f} "
        f"{avg_metrics['pearsonr_diff_baseline']:<15.6f}"
    )
    print("-" * 60)
    print(f"Uncertainty: {avg_metrics['uncertainty']:.6f}")
    print("=" * 60)
    print(f"\nResults saved to {results_path}")
    print(f"Metrics saved to {json_path}")
    print(f"Example figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
