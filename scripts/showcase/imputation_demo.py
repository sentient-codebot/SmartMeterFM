"""
Demonstrate imputation using Flow Matching model on WPuQ Heat Pump data.

This showcase script demonstrates how to use a trained Flow Matching model
for time series imputation (filling in missing values) using posterior sampling.

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
import calendar
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from tqdm import tqdm

from smartmeterfm.data_modules.heat_pump import WPuQ
from smartmeterfm.data_modules.lcl_electricity import LCLElectricity
from smartmeterfm.data_modules.wpuq_household import WPuQHousehold
from smartmeterfm.utils.configuration import DataConfig
from smartmeterfm.utils.plot import plot_time_series_comparison_advanced


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


def impute_with_flow(
    model,
    observed_data: torch.Tensor,
    mask: torch.Tensor,
    condition: dict,
    patch_size: int,
    num_samples: int = 10,
    num_steps: int = 100,
    device: str = "cuda",
) -> torch.Tensor:
    """Perform imputation using Flow Matching with posterior sampling.

    The model operates in patchified space [B, seq_patches, patch_size], but
    the projection (keeping observed values fixed) is applied in the actual
    temporal domain [B, 1, seq_length] to ensure per-timestep masking.

    Args:
        model: Trained FlowModelPL model.
        observed_data: The observed time series in real space [seq_length].
        mask: Binary mask in real space [seq_length] (1 = observed, 0 = missing).
        condition: Conditioning information (e.g., month).
        patch_size: Patch size used for vectorization.
        num_samples: Number of imputation samples to generate.
        num_steps: Number of ODE integration steps.
        device: Device to use.

    Returns:
        Imputed samples in real space [num_samples, seq_length].
    """
    model.eval()
    model.to(device)

    seq_length = observed_data.shape[0]
    num_patches = seq_length // patch_size
    # Model expects [B, num_patches, patch_size] (stored as [N, L, C])
    patch_shape = (1, num_patches, patch_size)

    observed_real = observed_data.unsqueeze(0).to(device)  # [1, seq_length]
    mask_real = mask.unsqueeze(0).to(device)  # [1, seq_length]

    all_imputed = []

    for _ in tqdm(range(num_samples), desc="Generating imputations"):
        with torch.no_grad():
            # Sample from prior in patchified space
            x_t = torch.randn(*patch_shape, device=device)
            dt = 1.0 / num_steps

            for step in range(num_steps):
                t = torch.full((1,), step * dt, device=device)

                # Get velocity prediction (model operates in patch space)
                velocity = model.model(x_t, t, y=condition)
                x_t = x_t + velocity * dt

                # Project in real temporal space: keep observed values fixed
                # Unpatchify: [B, L, C] -> [B, L*C]
                x_t_real = rearrange(x_t, "b l c -> b (l c)")
                x_t_real = x_t_real * (1 - mask_real) + observed_real * mask_real
                # Re-patchify: [B, L*C] -> [B, L, C]
                x_t = rearrange(x_t_real, "b (l c) -> b l c", c=patch_size)

            # Store result in real space
            x_t_real = rearrange(x_t, "b l c -> b (l c)")
            all_imputed.append(x_t_real.cpu())

    return torch.cat(all_imputed, dim=0)


def evaluate_imputation(
    original: torch.Tensor,
    imputed: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Evaluate imputation quality.

    Args:
        original: Original complete time series.
        imputed: Imputed samples [num_samples, *shape].
        mask: Binary mask (1 = observed, 0 = missing).

    Returns:
        Dictionary of evaluation metrics.
    """
    # Get missing positions
    missing_mask = (1 - mask).bool()

    # Mean imputation (average of samples)
    mean_imputed = imputed.mean(dim=0)

    # Extract values at missing positions
    original_missing = original.flatten()[missing_mask.flatten()]
    mean_imputed_missing = mean_imputed.flatten()[missing_mask.flatten()]

    # Calculate metrics
    mse = ((original_missing - mean_imputed_missing) ** 2).mean().item()
    mae = (original_missing - mean_imputed_missing).abs().mean().item()
    rmse = mse**0.5

    # Uncertainty (std of imputed values at missing positions)
    imputed_at_missing = imputed[:, missing_mask.expand_as(imputed[0])].view(
        imputed.shape[0], -1
    )
    uncertainty = imputed_at_missing.std(dim=0).mean().item()

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "uncertainty": uncertainty,
        "num_missing": missing_mask.sum().item(),
        "missing_rate_actual": missing_mask.float().mean().item(),
    }


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

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    from smartmeterfm.models.flow import FlowModelPL

    model = FlowModelPL.load_from_checkpoint(
        args.checkpoint, map_location=args.device, weights_only=False
    )

    # Load test data
    dataset_name = args.dataset
    resolution_map = {
        "wpuq": "15min",
        "wpuq_household": "15min",
        "lcl_electricity": "30min",
    }
    print(f"\nLoading {dataset_name} test data...")
    data_config = DataConfig(
        dataset=dataset_name,
        root=args.data_root,
        load=True,
        normalize=True,
        normalize_method="meanstd",
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
    patch_size = data_config.vectorize_window_size

    # Run imputation
    all_metrics = []

    print(f"\nRunning imputation on {len(test_profiles)} time series...")
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    example_indices = set(range(min(args.num_example_figures, len(test_profiles))))

    for idx in tqdm(range(len(test_profiles)), desc="Imputing"):
        original_patched = test_profiles[idx]  # [patch_size, num_patches] (patchified)
        month = test_labels["month"][idx].item()

        # Unpatchify to real temporal space: [C, L] -> [C*L]
        original_real = rearrange(original_patched, "l c -> (l c)")  # [seq_length]
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

        # Create observed data in real space (with missing values set to 0)
        observed_real = original_real.clone()
        observed_real = observed_real * mask

        # Create condition with position/masking info when year is available
        condition = {
            "month": torch.tensor([[month]], dtype=torch.long, device=args.device)
        }
        if "year" in test_labels:
            year = test_labels["year"][idx].item()
            weekday, days = calendar.monthrange(year, month + 1)
            condition["year"] = torch.tensor(
                [[year]], dtype=torch.long, device=args.device
            )
            condition["first_day_of_week"] = torch.tensor(
                [[weekday]], dtype=torch.long, device=args.device
            )
            condition["month_length"] = torch.tensor(
                [[days - 28]], dtype=torch.long, device=args.device
            )

        # Impute (operates in patch space internally, projects in real space)
        imputed_real = impute_with_flow(
            model,
            observed_real,
            mask,
            condition,
            patch_size=patch_size,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=args.device,
        )

        # Evaluate in real temporal space
        metrics = evaluate_imputation(original_real, imputed_real, mask)
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
            )
            plt.close(fig)

    # Aggregate results
    avg_metrics = {
        "mse": sum(m["mse"] for m in all_metrics) / len(all_metrics),
        "mae": sum(m["mae"] for m in all_metrics) / len(all_metrics),
        "rmse": sum(m["rmse"] for m in all_metrics) / len(all_metrics),
        "uncertainty": sum(m["uncertainty"] for m in all_metrics) / len(all_metrics),
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
    print("\n" + "=" * 50)
    print("Imputation Results Summary")
    print("=" * 50)
    print(f"Imputation type: {args.imputation_type}")
    print(f"Missing rate: {args.missing_rate:.1%}")
    print(f"Number of test series: {len(test_profiles)}")
    print(f"Samples per series: {args.num_samples}")
    print("-" * 50)
    print(f"Average MSE:         {avg_metrics['mse']:.6f}")
    print(f"Average MAE:         {avg_metrics['mae']:.6f}")
    print(f"Average RMSE:        {avg_metrics['rmse']:.6f}")
    print(f"Average Uncertainty: {avg_metrics['uncertainty']:.6f}")
    print("=" * 50)
    print(f"\nResults saved to {results_path}")
    print(f"Metrics saved to {json_path}")
    print(f"Example figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
