"""
Demonstrate super-resolution using Flow Matching model on WPuQ Heat Pump data.

This showcase script demonstrates how to use a trained Flow Matching model
for time series super-resolution (upsampling from low to high resolution).

Usage:
    uv run python scripts/showcase/super_resolution_demo.py --checkpoint path/to/checkpoint.ckpt --scale_factor 4

Example:
    # 4x super-resolution (e.g., 1h -> 15min)
    uv run python scripts/showcase/super_resolution_demo.py --checkpoint checkpoints/flow_001/last.ckpt --scale_factor 4

    # 2x super-resolution (e.g., 30min -> 15min)
    uv run python scripts/showcase/super_resolution_demo.py --checkpoint checkpoints/flow_001/last.ckpt --scale_factor 2
"""

import argparse
import calendar
import json
import os

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from smartmeterfm.data_modules.heat_pump import WPuQ
from smartmeterfm.data_modules.lcl_electricity import LCLElectricity
from smartmeterfm.data_modules.wpuq_household import WPuQHousehold
from smartmeterfm.utils.configuration import DataConfig


def downsample(data: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Downsample time series by averaging.

    Args:
        data: High-resolution time series [batch, seq_len, channels] or [seq_len, channels].
        scale_factor: Factor to downsample by.

    Returns:
        Low-resolution time series.
    """
    if data.dim() == 2:
        data = data.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # [batch, seq_len, channels] -> [batch, channels, seq_len]
    data = data.permute(0, 2, 1)

    # Average pooling
    data_lr = F.avg_pool1d(data, kernel_size=scale_factor, stride=scale_factor)

    # [batch, channels, seq_len_lr] -> [batch, seq_len_lr, channels]
    data_lr = data_lr.permute(0, 2, 1)

    if squeeze_output:
        data_lr = data_lr.squeeze(0)

    return data_lr


def upsample_linear(data: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Upsample time series using linear interpolation (baseline).

    Args:
        data: Low-resolution time series.
        scale_factor: Factor to upsample by.

    Returns:
        Upsampled time series.
    """
    if data.dim() == 2:
        data = data.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # [batch, seq_len, channels] -> [batch, channels, seq_len]
    data = data.permute(0, 2, 1)

    # Linear interpolation to target size (handles non-divisible lengths)
    target_len = data.shape[2] * scale_factor
    data_hr = F.interpolate(data, size=target_len, mode="linear", align_corners=False)

    # [batch, channels, seq_len_hr] -> [batch, seq_len_hr, channels]
    data_hr = data_hr.permute(0, 2, 1)

    if squeeze_output:
        data_hr = data_hr.squeeze(0)

    return data_hr


def super_resolve_with_flow(
    model,
    low_res_real: torch.Tensor,
    scale_factor: int,
    condition: dict,
    target_seq_len: int,
    patch_size: int,
    num_samples: int = 10,
    num_steps: int = 100,
    device: str = "cuda",
) -> torch.Tensor:
    """Perform super-resolution using Flow Matching with posterior sampling.

    The model operates in patchified space [B, patch_size, num_patches], but
    the SR projection (downsampling constraint) is applied in the actual
    temporal domain [B, 1, seq_length].

    Args:
        model: Trained FlowModelPL model.
        low_res_real: Low-resolution time series in real space [lr_seq_len].
        scale_factor: Upsampling factor.
        condition: Conditioning information.
        target_seq_len: Original high-resolution sequence length (real space).
        patch_size: Patch size used for vectorization.
        num_samples: Number of SR samples to generate.
        num_steps: Number of ODE integration steps.
        device: Device to use.

    Returns:
        Super-resolved samples in real space [num_samples, target_seq_len].
    """
    model.eval()
    model.to(device)

    # Model expects [B, num_patches, patch_size] (stored as [N, L, C])
    num_patches = target_seq_len // patch_size
    patch_shape = (1, num_patches, patch_size)

    low_res_expanded = low_res_real.unsqueeze(0).to(device)  # [1, lr_seq_len]

    all_sr = []

    for _ in tqdm(range(num_samples), desc="Generating SR samples"):
        with torch.no_grad():
            # Sample from prior in patchified space
            x_t = torch.randn(*patch_shape, device=device)
            dt = 1.0 / num_steps

            for step in range(num_steps):
                t = torch.full((1,), step * dt, device=device)

                # Get velocity prediction (model operates in patch space)
                velocity = model.model(x_t, t, y=condition)
                x_t = x_t + velocity * dt

                # Project in real temporal space
                # Unpatchify: [B, C, L] -> [B, L*C]
                x_t_real = rearrange(x_t, "b l c -> b (l c)")

                # Downsample in real space and compare to low-res input
                x_t_real_3d = x_t_real.unsqueeze(1)  # [B, 1, seq_len]
                x_t_down = F.avg_pool1d(
                    x_t_real_3d, kernel_size=scale_factor, stride=scale_factor
                ).squeeze(1)  # [B, lr_seq_len]

                # Compute correction in real space
                error = low_res_expanded - x_t_down  # [B, lr_seq_len]
                error_3d = error.unsqueeze(1)  # [B, 1, lr_seq_len]
                correction = F.interpolate(
                    error_3d, size=target_seq_len, mode="nearest"
                ).squeeze(1)  # [B, seq_len]

                # Apply correction with decreasing strength
                correction_strength = max(0, 1 - step / num_steps)
                x_t_real = x_t_real + correction_strength * correction

                # Re-patchify: [B, seq_len] -> [B, C, L]
                x_t = rearrange(x_t_real, "b (l c) -> b l c", c=patch_size)

            # Store result in real space
            x_t_real = rearrange(x_t, "b l c -> b (l c)")
            all_sr.append(x_t_real.cpu())

    return torch.cat(all_sr, dim=0)


def evaluate_super_resolution(
    original_hr: torch.Tensor,
    sr_samples: torch.Tensor,
    baseline_hr: torch.Tensor,
) -> dict:
    """Evaluate super-resolution quality.

    Args:
        original_hr: Original high-resolution time series.
        sr_samples: Super-resolved samples [num_samples, seq_len, channels].
        baseline_hr: Baseline (linear interpolation) high-resolution.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Mean SR (average of samples)
    mean_sr = sr_samples.mean(dim=0)

    # MSE
    mse_sr = ((original_hr - mean_sr) ** 2).mean().item()
    mse_baseline = ((original_hr - baseline_hr) ** 2).mean().item()

    # MAE
    mae_sr = (original_hr - mean_sr).abs().mean().item()
    mae_baseline = (original_hr - baseline_hr).abs().mean().item()

    # RMSE
    rmse_sr = mse_sr**0.5
    rmse_baseline = mse_baseline**0.5

    # Uncertainty
    uncertainty = sr_samples.std(dim=0).mean().item()

    # Improvement over baseline
    improvement_mse = (
        (mse_baseline - mse_sr) / mse_baseline * 100 if mse_baseline > 0 else 0
    )

    return {
        "mse_sr": mse_sr,
        "mse_baseline": mse_baseline,
        "mae_sr": mae_sr,
        "mae_baseline": mae_baseline,
        "rmse_sr": rmse_sr,
        "rmse_baseline": rmse_baseline,
        "uncertainty": uncertainty,
        "improvement_mse_pct": improvement_mse,
    }


def plot_sr_example(
    original_real: torch.Tensor,
    sr_samples_real: torch.Tensor,
    baseline_real: torch.Tensor,
    output_path: str,
    title: str = "Super-Resolution Example",
):
    """Plot a single super-resolution example with original, SR, and baseline.

    Args:
        original_real: Original HR time series [seq_length].
        sr_samples_real: SR samples [num_samples, seq_length].
        baseline_real: Baseline (linear interp) HR series [seq_length].
        output_path: Path to save the figure.
        title: Figure title.
    """
    with plt.style.context("smartmeterfm.utils.article_compatible"):
        fig, ax = plt.subplots(figsize=(12, 4))

        t = np.arange(original_real.shape[0])
        sr_mean = sr_samples_real.mean(dim=0).numpy()
        sr_std = sr_samples_real.std(dim=0).numpy()

        ax.plot(
            t,
            original_real.numpy(),
            label="Original HR",
            color="tab:blue",
            linewidth=1.5,
        )
        ax.plot(
            t,
            sr_mean,
            label="SR Mean",
            color="tab:orange",
            linewidth=1.5,
        )
        ax.fill_between(
            t,
            sr_mean - sr_std,
            sr_mean + sr_std,
            color="tab:orange",
            alpha=0.2,
            label="SR \u00b11\u03c3",
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
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate super-resolution using Flow Matching model"
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
        "--scale_factor",
        type=int,
        choices=[2, 3, 4, 6, 8],
        default=2,
        help="Upsampling factor (default: 2). Must evenly divide the sequence length.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of SR samples per time series (default: 10)",
    )
    parser.add_argument(
        "--num_test_series",
        type=int,
        default=100,
        help="Number of test time series to process (default: 100)",
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
        default="results/super_resolution",
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

    print(f"Scale factor: {args.scale_factor}x")
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

    # Run super-resolution
    all_metrics = []

    print(
        f"\nRunning {args.scale_factor}x super-resolution on {len(test_profiles)} time series..."
    )
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    example_indices = set(range(min(args.num_example_figures, len(test_profiles))))

    for idx in tqdm(range(len(test_profiles)), desc="Super-resolving"):
        original_patched = test_profiles[idx]  # [patch_size, num_patches] (patchified)
        month = test_labels["month"][idx].item()

        # Unpatchify to real temporal space: [C, L] -> [C*L]
        original_real = rearrange(original_patched, "l c -> (l c)")  # [seq_length]
        seq_length = original_real.shape[0]

        # Downsample in real temporal space
        lr_real = F.avg_pool1d(
            original_real.view(1, 1, -1),
            kernel_size=args.scale_factor,
            stride=args.scale_factor,
        ).squeeze()  # [lr_seq_len]

        # Baseline: linear interpolation in real space
        baseline_real = F.interpolate(
            lr_real.view(1, 1, -1), size=seq_length, mode="linear", align_corners=False
        ).squeeze()  # [seq_length]

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

        # Super-resolve (model in patch space, projection in real space)
        sr_samples_real = super_resolve_with_flow(
            model,
            lr_real,
            args.scale_factor,
            condition,
            target_seq_len=seq_length,
            patch_size=patch_size,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=args.device,
        )

        # Evaluate in real temporal space
        metrics = evaluate_super_resolution(
            original_real, sr_samples_real, baseline_real
        )
        metrics["sample_idx"] = idx
        metrics["month"] = month
        all_metrics.append(metrics)

        # Save example figure for selected indices
        if idx in example_indices:
            plot_sr_example(
                original_real=original_real,
                sr_samples_real=sr_samples_real,
                baseline_real=baseline_real,
                output_path=os.path.join(figures_dir, f"sr_example_{idx:03d}.png"),
                title=(
                    f"Super-Resolution Example {idx} "
                    f"(month={month}, {args.scale_factor}x)"
                ),
            )

    # Aggregate results
    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in [
            "mse_sr",
            "mse_baseline",
            "mae_sr",
            "mae_baseline",
            "rmse_sr",
            "rmse_baseline",
            "uncertainty",
            "improvement_mse_pct",
        ]
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"sr_{args.scale_factor}x_results.pt")
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
    print(f"Super-Resolution Results Summary ({args.scale_factor}x)")
    print("=" * 60)
    print(f"Number of test series: {len(test_profiles)}")
    print(f"Samples per series: {args.num_samples}")
    print("-" * 60)
    print(f"{'Metric':<25} {'Flow SR':<15} {'Baseline':<15}")
    print("-" * 60)
    print(
        f"{'MSE':<25} {avg_metrics['mse_sr']:<15.6f} {avg_metrics['mse_baseline']:<15.6f}"
    )
    print(
        f"{'MAE':<25} {avg_metrics['mae_sr']:<15.6f} {avg_metrics['mae_baseline']:<15.6f}"
    )
    print(
        f"{'RMSE':<25} {avg_metrics['rmse_sr']:<15.6f} {avg_metrics['rmse_baseline']:<15.6f}"
    )
    print("-" * 60)
    print(f"Uncertainty: {avg_metrics['uncertainty']:.6f}")
    print(f"Improvement over baseline: {avg_metrics['improvement_mse_pct']:.1f}% (MSE)")
    print("=" * 60)
    print(f"\nResults saved to {results_path}")
    print(f"Metrics saved to {json_path}")
    print(f"Example figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
