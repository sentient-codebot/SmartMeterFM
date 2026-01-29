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
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from smartmeterfm.data_modules.heat_pump import WPuQ
from smartmeterfm.models.measurement import get_operator
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

    # Linear interpolation
    data_hr = F.interpolate(data, scale_factor=scale_factor, mode="linear", align_corners=False)

    # [batch, channels, seq_len_hr] -> [batch, seq_len_hr, channels]
    data_hr = data_hr.permute(0, 2, 1)

    if squeeze_output:
        data_hr = data_hr.squeeze(0)

    return data_hr


def super_resolve_with_flow(
    model,
    low_res_data: torch.Tensor,
    scale_factor: int,
    condition: dict,
    num_samples: int = 10,
    num_steps: int = 100,
    device: str = "cuda",
) -> torch.Tensor:
    """Perform super-resolution using Flow Matching with posterior sampling.

    Args:
        model: Trained FlowModelPL model.
        low_res_data: Low-resolution time series.
        scale_factor: Upsampling factor.
        condition: Conditioning information.
        num_samples: Number of SR samples to generate.
        num_steps: Number of ODE integration steps.
        device: Device to use.

    Returns:
        Super-resolved samples tensor of shape [num_samples, *high_res_shape].
    """
    model.eval()
    model.to(device)

    # Get super-resolution operator
    sr_op = get_operator(name="super_resolution", scale_factor=scale_factor)

    # Target high-res shape
    hr_seq_len = low_res_data.shape[0] * scale_factor
    hr_shape = (hr_seq_len, low_res_data.shape[1])

    all_sr = []

    for _ in tqdm(range(num_samples), desc="Generating SR samples"):
        with torch.no_grad():
            # Sample from prior
            x_0 = torch.randn(1, *hr_shape, device=device)

            # ODE integration with SR constraint
            x_t = x_0
            dt = 1.0 / num_steps

            for step in range(num_steps):
                t = torch.full((1,), step * dt, device=device)

                # Get velocity prediction
                velocity = model.model(x_t, t, y=condition)

                # Update
                x_t = x_t + velocity * dt

                # Project: ensure downsampled version matches low-res input
                # This is a soft constraint using gradient-based projection
                x_t_down = downsample(x_t, scale_factor)
                low_res_expanded = low_res_data.unsqueeze(0).to(device)

                # Compute correction
                error = low_res_expanded - x_t_down
                # Distribute error back to high-res (simple nearest neighbor)
                correction = error.repeat_interleave(scale_factor, dim=1)

                # Apply correction with decreasing strength as t increases
                correction_strength = max(0, 1 - step / num_steps)
                x_t = x_t + correction_strength * correction

            all_sr.append(x_t.cpu())

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
    rmse_sr = mse_sr ** 0.5
    rmse_baseline = mse_baseline ** 0.5

    # Uncertainty
    uncertainty = sr_samples.std(dim=0).mean().item()

    # Improvement over baseline
    improvement_mse = (mse_baseline - mse_sr) / mse_baseline * 100 if mse_baseline > 0 else 0

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
        "--data_root",
        type=str,
        default="data/wpuq/",
        help="Root directory for WPuQ data",
    )
    parser.add_argument(
        "--scale_factor",
        type=int,
        choices=[2, 4, 8],
        default=4,
        help="Upsampling factor (default: 4)",
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
    model = FlowModelPL.load_from_checkpoint(args.checkpoint, map_location=args.device)

    # Load test data
    print("\nLoading WPuQ test data...")
    data_config = DataConfig(
        dataset="wpuq",
        root=args.data_root,
        load=True,
        normalize=True,
        normalize_method="meanstd",
        pit=False,
        resolution="15min",
        shuffle=False,
        vectorize=True,
        style_vectorize="patchify",
        vectorize_window_size=16,
        target_labels=["month"],
        train_season="whole_year",
        val_season="whole_year",
    )
    wpuq_data = WPuQ(data_config)

    test_profiles = wpuq_data.dataset.profile["test"][:args.num_test_series]
    test_labels = wpuq_data.dataset.label["test"]

    # Run super-resolution
    all_metrics = []

    print(f"\nRunning {args.scale_factor}x super-resolution on {len(test_profiles)} time series...")
    for idx in tqdm(range(len(test_profiles)), desc="Super-resolving"):
        original_hr = test_profiles[idx]  # [seq_len, channels]
        month = test_labels["month"][idx].item()

        # Downsample to create low-res input
        low_res = downsample(original_hr, args.scale_factor)

        # Baseline: linear interpolation
        baseline_hr = upsample_linear(low_res, args.scale_factor)

        # Create condition
        condition = {
            "month": torch.tensor([[month]], dtype=torch.long, device=args.device)
        }

        # Super-resolve with Flow model
        sr_samples = super_resolve_with_flow(
            model,
            low_res,
            args.scale_factor,
            condition,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=args.device,
        )

        # Evaluate
        metrics = evaluate_super_resolution(original_hr, sr_samples, baseline_hr)
        metrics["sample_idx"] = idx
        metrics["month"] = month
        all_metrics.append(metrics)

    # Aggregate results
    avg_metrics = {
        key: sum(m[key] for m in all_metrics) / len(all_metrics)
        for key in ["mse_sr", "mse_baseline", "mae_sr", "mae_baseline",
                    "rmse_sr", "rmse_baseline", "uncertainty", "improvement_mse_pct"]
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"sr_{args.scale_factor}x_results.pt")
    torch.save({
        "metrics": all_metrics,
        "avg_metrics": avg_metrics,
        "args": vars(args),
    }, results_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Super-Resolution Results Summary ({args.scale_factor}x)")
    print("=" * 60)
    print(f"Number of test series: {len(test_profiles)}")
    print(f"Samples per series: {args.num_samples}")
    print("-" * 60)
    print(f"{'Metric':<25} {'Flow SR':<15} {'Baseline':<15}")
    print("-" * 60)
    print(f"{'MSE':<25} {avg_metrics['mse_sr']:<15.6f} {avg_metrics['mse_baseline']:<15.6f}")
    print(f"{'MAE':<25} {avg_metrics['mae_sr']:<15.6f} {avg_metrics['mae_baseline']:<15.6f}")
    print(f"{'RMSE':<25} {avg_metrics['rmse_sr']:<15.6f} {avg_metrics['rmse_baseline']:<15.6f}")
    print("-" * 60)
    print(f"Uncertainty: {avg_metrics['uncertainty']:.6f}")
    print(f"Improvement over baseline: {avg_metrics['improvement_mse_pct']:.1f}% (MSE)")
    print("=" * 60)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
