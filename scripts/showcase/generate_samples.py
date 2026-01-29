"""
Generate conditional samples using trained Flow Matching or VAE models.

This showcase script demonstrates how to generate energy profiles conditioned
on specific attributes (e.g., month) using trained models.

Usage:
    uv run python scripts/showcase/generate_samples.py --model_type flow --checkpoint path/to/checkpoint.ckpt --num_samples 1000

Example:
    # Generate samples with Flow model
    uv run python scripts/showcase/generate_samples.py --model_type flow --checkpoint checkpoints/flow_001/last.ckpt --num_samples 1000 --output_dir samples/flow_001

    # Generate samples for specific months
    uv run python scripts/showcase/generate_samples.py --model_type flow --checkpoint checkpoints/flow_001/last.ckpt --num_samples 500 --months 0 5 11

    # Generate with VAE model
    uv run python scripts/showcase/generate_samples.py --model_type vae --checkpoint checkpoints/vae_001/last.ckpt --num_samples 1000
"""

import argparse
import os

import torch
from tqdm import tqdm


def generate_flow_samples(
    checkpoint_path: str,
    num_samples: int,
    months: list[int],
    num_steps: int = 100,
    batch_size: int = 256,
    cfg_scale: float = 1.0,
    device: str = "cuda",
) -> dict[int, torch.Tensor]:
    """Generate samples using a trained Flow Matching model.

    Args:
        checkpoint_path: Path to the model checkpoint.
        num_samples: Number of samples to generate per month.
        months: List of months (0-11) to generate samples for.
        num_steps: Number of ODE integration steps.
        batch_size: Batch size for generation.
        cfg_scale: Classifier-free guidance scale.
        device: Device to use for generation.

    Returns:
        Dictionary mapping month index to generated samples tensor.
    """
    from smartmeterfm.models.flow import FlowModelPL

    print(f"Loading Flow model from {checkpoint_path}...")
    model = FlowModelPL.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    # Use EMA weights if available
    if hasattr(model, "ema") and model.ema is not None:
        model.ema.copy_params_from_ema_to_model()
        print("Using EMA weights for generation")

    samples_by_month = {}

    for month in months:
        print(f"\nGenerating samples for month {month + 1}/12...")
        all_samples = []

        # Calculate batch sizes
        num_batches = (num_samples + batch_size - 1) // batch_size
        remaining = num_samples

        for batch_idx in tqdm(range(num_batches), desc=f"Month {month + 1}"):
            curr_batch_size = min(batch_size, remaining)
            remaining -= curr_batch_size

            # Create condition tensor
            condition = {
                "month": torch.full((curr_batch_size, 1), month, dtype=torch.long, device=device)
            }

            # Generate samples
            with torch.no_grad():
                # Get sample shape from model
                sample_shape = (curr_batch_size, model.hparams.get("seq_length", 6),
                              model.hparams.get("num_in_channel", 16))

                # Sample from prior
                x_0 = torch.randn(sample_shape, device=device)

                # Use the model's sample method if available
                if hasattr(model, "sample"):
                    x_1 = model.sample(
                        x_0=x_0,
                        condition=condition,
                        num_steps=num_steps,
                        cfg_scale=cfg_scale,
                    )
                else:
                    # Manual ODE integration using Euler method
                    x_t = x_0
                    dt = 1.0 / num_steps
                    for step in range(num_steps):
                        t = torch.full((curr_batch_size,), step * dt, device=device)
                        velocity = model.model(x_t, t, y=condition)
                        x_t = x_t + velocity * dt
                    x_1 = x_t

                all_samples.append(x_1.cpu())

        samples_by_month[month] = torch.cat(all_samples, dim=0)
        print(f"Generated {samples_by_month[month].shape[0]} samples for month {month + 1}")

    return samples_by_month


def generate_vae_samples(
    checkpoint_path: str,
    num_samples: int,
    months: list[int],
    batch_size: int = 256,
    device: str = "cuda",
) -> dict[int, torch.Tensor]:
    """Generate samples using a trained VAE model.

    Args:
        checkpoint_path: Path to the model checkpoint.
        num_samples: Number of samples to generate per month.
        months: List of months (0-11) to generate samples for.
        batch_size: Batch size for generation.
        device: Device to use for generation.

    Returns:
        Dictionary mapping month index to generated samples tensor.
    """
    from smartmeterfm.models.baselines.cond_gen.vae import VAEModelPL

    print(f"Loading VAE model from {checkpoint_path}...")
    model = VAEModelPL.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)

    samples_by_month = {}

    for month in months:
        print(f"\nGenerating samples for month {month + 1}/12...")
        all_samples = []

        num_batches = (num_samples + batch_size - 1) // batch_size
        remaining = num_samples

        for batch_idx in tqdm(range(num_batches), desc=f"Month {month + 1}"):
            curr_batch_size = min(batch_size, remaining)
            remaining -= curr_batch_size

            # Create condition tensor
            condition = {
                "month": torch.full((curr_batch_size, 1), month, dtype=torch.long, device=device)
            }

            with torch.no_grad():
                # Sample from the VAE
                if hasattr(model, "sample"):
                    x = model.sample(condition=condition, num_samples=curr_batch_size)
                else:
                    # Manual sampling from prior
                    z = torch.randn(curr_batch_size, model.vae.latent_dim, device=device)
                    x = model.vae.decode(z, condition)

                all_samples.append(x.cpu())

        samples_by_month[month] = torch.cat(all_samples, dim=0)
        print(f"Generated {samples_by_month[month].shape[0]} samples for month {month + 1}")

    return samples_by_month


def save_samples(samples_by_month: dict[int, torch.Tensor], output_dir: str):
    """Save generated samples to disk.

    Args:
        samples_by_month: Dictionary mapping month to samples tensor.
        output_dir: Directory to save samples.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save all samples together
    all_samples_path = os.path.join(output_dir, "generated_samples.pt")
    torch.save(samples_by_month, all_samples_path)
    print(f"\nSaved all samples to {all_samples_path}")

    # Save per-month samples
    for month, samples in samples_by_month.items():
        month_path = os.path.join(output_dir, f"month_{month:02d}_samples.pt")
        torch.save(samples, month_path)

    # Save summary
    summary_path = os.path.join(output_dir, "generation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Generation Summary\n")
        f.write("=" * 40 + "\n\n")
        total_samples = 0
        for month, samples in sorted(samples_by_month.items()):
            f.write(f"Month {month + 1:2d}: {samples.shape[0]:6d} samples, shape: {list(samples.shape)}\n")
            total_samples += samples.shape[0]
        f.write(f"\nTotal samples: {total_samples}\n")

    print(f"Saved generation summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate conditional samples using trained models"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["flow", "vae"],
        required=True,
        help="Type of model to use for generation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        "-N",
        type=int,
        default=1000,
        help="Number of samples to generate per condition (default: 1000)",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=list(range(12)),
        help="Months to generate samples for (0-11, default: all months)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of ODE integration steps for Flow model (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for generation (default: 256)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale for Flow model (default: 1.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="samples/generated",
        help="Directory to save generated samples (default: samples/generated)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for generation",
    )
    args = parser.parse_args()

    # Validate months
    for month in args.months:
        if month < 0 or month > 11:
            raise ValueError(f"Invalid month: {month}. Must be 0-11.")

    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of samples per month: {args.num_samples}")
    print(f"Months: {[m + 1 for m in args.months]}")
    print(f"Device: {args.device}")

    # Generate samples
    if args.model_type == "flow":
        samples_by_month = generate_flow_samples(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            months=args.months,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            cfg_scale=args.cfg_scale,
            device=args.device,
        )
    else:  # vae
        samples_by_month = generate_vae_samples(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            months=args.months,
            batch_size=args.batch_size,
            device=args.device,
        )

    # Save samples
    save_samples(samples_by_month, args.output_dir)

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
