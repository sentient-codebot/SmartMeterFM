#!/bin/bash

#SBATCH --job-name="lcl-eval-samples"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/SmartMeterFM

# init venv
source .venv/bin/activate
uv sync

# Evaluate generated samples against real test data
python scripts/showcase/evaluate_samples.py \
    --samples_dir results/generated/LCL-0414 \
    --config configs/showcase/lcl_electricity_flow_monthly.toml \
    --output_dir results/eval/LCL-0414

echo "**************** [LCL-0414] evaluation completed. **************************"

# Plot generated vs real
python scripts/showcase/plot_generated_samples.py \
    --samples_dir results/generated/LCL-0414 \
    --config configs/showcase/lcl_electricity_flow_monthly.toml \
    --output_dir results/figures/LCL-0414 \
    --eval_metrics results/eval/LCL-0414/eval_metrics.json

echo "**************** [LCL-0414] plotting completed. **************************"
