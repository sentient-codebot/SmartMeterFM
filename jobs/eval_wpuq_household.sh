#!/bin/bash

#SBATCH --job-name="eval-wpuq-household"
#SBATCH --partition=gpu_h100
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

# Step 1: Generate samples from trained Flow model
python scripts/showcase/generate_samples.py \
    --model_type flow \
    --checkpoint checkpoints/wpuq_household_50k/last.ckpt \
    --num_samples 1000 \
    --num_steps 100 \
    --batch_size 256 \
    --cfg_scale 1.0 \
    --output_dir samples/wpuq_household

echo "**************** [wpuq_household] sample generation completed. **************************"

# Step 2: Evaluate generated samples against real test data
python scripts/showcase/evaluate_samples.py \
    --samples_dir samples/wpuq_household \
    --config configs/showcase/wpuq_household_flow_monthly.toml \
    --output_dir results/eval/wpuq_household_50k \
    --device cuda

echo "**************** [wpuq_household_50k] evaluation completed. **************************"
