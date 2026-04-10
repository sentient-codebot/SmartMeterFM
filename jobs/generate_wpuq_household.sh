#!/bin/bash

#SBATCH --job-name="generate-wpuq-household"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/SmartMeterFM

# init venv
source .venv/bin/activate
uv sync

# Generate samples from trained Flow model
# Update --checkpoint to the actual checkpoint path after training
python scripts/showcase/generate_samples.py \
    --model_type flow \
    --checkpoint checkpoints/wpuq_household_50k/last.ckpt \
    --num_samples 1000 \
    --num_steps 100 \
    --batch_size 256 \
    --cfg_scale 1.0 \
    --output_dir samples/wpuq_household

echo "**************** [wpuq_household] sample generation completed. **************************"
