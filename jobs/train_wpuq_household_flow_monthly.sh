#!/bin/bash

#SBATCH --job-name="wpuq-household-flow-monthly"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/SmartMeterFM

# init venv
source .venv/bin/activate
uv sync

python scripts/showcase/train_flow.py \
    --config configs/showcase/wpuq_household_flow_monthly.toml \
    --time_id wpuq_household_50k

echo "**************** [wpuq_household_50k] training completed. **************************"
