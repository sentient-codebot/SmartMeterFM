#!/bin/bash

#SBATCH --job-name="lcl-flow-0421-01"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/SmartMeterFM

# init venv
source .venv/bin/activate
uv sync

python scripts/showcase/train_flow.py \
    --config configs/showcase/lcl_0421_01.toml \
    --time_id LCL-0421-01 \
    --seed 42

echo "**************** [LCL-0421-01] training completed. **************************"
