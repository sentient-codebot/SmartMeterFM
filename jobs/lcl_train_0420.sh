#!/bin/bash

#SBATCH --job-name="lcl-flow-0420"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/SmartMeterFM

# init venv
source .venv/bin/activate
uv sync

python scripts/showcase/train_flow.py \
    --config configs/showcase/lcl_electricity_flow_monthly.toml \
    --time_id LCL-0420

echo "**************** [LCL-0420] training completed. **************************"
