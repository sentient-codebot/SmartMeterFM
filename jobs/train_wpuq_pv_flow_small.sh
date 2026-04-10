#!/bin/bash

#SBATCH --job-name="wpuq-pv-flow-small"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/SmartMeterFM

# init venv
source .venv/bin/activate
uv sync

python scripts/showcase/train_flow.py \
    --config configs/showcase/wpuq_pv_flow_small.toml \
    --time_id wpuq_pv_50k

echo "**************** [wpuq_pv_50k] training completed. **************************"
