#!/bin/bash

#SBATCH --job-name="lcl-sr-v2"
#SBATCH --partition=gpu_a100
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

TIMEID="LCL-0414-01"
CKPT="checkpoints/${TIMEID}/last.ckpt"

# 2x super-resolution (1h → 30min)
python scripts/showcase/super_resolution_demo.py \
    --checkpoint $CKPT \
    --dataset lcl_electricity \
    --data_root data/lcl_electricity/ \
    --scale_factor 2 \
    --num_samples 10 \
    --num_test_series 100 \
    --num_steps 100 \
    --output_dir results/super_resolution/${TIMEID}/2x

echo "**************** [${TIMEID}] 2x super-resolution completed. **************************"

# 4x super-resolution (2h → 30min)
python scripts/showcase/super_resolution_demo.py \
    --checkpoint $CKPT \
    --dataset lcl_electricity \
    --data_root data/lcl_electricity/ \
    --scale_factor 4 \
    --num_samples 10 \
    --num_test_series 100 \
    --num_steps 100 \
    --output_dir results/super_resolution/${TIMEID}/4x

echo "**************** [${TIMEID}] 4x super-resolution completed. **************************"
