#!/bin/bash

#SBATCH --job-name="lcl-sr-0507"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

# 4x super-resolution evaluation for LCL-0507 (2h coarse → 30min target).
# 2x is intentionally skipped (per project eval defaults — too simple to
# discriminate between models).

module load 2025

cd ~/projects/SmartMeterFM

uv sync

TIMEID="LCL-0507"
CKPT="checkpoints/${TIMEID}/last.ckpt"

uv run python scripts/showcase/super_resolution_demo.py \
    --checkpoint $CKPT \
    --dataset lcl_electricity \
    --data_root data/lcl_electricity/ \
    --scale_factor 4 \
    --num_samples 10 \
    --num_test_series 100 \
    --num_steps 200 \
    --output_dir results/super_resolution/${TIMEID}/4x

echo "**************** [${TIMEID}] 4x super-resolution completed. **************************"
