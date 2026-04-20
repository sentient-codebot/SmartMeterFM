#!/bin/bash

#SBATCH --job-name="lcl-imput-0420-01"
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

TIMEID="LCL-0420-01"
CKPT="checkpoints/${TIMEID}/last.ckpt"

# MNAR consecutive block imputation (20% missing)
python scripts/showcase/imputation_demo.py \
    --checkpoint $CKPT \
    --dataset lcl_electricity \
    --data_root data/lcl_electricity/ \
    --imputation_type mnar_consecutive \
    --missing_rate 0.2 \
    --num_samples 10 \
    --num_test_series 100 \
    --num_steps 500 \
    --output_dir results/imputation/${TIMEID}/mnar_20

echo "**************** [${TIMEID}] MNAR imputation completed. **************************"
