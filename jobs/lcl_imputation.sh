#!/bin/bash

#SBATCH --job-name="lcl-imputation"
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

CKPT="checkpoints/LCL-0414/last-v3.ckpt"

# MCAR imputation (20% missing)
python scripts/showcase/imputation_demo.py \
    --checkpoint $CKPT \
    --dataset lcl_electricity \
    --data_root data/lcl_electricity/ \
    --imputation_type mcar \
    --missing_rate 0.2 \
    --num_samples 10 \
    --num_test_series 100 \
    --num_steps 100 \
    --output_dir results/imputation/LCL-0414/mcar_20

echo "**************** [LCL-0414] MCAR imputation completed. **************************"

# MNAR consecutive block imputation (20% missing)
python scripts/showcase/imputation_demo.py \
    --checkpoint $CKPT \
    --dataset lcl_electricity \
    --data_root data/lcl_electricity/ \
    --imputation_type mnar_consecutive \
    --missing_rate 0.2 \
    --num_samples 10 \
    --num_test_series 100 \
    --num_steps 100 \
    --output_dir results/imputation/LCL-0414/mnar_20

echo "**************** [LCL-0414] MNAR imputation completed. **************************"
