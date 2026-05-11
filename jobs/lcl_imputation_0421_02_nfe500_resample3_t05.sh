#!/bin/bash

#SBATCH --job-name="lcl-imput-0421-02-nfe500-K3-t05"
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

TIMEID="LCL-0421-02"
CKPT="checkpoints/${TIMEID}/last.ckpt"

# MNAR consecutive block imputation (20% missing), NFE=500, K=3 resample, t_threshold=0.5
python scripts/showcase/imputation_demo.py \
    --checkpoint $CKPT \
    --dataset lcl_electricity \
    --data_root data/lcl_electricity/ \
    --imputation_type mnar_consecutive \
    --missing_rate 0.2 \
    --num_samples 10 \
    --num_test_series 100 \
    --num_steps 500 \
    --resample_steps 3 \
    --resample_t_threshold 0.5 \
    --time_grid_mode uniform \
    --output_dir results/imputation/${TIMEID}/mnar_20_nfe500_resample3_t05

echo "**************** [${TIMEID}] MNAR imputation NFE=500 K=3 t=0.5 completed. **************************"
