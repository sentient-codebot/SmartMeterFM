#!/bin/bash

#SBATCH --job-name="lcl-imput-0421-02-nfe227-K3"
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

# Iso-compute comparison vs NFE=500 K=1 baseline.
# Net queries = N * (1 + 3 * tau) = 227 * (1 + 3 * 0.4) = 227 * 2.2 = 499.4 ~= 500
# Same compute budget as NFE=500 K=1, but spent on K=3 resampling instead of more outer steps.
python scripts/showcase/imputation_demo.py \
    --checkpoint $CKPT \
    --dataset lcl_electricity \
    --data_root data/lcl_electricity/ \
    --imputation_type mnar_consecutive \
    --missing_rate 0.2 \
    --num_samples 10 \
    --num_test_series 100 \
    --num_steps 227 \
    --resample_steps 3 \
    --resample_t_threshold 0.4 \
    --time_grid_mode uniform \
    --output_dir results/imputation/${TIMEID}/mnar_20_nfe227_resample3

echo "**************** [${TIMEID}] MNAR imputation NFE=227 K=3 (iso-compute vs NFE=500 K=1) completed. **************************"
