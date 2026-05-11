#!/bin/bash

#SBATCH --job-name="lcl-imput-0507-nfe227-K3"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

# Iso-compute K=3 imputation eval for LCL-0507.
# Mirrors jobs/lcl_imputation_0421_02_nfe227_resample3.sh — same recipe so the
# numbers are directly comparable to docs/reflow_and_posterior_sampling.md
# §2.1d (the iso-compute K=3 vs K=1 table on 0421-02).
#
# Net queries = N * (1 + 3 * tau) = 227 * (1 + 3 * 0.4) = 227 * 2.2 = 499.4 ~= 500
# Same compute budget as NFE=500 K=1, redirected from outer steps to K=3
# RePaint-style inner iterations on the t<0.4 region.

module load 2025

cd ~/projects/SmartMeterFM

uv sync

TIMEID="LCL-0507"
CKPT="checkpoints/${TIMEID}/last.ckpt"

uv run python scripts/showcase/imputation_demo.py \
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
