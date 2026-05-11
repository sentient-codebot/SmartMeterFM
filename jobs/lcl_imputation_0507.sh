#!/bin/bash

#SBATCH --job-name="lcl-imput-0507-nfe500-K3"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

# MNAR consecutive-block imputation for LCL-0507.
# Same posterior-sampling recipe as lcl_imputation_0421_02_nfe500_resample3.sh
# (NFE=500, K=3 resample at t<0.4, uniform grid) — this combination has
# previously dominated K=1 at iso-compute on the 0421-02 baseline, so it is
# the right default to compare against.
#
# MCAR is intentionally skipped (per project eval defaults).

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
    --num_steps 500 \
    --resample_steps 3 \
    --resample_t_threshold 0.4 \
    --time_grid_mode uniform \
    --output_dir results/imputation/${TIMEID}/mnar_20_nfe500_resample3

echo "**************** [${TIMEID}] MNAR imputation NFE=500 K=3 completed. **************************"
