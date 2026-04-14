#!/bin/bash

#SBATCH --job-name="lcl-eval-0414-01"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=04:00:00
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
CONFIG="configs/showcase/lcl_electricity_flow_monthly.toml"

# ============================================================
# 1. Generate samples + evaluate
# ============================================================

OUTDIR="results/generated/${TIMEID}"
EVALDIR="results/eval/${TIMEID}"
FIGDIR="results/figures/${TIMEID}"

srun python scripts/showcase/generate_samples.py \
    --model_type flow \
    --checkpoint $CKPT \
    --num_samples 1000 \
    --months 0 1 2 3 4 5 6 7 8 9 10 11 \
    --num_steps 100 \
    --batch_size 256 \
    --cfg_scale 1.0 \
    --output_dir $OUTDIR

echo "**************** [${TIMEID}] sample generation completed. **************************"

srun python scripts/showcase/evaluate_samples.py \
    --samples_dir $OUTDIR \
    --config $CONFIG \
    --output_dir $EVALDIR

echo "**************** [${TIMEID}] evaluation completed. **************************"

srun python scripts/showcase/plot_generated_samples.py \
    --samples_dir $OUTDIR \
    --config $CONFIG \
    --output_dir $FIGDIR \
    --eval_metrics $EVALDIR/eval_metrics.json

echo "**************** [${TIMEID}] plotting completed. **************************"

# ============================================================
# 2. Imputation
# ============================================================

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
    --output_dir results/imputation/${TIMEID}/mcar_20

echo "**************** [${TIMEID}] MCAR imputation completed. **************************"

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
    --output_dir results/imputation/${TIMEID}/mnar_20

echo "**************** [${TIMEID}] MNAR imputation completed. **************************"

# ============================================================
# 3. Super-resolution
# ============================================================

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
