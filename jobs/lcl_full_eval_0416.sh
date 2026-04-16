#!/bin/bash
# reruns all metrics for three tasks using existing data.

#SBATCH --job-name="lcl-full-eval"
#SBATCH --partition=gpu_a100
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

TIMEID="LCL-0416"
CKPT="checkpoints/${TIMEID}/last.ckpt"
CONFIG="configs/showcase/lcl_electricity_flow_monthly.toml"

# 1. Re-evaluate generated samples (no regeneration needed)
srun python scripts/showcase/evaluate_samples.py \
    --samples_dir results/generated/${TIMEID} \
    --config $CONFIG \
    --output_dir results/eval/${TIMEID} \
    --year 2013

echo "**************** [${TIMEID}] evaluate_samples completed. **************************"

# 2. Imputation: MCAR 20%
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

# 3. Imputation: MNAR consecutive 20%
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

# 4. Super-resolution: 2x
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

# 5. Super-resolution: 4x
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
