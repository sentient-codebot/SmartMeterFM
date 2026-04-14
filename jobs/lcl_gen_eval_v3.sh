#!/bin/bash

#SBATCH --job-name="lcl-gen-eval-v3"
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
OUTDIR="results/generated/LCL-0414"
EVALDIR="results/eval/LCL-0414"
FIGDIR="results/figures/LCL-0414"
CONFIG="configs/showcase/lcl_electricity_flow_monthly.toml"

# 1. Generate samples
srun python scripts/showcase/generate_samples.py \
    --model_type flow \
    --checkpoint $CKPT \
    --num_samples 1000 \
    --months 0 1 2 3 4 5 6 7 8 9 10 11 \
    --num_steps 100 \
    --batch_size 256 \
    --cfg_scale 1.0 \
    --output_dir $OUTDIR

echo "**************** [LCL-0414] sample generation completed. **************************"

# 2. Evaluate generated samples against real test data
srun python scripts/showcase/evaluate_samples.py \
    --samples_dir $OUTDIR \
    --config $CONFIG \
    --output_dir $EVALDIR

echo "**************** [LCL-0414] evaluation completed. **************************"

# 3. Plot generated vs real
srun python scripts/showcase/plot_generated_samples.py \
    --samples_dir $OUTDIR \
    --config $CONFIG \
    --output_dir $FIGDIR \
    --eval_metrics $EVALDIR/eval_metrics.json

echo "**************** [LCL-0414] plotting completed. **************************"
