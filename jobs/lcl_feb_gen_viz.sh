#!/bin/bash

#SBATCH --job-name="lcl-feb-viz"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
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
OUTDIR="results/generated/${TIMEID}-feb"
EVALDIR="results/eval/${TIMEID}-feb"
FIGDIR="results/figures/${TIMEID}-feb"

# Generate samples for February only (month index 1)
srun python scripts/showcase/generate_samples.py \
    --model_type flow \
    --checkpoint $CKPT \
    --num_samples 1000 \
    --months 1 \
    --num_steps 100 \
    --batch_size 256 \
    --cfg_scale 1.0 \
    --year 2013 \
    --output_dir $OUTDIR

echo "**************** [${TIMEID}] Feb sample generation completed. **************************"

# Evaluate
srun python scripts/showcase/evaluate_samples.py \
    --samples_dir $OUTDIR \
    --config $CONFIG \
    --output_dir $EVALDIR \
    --year 2013

echo "**************** [${TIMEID}] Feb evaluation completed. **************************"

# Visualize
srun python scripts/showcase/plot_generated_samples.py \
    --samples_dir $OUTDIR \
    --config $CONFIG \
    --output_dir $FIGDIR \
    --eval_metrics $EVALDIR/eval_metrics.json \
    --year 2013

echo "**************** [${TIMEID}] Feb plotting completed. **************************"
