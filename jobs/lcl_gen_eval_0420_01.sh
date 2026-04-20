#!/bin/bash

#SBATCH --job-name="lcl-geneval-0420-01"
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

TIMEID="LCL-0420-01"
CKPT="checkpoints/${TIMEID}/last.ckpt"
CONFIG="configs/showcase/lcl_electricity_flow_monthly.toml"
OUTDIR="results/generated/${TIMEID}"
EVALDIR="results/eval/${TIMEID}"
FIGDIR="results/figures/${TIMEID}"

srun python scripts/showcase/generate_samples.py \
    --model_type flow \
    --dataset lcl \
    --checkpoint $CKPT \
    --num_samples 1000 \
    --months 0 1 2 3 4 5 6 7 8 9 10 11 \
    --num_steps 100 \
    --batch_size 256 \
    --cfg_scale 1.0 \
    --year 2013 \
    --output_dir $OUTDIR

echo "**************** [${TIMEID}] sample generation completed. **************************"

srun python scripts/showcase/evaluate_samples.py \
    --samples_dir $OUTDIR \
    --config $CONFIG \
    --output_dir $EVALDIR \
    --year 2013

echo "**************** [${TIMEID}] evaluation completed. **************************"

srun python scripts/showcase/plot_generated_samples.py \
    --samples_dir $OUTDIR \
    --config $CONFIG \
    --output_dir $FIGDIR \
    --eval_metrics $EVALDIR/eval_metrics.json \
    --year 2013

echo "**************** [${TIMEID}] plotting completed. **************************"
