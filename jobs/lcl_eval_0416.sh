#!/bin/bash

#SBATCH --job-name="lcl-eval-0416"
#SBATCH --partition=genoa
#SBATCH --mem=140G
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
CONFIG="configs/showcase/lcl_electricity_flow_monthly.toml"
OUTDIR="results/generated/${TIMEID}"
EVALDIR="results/eval/${TIMEID}"
FIGDIR="results/figures/${TIMEID}"

python scripts/showcase/evaluate_samples.py \
    --samples_dir $OUTDIR \
    --config $CONFIG \
    --output_dir $EVALDIR \
    --year 2013

echo "**************** [${TIMEID}] evaluation completed. **************************"

python scripts/showcase/plot_generated_samples.py \
    --samples_dir $OUTDIR \
    --config $CONFIG \
    --output_dir $FIGDIR \
    --eval_metrics $EVALDIR/eval_metrics.json \
    --year 2013

echo "**************** [${TIMEID}] plotting completed. **************************"
