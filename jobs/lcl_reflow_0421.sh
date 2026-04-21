#!/bin/bash

#SBATCH --job-name="lcl-reflow-0421"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/SmartMeterFM

# init venv
source .venv/bin/activate
uv sync

CONFIG="configs/showcase/lcl_0421_reflow.toml"
CACHE="data/reflow_pairs/lcl_0421_01_200nfe.pt"
TIMEID="LCL-0421-REFLOW"

echo "**************** [${TIMEID}] phase 1/2: teacher pair generation **************************"
if [ -f "$CACHE" ]; then
    echo "Cache exists at $CACHE — skipping pair generation."
else
    python scripts/showcase/generate_reflow_pairs.py \
        --config "$CONFIG" \
        --out_path "$CACHE" \
        --seed 42
fi
echo "**************** [${TIMEID}] phase 1/2 complete **************************"

echo "**************** [${TIMEID}] phase 2/2: reflow training **************************"
python scripts/showcase/reflow_distill.py \
    --config "$CONFIG" \
    --time_id "$TIMEID" \
    --seed 42
echo "**************** [${TIMEID}] phase 2/2 complete **************************"
