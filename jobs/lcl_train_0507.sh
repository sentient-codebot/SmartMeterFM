#!/bin/bash

#SBATCH --job-name="lcl-flow-0507"
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

# Training run for the 0507 LCL flow-matching model with the new
# per-household conditions enabled (tariff_type, acorn_grouped) on top of
# the 0421 hyperparameter set.  Inherits all training hyperparameters from
# configs/showcase/lcl_0421.toml; the only deltas are the additional
# target_labels and the exp_id rename.
#
# Prerequisites:
#   - data/lcl_electricity/raw/lcl_electricity_{2012,2013}_*.npz produced by
#     PreLCLElectricityCSV (jobs/lcl_preprocess_csv.sh).
#   - The new NPZ schema includes <m>_tariff_type and <m>_acorn_grouped keys.

module load 2025

cd ~/projects/SmartMeterFM

uv sync

uv run python scripts/showcase/train_flow.py \
    --config configs/showcase/lcl_0507_tariff_acorn.toml \
    --time_id LCL-0507 \
    --seed 42

echo "**************** [LCL-0507] training completed. **************************"
