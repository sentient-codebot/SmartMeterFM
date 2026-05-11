#!/bin/bash

#SBATCH --job-name="lcl-preprocess-csv"
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

# Preprocess the original Low Carbon London halfhourly CSVs into the project's
# monthly NPZ schema, attaching per-row tariff_type (Std=0, ToU=1) and
# acorn_grouped (Affluent=0, Comfortable=1, Adversity=2, ACORN-U=3) labels.
#
# Inputs (must already exist):
#   data/lcl_electricity/raw/halfhourly_dataset/block_*.csv[.gz]   (~112 files)
#   data/lcl_electricity/raw/informations_households.csv
#
# Outputs (overwrites previous TSF-derived NPZs):
#   data/lcl_electricity/raw/lcl_electricity_{2012,2013}_{train,val,test}.npz
#
# Single-core pandas streaming — do not bump --cpus-per-task above 1, the
# preprocessing isn't parallelised.

module load 2025

cd ~/projects/SmartMeterFM

# Resolve dependencies upfront so the heavy step doesn't pay the sync cost.
uv sync

# --- Preprocessing ---
uv run python -c "
from smartmeterfm.data_modules.lcl_electricity import PreLCLElectricityCSV
for y in (2012, 2013):
    PreLCLElectricityCSV(root='data/lcl_electricity/raw', year=y).load_process_save()
"
echo "**************** [LCL-CSV-PREPROC] preprocessing completed for 2012, 2013. **************************"

# --- Verification: print NPZ schema for 2012 train ---
uv run python -c "
import numpy as np
z = np.load('data/lcl_electricity/raw/lcl_electricity_2012_train.npz')
print('keys:', sorted(z.files))
print('sizes per month:', {m: z[str(m)].shape for m in range(1, 13) if str(m) in z.files})
if '1_tariff_type' in z.files:
    print('unique tariff_type (Jan):  ', np.unique(z['1_tariff_type']).tolist())
if '1_acorn_grouped' in z.files:
    print('unique acorn_grouped (Jan):', np.unique(z['1_acorn_grouped']).tolist())
"
echo "**************** [LCL-CSV-PREPROC] verification completed. **************************"
