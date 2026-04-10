#!/bin/bash

#SBATCH --job-name="preprocess-household"
#SBATCH --partition=genoa
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

module load 2025

cd ~/projects/SmartMeterFM

# init venv
source .venv/bin/activate
uv sync

# Preprocess WPuQ Household data with monthly segments
for YEAR in 2018 2019 2020; do
    python -c "
from smartmeterfm.data_modules.wpuq_household import PreWPuQHousehold
pre = PreWPuQHousehold(root='data/wpuq/raw', year=${YEAR})
pre.load_process_save()
"
    echo "**************** Year ${YEAR} preprocessing completed. **************************"
done

echo "**************** All household preprocessing completed. **************************"
