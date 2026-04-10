#!/bin/bash

#SBATCH --job-name="preprocess-pv-monthly"
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

# Preprocess WPuQ PV data with monthly segments
for YEAR in 2018 2019 2020; do
    python -c "
from smartmeterfm.data_modules.wpuq_pv import PreWPuQPV
pre = PreWPuQPV(root='data/wpuq/raw', year=${YEAR}, segment_type='monthly')
result = pre.load_process_save()
print('Year ${YEAR} - Train samples/month:', result[0])
print('Year ${YEAR} - Val samples/month:', result[1])
print('Year ${YEAR} - Test samples/month:', result[2])
"
    echo "**************** Year ${YEAR} preprocessing completed. **************************"
done

echo "**************** All monthly preprocessing completed. **************************"
