#!/bin/bash

#SBATCH --job-name="lcl-gen-samples"
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

CKPT="checkpoints/LCL-0414/last-v2.ckpt"

python scripts/showcase/generate_samples.py \
    --model_type flow \
    --checkpoint $CKPT \
    --num_samples 1000 \
    --months 0 1 2 3 4 5 6 7 8 9 10 11 \
    --num_steps 100 \
    --batch_size 256 \
    --cfg_scale 1.0 \
    --output_dir results/generated/LCL-0414

echo "**************** [LCL-0414] sample generation completed. **************************"
