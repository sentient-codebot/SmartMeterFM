# SmartMeterFM Showcase Scripts

This directory contains showcase scripts that demonstrate how to train and use Flow Matching models for energy time series tasks. These scripts use publicly available WPuQ (Heat Pump) data and do not require any proprietary data.

## Overview

The showcase demonstrates four main capabilities:

1. **Conditional Generation**: Generate energy profiles conditioned on temporal attributes (e.g., month)
2. **Imputation**: Fill in missing values in time series data
3. **Super-Resolution**: Upsample low-resolution profiles to high resolution
4. **Baseline Comparison**: VAE baseline for comparison with Flow Matching

## Data

These scripts use the **WPuQ Heat Pump Dataset** from Germany, which contains:
- Heat pump electricity consumption data from 2018-2020
- Multiple households with 10-second resolution (can be resampled to 1min, 15min, 30min, 1h)
- Organized by month for conditional generation

### Data Preparation

1. Download the WPuQ dataset and place it in `data/wpuq/`
2. Run the preprocessing script (if needed):
   ```bash
   uv run python -c "from smartmeterfm.data_modules.heat_pump import PreHeatPump; PreHeatPump(root='data/wpuq/', year=2018).load_process_save()"
   ```

## Scripts

### 1. Training Flow Matching Model

Train a Flow Matching model for conditional energy profile generation:

```bash
# Basic training
uv run python scripts/showcase/train_flow.py \
    --config configs/showcase/wpuq_flow_small.toml \
    --time_id flow_test_001

# Multi-GPU training
uv run python scripts/showcase/train_flow.py \
    --config configs/showcase/wpuq_flow_small.toml \
    --time_id flow_test_001 \
    --num_gpus 2

# Resume from checkpoint
uv run python scripts/showcase/train_flow.py \
    --config configs/showcase/wpuq_flow_small.toml \
    --time_id flow_test_001 \
    --resume_ckpt checkpoints/flow_test_001/last.ckpt
```

### 2. Training VAE Baseline

Train a Conditional VAE baseline for comparison:

```bash
uv run python scripts/showcase/train_vae_baseline.py \
    --config configs/showcase/wpuq_vae.toml \
    --time_id vae_test_001 \
    --latent_dim 128 \
    --beta 0.01
```

**Note**: Beta values >= 0.1 may cause posterior collapse. Recommended: beta=0.01

### 3. Generating Samples

Generate conditional samples from trained models:

```bash
# Generate with Flow model
uv run python scripts/showcase/generate_samples.py \
    --model_type flow \
    --checkpoint checkpoints/flow_test_001/last.ckpt \
    --num_samples 1000 \
    --output_dir samples/flow_test_001

# Generate for specific months only
uv run python scripts/showcase/generate_samples.py \
    --model_type flow \
    --checkpoint checkpoints/flow_test_001/last.ckpt \
    --num_samples 500 \
    --months 0 5 11 \
    --output_dir samples/flow_winter_summer

# Generate with VAE model
uv run python scripts/showcase/generate_samples.py \
    --model_type vae \
    --checkpoint checkpoints/vae_test_001/last.ckpt \
    --num_samples 1000 \
    --output_dir samples/vae_test_001
```

### 4. Imputation Demo

Demonstrate missing data imputation:

```bash
# MCAR (Missing Completely At Random) imputation
uv run python scripts/showcase/imputation_demo.py \
    --checkpoint checkpoints/flow_test_001/last.ckpt \
    --imputation_type mcar \
    --missing_rate 0.2

# Block-wise consecutive missing
uv run python scripts/showcase/imputation_demo.py \
    --checkpoint checkpoints/flow_test_001/last.ckpt \
    --imputation_type mnar_consecutive \
    --missing_rate 0.3 \
    --min_block_size 5
```

### 5. Super-Resolution Demo

Demonstrate temporal super-resolution:

```bash
# 4x super-resolution (e.g., 1h -> 15min)
uv run python scripts/showcase/super_resolution_demo.py \
    --checkpoint checkpoints/flow_test_001/last.ckpt \
    --scale_factor 4

# 2x super-resolution (e.g., 30min -> 15min)
uv run python scripts/showcase/super_resolution_demo.py \
    --checkpoint checkpoints/flow_test_001/last.ckpt \
    --scale_factor 2
```

## Configuration Files

- `configs/showcase/wpuq_flow_small.toml`: Flow Matching model configuration
- `configs/showcase/wpuq_vae.toml`: VAE baseline configuration

Key configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data.resolution` | Time resolution | 15min |
| `model.dim_base` | Base dimension for embeddings | 64 |
| `model.num_decoder_layer` | Number of transformer layers | 6 |
| `train.batch_size` | Training batch size | 64 |
| `train.num_train_step` | Total training steps | 50000 |
| `sample.num_sampling_step` | ODE integration steps | 100 |

## Output Structure

```
checkpoints/
├── flow_test_001/
│   ├── last.ckpt
│   └── flow-step=050000-val_loss=0.1234.ckpt
└── vae_test_001/
    └── ...

samples/
├── flow_test_001/
│   ├── generated_samples.pt
│   ├── month_00_samples.pt
│   └── generation_summary.txt
└── ...

results/
├── configs/
│   └── exp_config_flow_test_001.yaml
├── imputation/
│   └── imputation_results.pt
└── super_resolution/
    └── sr_4x_results.pt
```

## Evaluation Metrics

### Generation Quality
- **MkMMD**: Maximum Mean Discrepancy with multiple kernels
- **Frechet Distance**: Distribution similarity
- **KL Divergence**: Statistical divergence
- **Wasserstein Distance**: Optimal transport distance

### Imputation Quality
- **MSE/MAE/RMSE**: Reconstruction error at missing positions
- **Uncertainty**: Standard deviation of imputed samples

### Super-Resolution Quality
- **MSE/MAE/RMSE**: Compared to original high-resolution
- **Improvement**: Percentage improvement over linear interpolation baseline

## Requirements

- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- flow_matching >= 1.0
- einops, numpy, scipy, h5py

Install with:
```bash
uv sync
```

## References

- WPuQ Dataset: German Heat Pump Electricity Consumption Dataset
- Flow Matching: [Lipman et al., 2023](https://arxiv.org/abs/2210.02747)
