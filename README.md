# SmartMeterFM

Flow Matching for Smart Meter Time Series Generation

## Overview

SmartMeterFM applies [Flow Matching](https://arxiv.org/abs/2210.02747) to smart meter energy time series data. It supports:

- **Conditional Generation** — generate energy profiles conditioned on temporal attributes (month, season, etc.)
- **Imputation** — fill in missing values via posterior sampling (MCAR and block-wise patterns)
- **Super-Resolution** — upsample low-resolution profiles to higher temporal resolution
- **Guided Sampling** — steer generation using measurement operators at inference time

## Installation

Requires Python >= 3.12. The project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/your-org/SmartMeterFM.git
cd SmartMeterFM
uv sync
```

## Quick Start

### 1. Prepare Data

Download the [WPuQ Heat Pump dataset](https://data.open-power-system-data.org/household_data/) and place it in `data/wpuq/`. Then preprocess:

```bash
uv run python -c "
from smartmeterfm.data_modules.heat_pump import PreHeatPump
PreHeatPump(root='data/wpuq/', year=2018).load_process_save()
"
```

### 2. Train a Model

```bash
uv run python scripts/showcase/train_flow.py \
    --config configs/showcase/wpuq_flow_small.toml \
    --time_id my_first_run
```

### 3. Generate Samples

```bash
uv run python scripts/showcase/generate_samples.py \
    --model_type flow \
    --checkpoint checkpoints/my_first_run/last.ckpt \
    --num_samples 1000 \
    --output_dir samples/my_first_run
```

### 4. Imputation & Super-Resolution

```bash
# Imputation (20% MCAR missing data)
uv run python scripts/showcase/imputation_demo.py \
    --checkpoint checkpoints/my_first_run/last.ckpt \
    --imputation_type mcar --missing_rate 0.2

# Super-resolution (4x: 1h → 15min)
uv run python scripts/showcase/super_resolution_demo.py \
    --checkpoint checkpoints/my_first_run/last.ckpt \
    --scale_factor 4
```

See [scripts/showcase/README.md](scripts/showcase/README.md) for detailed usage, configuration options, and evaluation metrics.

## Supported Datasets

| Dataset | Description | Config |
|---------|-------------|--------|
| **WPuQ Heat Pump** | German household heat pump electricity consumption (2018-2020) | `configs/showcase/wpuq_flow_small.toml` |
| **WPuQ PV** | German household solar PV generation with directional data (East/South/West) | `configs/showcase/wpuq_pv_flow_small.toml` |

## Project Structure

```
SmartMeterFM/
├── src/smartmeterfm/
│   ├── data_modules/         # Dataset loading and preprocessing
│   │   ├── heat_pump.py      # WPuQ Heat Pump dataset
│   │   └── wpuq_pv.py        # WPuQ PV dataset
│   ├── models/
│   │   ├── flow.py            # FlowModelPL — main flow matching model
│   │   ├── nn_components.py   # DenoisingTransformer, attention layers
│   │   ├── measurement.py     # Measurement operators for guided sampling
│   │   ├── embedders/         # Condition embedders (month, season, etc.)
│   │   └── baselines/         # VAE, GAN, imputation, super-resolution baselines
│   ├── interfaces/            # Inference interface
│   └── utils/                 # Configuration, evaluation metrics, callbacks
├── scripts/showcase/          # Training, generation, and demo scripts
├── configs/showcase/          # TOML training configurations
└── pyproject.toml
```

## How to Extend

### Adding a New Dataset

1. Create `src/smartmeterfm/data_modules/your_dataset.py` following the pattern in `heat_pump.py`:
   - A `Pre*` class for raw data preprocessing (→ NPZ files split by month)
   - A main class extending `TimeSeriesDataCollection`
2. Add a config in `configs/showcase/`

### Adding a New Embedder

Use the registry system in `models/embedders/`:

```python
from smartmeterfm.models.embedders._registry import register_embedder

@register_embedder("my_embedder")
class MyEmbedder(nn.Module):
    def __init__(self, dim_embedding, ...):
        ...
    def forward(self, labels, train=True, force_drop_ids=None):
        ...
```

See `models/embedders/wpuq.py` for examples (`wpuq_month`, `wpuq_month_season`, `wpuq_full`).

## References

- **Flow Matching**: Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), 2023
- **WPuQ Dataset**: [Open Power System Data — Household Data](https://data.open-power-system-data.org/household_data/)

## License

[MIT](LICENSE)
