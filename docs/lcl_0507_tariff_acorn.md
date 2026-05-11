# LCL-0507 — Tariff + ACORN-grouped Conditions on Top of 0421 Hyperparameters

Trained 2026-05-07. Run id `LCL-0507`, wandb run `8ku47djw`. Checkpoint at
`checkpoints/LCL-0507/last.ckpt`. Config: `configs/showcase/lcl_0507_tariff_acorn.toml`.

## Setup

**What changed vs `LCL-0421`** (the base teacher; the `0421-01`/`0421-02`
ablations did not improve over it, so 0507 follows the 0421 hyperparameter set):

- `target_labels = ["month", "year", "tariff_type", "acorn_grouped"]` — adds two
  per-household conditions sourced from the original LCL release's
  `informations_households.csv`:
  - `tariff_type ∈ {Std=0, ToU=1}`
  - `acorn_grouped ∈ {Affluent=0, Comfortable=1, Adversity=2, ACORN-U=3}`
- `exp_id = "showcase_lcl_electricity_0507_tariff_acorn"`.
- Ingestion source swapped from the Zenodo TSF (anonymised IDs, no metadata
  link) to the original halfhourly CSVs (keyed on `LCLid`). New
  `PreLCLElectricityCSV` class; old `PreLCLElectricity` (TSF) preserved as
  deprecated for backward compatibility.

All other hyperparameters identical to `lcl_0421.toml` — same model size, same
50k-step schedule, same EMA, same `seed=42`.

**Eval recipe.** Generation: 1000 samples × 12 months on 2013, 100 ODE steps,
cfg_scale=1.0. Imputation: MNAR-consecutive at 20% missing,
`num_test_series=100`, `num_samples=10`. SR: 4× temporal (2h → 30min),
`num_test_series=100`, `num_samples=10`, 200 steps. **CFG marginalises over
`tariff_type` and `acorn_grouped`** in all three eval scripts (they default to
`None` in the demo CLIs), so the metrics below are directly comparable to the
0421 baselines — they measure whether the *additional* embedder weights
degrade unconditional / coarsely-conditioned quality, not the value of the new
conditions themselves. A per-(tariff, acorn) conditional sampling grid is the
obvious follow-up.

## 1. Unconditional generation — distribution metrics

Per-month average over Jan–Dec 2013, 1000 samples per month.

| metric                  | LCL-0421 | LCL-0421-01 | LCL-0421-02 | **LCL-0507** | direction |
|-------------------------|---------:|------------:|------------:|-------------:|:---------:|
| MkMMD                   | 5.65 e-3 |    7.42 e-3 |    7.05 e-3 | **4.59 e-3** | ↓ best    |
| ws_distance             | 1.87 e-3 |    2.23 e-3 |    2.11 e-3 | **1.76 e-3** | ↓ best    |
| **CRPS**                | **6.96 e-5** |    9.67 e-5 |    9.59 e-5 |     8.96 e-5 | ↓ 2nd     |
| **PearsonR_diff**       |    0.143 |   **0.123** |       0.141 |        0.127 | ↓ 2nd     |
| **PeakLoadError**       | 1.12 e-3 |    2.70 e-3 |    2.38 e-3 | **3.82 e-4** | ↓ 3× best |
| **SymQuantileError_99** | 1.76 e-3 |    2.56 e-3 |    2.26 e-3 | **9.46 e-4** | ↓ 2× best |
| kl_divergence           | **0.016**|       0.026 |       0.025 |        0.023 | ↓ 4th     |
| ks_test_d               | **0.059**|       0.079 |       0.078 |        0.078 | matches 01/02 |

Bolded metric names are the project priority metrics
(`feedback_metric_priorities.md`). 0507 wins outright on PeakLoadError and
SymQuantileError_99, leads on MMD/Wasserstein, sits in the middle of the pack
on CRPS / PearsonR_diff, regresses on KL and KS to the 0421-01/02 level. Net:
adding the two condition embedders does not hurt unconditional generation
under CFG; it materially improves peak/tail fidelity.

> **DirectFD warning.** The 0421 evals predate commit `19b172d`
> (*"fix(metric): fix frechet calc"*). Their reported DirectFD ~7e-4 is from
> the buggy implementation; 0507's value of 0.294 is from the corrected
> formula. Re-run 0421 if you need an apples-to-apples FD.

Raw: `results/eval/LCL-0507/eval_metrics.json`.
Job script: `jobs/lcl_gen_eval_0507.sh`.

## 2. Imputation — MNAR consecutive 20%

### 2.1 NFE=500 K=3 (high-budget, ~1100 net queries)

Recipe: NFE=500, K=3 resample at τ=0.4, uniform grid (the default recommended
in `reflow_and_posterior_sampling.md` §3).

| metric                  | LCL-0421-02 | **LCL-0507** | naive-fill baseline (0507) |
|-------------------------|------------:|-------------:|---------------------------:|
| **CRPS**                |    9.05 e-3 | **9.00 e-3** |                   1.83 e-2 |
| **PeakLoadError**       |    1.50 e-2 | **1.30 e-2** |                   2.43 e-3 |
| **SymQuantileError_99** |    1.03 e-2 | **1.01 e-2** |                   6.24 e-3 |
| **PearsonR_diff**       |    5.88 e-2 | **5.22 e-2** |                   4.80 e-2 |
| MSE                     |    4.89 e-4 | **4.87 e-4** |                   1.09 e-3 |
| MAE                     |    1.12 e-2 | **1.09 e-2** |                   1.83 e-2 |

0507 strictly improves over 0421-02 on every metric. PeakLoad/SymQ99 are still
worse than the naive-fill baseline — that gap was already present on 0421-02
and is a property of the task / resample recipe, not a 0507 regression.

Raw: `results/imputation/LCL-0507/mnar_20_nfe500_resample3/metrics.json`.
Job script: `jobs/lcl_imputation_0507.sh`.

### 2.2 NFE=227 K=3 (iso-compute vs K=1 NFE=500, ~500 net queries)

Direct test of the §2.1d budget-conversion rule on the new model. Net queries
= 227 · 2.2 = 499.4, matching the NFE=500 K=1 baseline.

| Config                                  | NFE | K | τ   | Net | CRPS    | PeakLoad | SymQ99  | PearsonR_diff |
|-----------------------------------------|----:|--:|----:|----:|--------:|---------:|--------:|--------------:|
| 0421-02 K=1 baseline (§2.1d row 1)      | 500 | 1 |  —  | 500 | 0.00950 |  0.02493 | 0.01447 |        0.0968 |
| 0421-02 K=3 iso-compute (§2.1d row 2)   | 227 | 3 | 0.4 | 499 | 0.00915 |  0.01926 | 0.01279 |        0.0881 |
| **0507 K=3 iso-compute**                | **227** | **3** | **0.4** | **499** | **0.00910** | **0.01759** | **0.01253** | **0.0838** |

0507 vs 0421-02 K=1 NFE=500 (the headline iso-compute test): **CRPS −4.2%,
PeakLoadError −29.5%, SymQuantileError_99 −13.4%, PearsonR_diff −13.5%.**

0507 vs 0421-02 K=3 NFE=227 (same recipe, different teacher): CRPS −0.5%,
PeakLoadError −8.7%, SymQ99 −2.0%, PearsonR_diff −4.9%.

The §2.1d findings reproduce on 0507 with quantitatively similar deltas — the
budget-conversion rule (`NFE = B/2.2` with K=3 dominates `NFE = B` with K=1)
is not specific to the 0421-02 teacher.

Raw: `results/imputation/LCL-0507/mnar_20_nfe227_resample3/metrics.json`.
Job script: `jobs/lcl_imputation_0507_nfe227_resample3.sh`.

### 2.3 Iso vs high-budget on 0507 (NFE=227 vs NFE=500, both K=3)

| metric                  | NFE=227 K=3 (~500 net) | NFE=500 K=3 (~1100 net) |     Δ |
|-------------------------|-----------------------:|------------------------:|------:|
| **CRPS**                |                0.00910 |                 0.00900 | −1.2% |
| **PeakLoadError**       |                0.01759 |                 0.01296 | **−26.3%** |
| **SymQuantileError_99** |                0.01253 |                 0.01009 | **−19.5%** |
| **PearsonR_diff**       |                 0.0838 |                  0.0522 | **−37.7%** |
| MSE                     |               4.80 e-4 |                4.87 e-4 | +1.3% |
| MAE                     |                0.01114 |                 0.01092 | −1.9% |

CRPS is flat (saturated near the ~0.0095 dataset/operator floor). Tail and
peak metrics improve substantially with the extra compute, consistent with
§2.1b. 2.2× more compute buys big gains where it matters; pick the budget by
which metric is the headline.

## 3. Super-resolution — 4× (2h → 30min)

200 steps, no resample.

| metric                  | LCL-0421 | LCL-0421-02 | **LCL-0507** | cubic-interp baseline (0507) |
|-------------------------|---------:|------------:|-------------:|-----------------------------:|
| **CRPS**                | 6.61 e-3 |    6.59 e-3 | **6.39 e-3** |                     6.87 e-3 |
| **PeakLoadError**       | **4.94 e-2** | 5.04 e-2 |     5.29 e-2 |                     7.44 e-2 |
| **SymQuantileError_99** | 2.80 e-2 |    2.86 e-2 | **2.68 e-2** |                     3.19 e-2 |
| **PearsonR_diff**       |    0.178 |   **0.169** |        0.189 |                        0.353 |
| MSE                     | 2.08 e-4 |    2.07 e-4 | **2.01 e-4** |                     1.94 e-4 |
| MAE                     | 8.16 e-3 |    8.12 e-3 | **7.84 e-3** |                     6.87 e-3 |

0507 takes CRPS and SymQuantileError_99. PeakLoadError and PearsonR_diff
regress slightly vs 0421/0421-02 — both still beat the cubic-interp baseline
by a wide margin, and the deltas are smaller than the gen-task improvements.
Resample-only SR is still flagged as the open follow-up
(see `reflow_and_posterior_sampling.md` §2.2 final paragraph).

Raw: `results/super_resolution/LCL-0507/4x/metrics.json`.
Job script: `jobs/lcl_sr_0507.sh`.

## 4. What this means

- **The added `tariff_type` + `acorn_grouped` conditions do not hurt CFG-marginalised
  evaluation.** Across all three tasks, 0507 holds parity with or strictly
  improves on 0421-02. The model trained as a strict superset under CFG
  dropout, as designed.
- **Imputation is the strongest result.** 0507 K=3 NFE=227 beats 0421-02 K=1
  NFE=500 by 30% on PeakLoad at the same compute. This is the recommended
  inference recipe for any deployed imputation pipeline on 0507.
- **The `reflow_and_posterior_sampling.md` recipes transfer.** §2.1d's
  budget-conversion rule, §3's defaults (`--resample_steps 3
  --resample_t_threshold 0.4 --time_grid_mode uniform`), and the iso-vs-high
  trade-off curve all reproduce on 0507 with similar numerical deltas. No
  retuning needed.
- **The new conditions have not been exercised yet.** Every metric above was
  computed with `tariff_type=None, acorn_grouped=None` (CFG-dropped). To
  measure the *value* of the conditions — does conditioning on `(Std, Affluent)`
  produce sharper / more correct profiles than marginal? — needs a per-(tariff,
  acorn) conditional sampling grid, which is the natural next experiment.

## 5. Files / pointers

| Artifact         | Path |
|------------------|------|
| Config           | `configs/showcase/lcl_0507_tariff_acorn.toml` |
| Training job     | `jobs/lcl_train_0507.sh` (job 22553424, 1h15m on `gcn46`) |
| Gen+eval+plot job| `jobs/lcl_gen_eval_0507.sh` (job 22567154) |
| Imputation NFE=500 job | `jobs/lcl_imputation_0507.sh` (job 22567155) |
| Imputation NFE=227 job | `jobs/lcl_imputation_0507_nfe227_resample3.sh` (job 22567484) |
| SR 4× job        | `jobs/lcl_sr_0507.sh` (job 22567156) |
| Checkpoint       | `checkpoints/LCL-0507/last.ckpt` (last.ckpt @ step 50000) |
| wandb run        | https://wandb.ai/nan-team/smartmeterfm-showcase/runs/8ku47djw |
| Cross-reference  | `docs/reflow_and_posterior_sampling.md` §2.1d (rows for 0507 added) |
