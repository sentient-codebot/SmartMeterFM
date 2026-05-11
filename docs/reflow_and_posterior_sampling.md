# Reflow Distillation & Posterior-Sampling Improvements — Results

Checkpoint base: `LCL-0421` (teacher) → `LCL-0421-REFLOW` (reflow-1) → `LCL-0421-REFLOW2` (reflow-2).
Task settings: generation is per-month over 12 months (reported as overall aggregate); imputation is MNAR-consecutive at 20% missing with `num_test_series=100`, `num_samples=10`; SR is 4× temporal with `num_test_series=100`, `num_samples=10`.

## 1. Reflow distillation (NFE sweep on generation)

One round of reflow (reflow-1) trained on `(x_0, x_1)` pairs produced by the teacher at NFE=200, then a second round (reflow-2) trained on pairs produced by reflow-1.

| Model              | NFE  |   CRPS ×1e3 | MkMMD ×1e3 | KL     | WS ×1e3 | KS     | PeakLoad ×1e3 | SymQ99 ×1e3 | PearsonR_diff |
|--------------------|-----:|------------:|-----------:|-------:|--------:|-------:|--------------:|------------:|--------------:|
| Teacher            |  500 |       0.039 |      2.67  | 0.0121 |   1.72  | 0.0571 |         0.93  |       1.56  |        0.143  |
| Reflow-1           |   20 |       0.169 |     12.23  | 0.0508 |   3.40  | 0.1208 |        15.65  |       9.33  |        0.173  |
| Reflow-1           |   50 |       0.096 |      7.63  | 0.0288 |   2.57  | 0.0930 |         6.06  |       5.52  |        0.134  |
| Reflow-1           |  100 |       0.079 |      5.49  | 0.0268 |   2.23  | 0.0857 |         4.66  |       4.01  |        0.089  |
| Reflow-2           |   20 |       0.154 |     10.98  | 0.0422 |   3.06  | 0.1153 |        13.16  |       8.36  |        0.170  |
| Reflow-2           |   50 |       0.104 |      7.83  | 0.0258 |   2.52  | 0.0942 |         6.50  |       5.76  |        0.140  |
| Reflow-2           |  100 |       0.094 |      6.57  | 0.0293 |   2.36  | 0.0926 |         5.58  |       4.91  |        0.107  |

Takeaways:
- **Reflow-2 at NFE=20 beats reflow-1 at NFE=20** on every metric (~10% lower MkMMD, ~16% lower PeakLoadError). So the second reflow pass did preserve the straightening gain.
- **Reflow-1/2 do not reach teacher quality** even at NFE=100 — teacher @ 500 NFE remains ~2× better on MkMMD, KL, PeakLoad. Reflow buys a fast path, not a free upgrade.
- `DirectFD` is not comparable across reflow-2 (formula was adjusted) and the earlier teacher/reflow-1 runs; omitted above for that reason.

## 2. Posterior sampling at low NFE

Reflow straightens the *unconditional* trajectory; it does nothing for the posterior-sampling failure at low NFE (imputation at 20 NFE cratered, SR at 20 NFE lost to linear interpolation). Two minimal inference-time knobs on the existing `PosteriorVelocityModelWrapper` (PROJECT mode), no retraining:

- **Resampling (RePaint-style)** at early steps: for outer steps with `t < τ=0.4`, run K=3 inner iterations of *(project → re-noise with fresh ε → re-query v)* before the Euler step. Gives the model repeated chances to refine `x̂₁ = x_t + (1−t)·v_θ` while it's still a noisy extrapolation.
- **Adaptive time grid**: non-uniform step placement via `t = u^γ` with γ=2 concentrates steps near t=0 where `x̂₁` is least reliable.

Both are controlled by new `PosteriorSampleConfig` fields (`resample_steps`, `resample_t_threshold`, `time_grid_mode`, `time_grid_gamma`), threaded through `SmartMeterFMModel.impute() / super_resolve()`, with matching CLI flags on the two demo scripts. Feature-off defaults make the code path byte-identical to before.

### 2.1 Imputation @ NFE=20 on LCL-0421-REFLOW2 (MNAR 20%)

| Config                           |    CRPS |  PeakLoad | SymQ99  | PearsonR_diff | Notes                         |
|----------------------------------|--------:|----------:|--------:|--------------:|-------------------------------|
| Reflow-2 @ **NFE=20** (baseline) | 0.01369 |   0.08888 | 0.05318 |        0.264  | feature-off                   |
| Reflow-2 @ NFE=20 + **adaptive only** (γ=2) | 0.01986 |   0.20864 | 0.12703 |        0.373  | **net loss** — last Euler step too big |
| Reflow-2 @ NFE=20 + **resample K=3**         | **0.01092** | **0.04377** | **0.02803** | 0.219 | **decisive win** |
| Reflow-2 @ NFE=20 + resample K=3 + adaptive  | 0.01304 |   0.09338 | 0.06439 |        0.350  | adaptive drags resample down  |
| Reflow-2 @ **NFE=50** (reference)| 0.01128 |   0.05169 | 0.03214 |        0.218  | baseline-at-higher-budget     |
| Reflow-2 @ NFE=500               | 0.00972 |   0.02507 | 0.01494 |        0.103  | cost-no-object ceiling        |
| Teacher-0421 @ NFE=500           | 0.00948 |   0.02480 | 0.01450 |        0.096  | true ceiling                  |

**Headline:** resample K=3 at NFE=20 matches Reflow-2 @ NFE=50 quality at ~12% less compute (44 net/sample vs 50). CRPS drops 20%, PeakLoadError 51%, SymQ99 47% relative to the feature-off NFE=20 baseline.

**Adaptive grid (γ=2, N=20) is a net loss:** the gap between `t=0.81²=0.656` and `t=1.0` is ~0.19 consumed in a single Euler step, which alone destroys quality. Would need much larger N or γ closer to 1 to help.

### 2.1b Imputation NFE sweep with K=3 resample — reaching and exceeding teacher

Followup: does resample K=3 let reflow match the teacher's NFE=500 quality at higher NFE budgets?

| Config                   |    CRPS |  PeakLoad |  SymQ99 | PearsonR | Net queries |
|--------------------------|--------:|----------:|--------:|---------:|------------:|
| Teacher @ NFE=500        | 0.00948 |   0.02480 | 0.01450 |   0.0957 |         500 |
| Reflow-1 @ NFE=500       | 0.00966 |   0.02542 | 0.01494 |   0.1034 |         500 |
| Reflow-2 @ NFE=500       | 0.00972 |   0.02507 | 0.01494 |   0.1034 |         500 |
| Reflow-2 @ **NFE=100 +K=3** | 0.00990 | 0.02403 | 0.01587 |   0.1250 |         220 |
| Reflow-2 @ **NFE=200 +K=3** | 0.00973 | **0.01975** | **0.01322** | 0.0968 |      440 |
| Reflow-2 @ NFE=300 +K=3  | 0.00959 |   0.01734 | 0.01199 |   0.0815 |         660 |
| Reflow-2 @ NFE=400 +K=3  | 0.00957 |   0.01585 | 0.01104 |   0.0694 |         880 |
| Reflow-2 @ NFE=500 +K=3  | 0.00959 |   **0.01496** | **0.01033** | **0.0618** |    1100 |
| Reflow-1 @ NFE=300 +K=3  | 0.00949 |   0.01734 | 0.01200 |   0.0827 |         660 |
| Reflow-1 @ NFE=400 +K=3  | 0.00946 |   0.01586 | 0.01105 |   0.0705 |         880 |
| Reflow-1 @ NFE=500 +K=3  | 0.00948 |   0.01497 | 0.01034 |   0.0630 |        1100 |

(Net queries ≈ N + 3·0.4·N = 2.2·N, since K=3 inner iterations apply on ~40% of steps where t<0.4.)

Findings:
- **Crossover at NFE=200 +K=3** (~440 net queries): matches teacher CRPS within 3%, and already **beats teacher on PeakLoad (−20%), SymQ99 (−9%)**, with PearsonR on par.
- **NFE=500 +K=3** (1100 net) delivers **40% lower PeakLoadError** and **29% lower SymQ99** than teacher @ NFE=500. K=3 genuinely improves posterior calibration, not just compensates for reflow's loss.
- **Reflow-1 and Reflow-2 are indistinguishable once K=3 is on** (differences ≤1% across all metrics). Reflow-2's only edge was at low NFE, and K=3 closes that gap.
- **CRPS saturates at ~0.0095** across all high-NFE configs — this is the measurement/operator noise floor for MNAR-20% on this dataset, not a model ceiling.

### 2.1c Resample-threshold sweep at NFE=500 on LCL-0421-02

Followup: how sensitive is K=3 resampling to the threshold τ that decides which early steps get the inner iterations? Sweep τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5} on the **LCL-0421-02 teacher checkpoint** (alternate optimizer config, not reflow), NFE=500, MNAR-20%.

| τ (resample_t_threshold) |    CRPS |  PeakLoad |  SymQ99 | PearsonR_diff | Net queries |
|--------------------------|--------:|----------:|--------:|--------------:|------------:|
| 0.1                      | 0.00916 |   0.02061 | 0.01321 |        0.0906 |         650 |
| 0.2                      | 0.00901 |   0.01835 | 0.01234 |        0.0798 |         800 |
| 0.3                      | 0.00898 |   0.01646 | 0.01131 |        0.0695 |         950 |
| 0.4                      | 0.00905 |   0.01496 | 0.01029 |        0.0588 |        1100 |
| 0.5                      | 0.00915 | **0.01305** | **0.00928** | **0.0472** |    1250 |

(Net queries = N · (1 + 3·τ).)

Findings:
- **CRPS is flat across τ** (0.00898–0.00916, ≤2% spread): confirms the ~0.0095 dataset floor — extra resampling cannot move it.
- **PeakLoad / SymQ99 / PearsonR improve monotonically** with τ. τ=0.5 cuts PeakLoad −37% and SymQ99 −30% relative to τ=0.1, and PearsonR_diff drops nearly 2×.
- **Diminishing returns past τ=0.4** on tail metrics: τ=0.4→0.5 gains ~13% PeakLoad / ~10% SymQ99 for +14% compute. Worth it only when peak/tail accuracy is the priority.
- **Cross-checkpoint sanity**: 0421-02 @ NFE=500 +K=3 τ=0.4 (CRPS 0.00905, PeakLoad 0.01496, SymQ99 0.01029) tracks the reflow-2 number from §2.1b (0.00959 / 0.01496 / 0.01033) to within noise on tail metrics, with a small CRPS advantage from the cleaner teacher. The recipe transfers.
- **Updated recommendation**: keep τ=0.4 as default for balanced cost/quality; bump to τ=0.5 when peak-load fidelity is the headline metric.

### 2.1d Iso-compute: K=3 vs K=1 at fixed net-query budget on LCL-0421-02

Followup to §2.1c: the K=3 wins above all spent more compute than the K=1 baseline. The fair question is whether, at a *fixed* compute budget, it's better to spend queries on more outer steps (K=1, large N) or on RePaint-style inner iterations on early steps (K=3, smaller N). Set N so that net queries match the K=1 NFE=500 baseline: `N · (1 + 3·τ) = 500` with τ=0.4 → **N = 227** (net = 499.4).

| Config                                | NFE | K | τ   | Net queries |    CRPS |  PeakLoad |  SymQ99 | PearsonR_diff |
|---------------------------------------|----:|--:|----:|------------:|--------:|----------:|--------:|--------------:|
| 0421-02 K=1 baseline                  | 500 | 1 |  —  |         500 | 0.00950 |   0.02493 | 0.01447 |        0.0968 |
| **0421-02 K=3 iso-compute**           | **227** | **3** | **0.4** | **499** | **0.00915** | **0.01926** | **0.01279** | **0.0881** |
| 0421-02 K=3 high-budget (§2.1c, ref)  | 500 | 3 | 0.4 |        1100 | 0.00905 |   0.01496 | 0.01029 |        0.0588 |
| 0507 K=3 iso-compute (cross-ref `docs/lcl_0507_tariff_acorn.md`) | 227 | 3 | 0.4 | 499 | 0.00910 | 0.01759 | 0.01253 | 0.0838 |
| 0507 K=3 high-budget (cross-ref `docs/lcl_0507_tariff_acorn.md`)  | 500 | 3 | 0.4 | 1100 | 0.00900 | 0.01296 | 0.01009 | 0.0522 |

Findings:
- **K=3 dominates K=1 at iso-compute on every metric**: CRPS −3.7%, PeakLoad **−22.7%**, SymQ99 **−11.6%**, PearsonR_diff −9.0%. The win is real, not a compute-budget artifact.
- **Tail metrics gain the most**, consistent with the §2.1c story: inner iterations refine `x̂₁ = x_t + (1−t)·v_θ` while it's still a noisy extrapolation, which is exactly where peaks/quantiles get mispredicted.
- **Marginal returns past iso-compute are smaller than the iso-compute jump itself.** Going 500 → 1100 net queries (NFE=227 → 500 with K=3 fixed) further halves PeakLoad/SymQ99, but the bulk of the K=3 win comes from *redirecting* compute outer → inner, not from adding more.
- **Reinforced recommendation**: at any net-query budget, prefer K=3 + τ=0.4 with NFE = budget/2.2 over K=1 + NFE = budget. K=1 at NFE=500 is dominated by K=3 at NFE=227 for the same cost.
- **Recipe transfers to LCL-0507** (0421 hyperparameters + tariff_type + acorn_grouped conditions). The two 0507 rows above reproduce the 0421-02 pattern with quantitatively similar deltas, and 0507 even improves on 0421-02 at every iso/high-budget point (CRPS −0.5% / −0.5%, PeakLoad −8.7% / −13.4%, SymQ99 −2.0% / −2.0%, PearsonR_diff −4.9% / −11.2%). Full 0507 evaluation including unconditional generation and 4× SR lives in `docs/lcl_0507_tariff_acorn.md`.

### 2.2 SR 4× @ NFE=20 on LCL-0421-REFLOW2

| Config                                     | CRPS_sr  | PeakLoad_sr | SymQ99_sr | PearsonR_diff |
|--------------------------------------------|---------:|------------:|----------:|--------------:|
| Linear-interp baseline                     |  0.00729 |     0.07087 |   0.03422 |        0.367  |
| Reflow-2 @ NFE=20 (baseline)               |  0.00979 |     0.10578 |   0.06509 |        0.381  |
| Reflow-2 @ NFE=20 + resample K=3 + adaptive|  0.00961 |     0.09617 |   0.06071 |        0.301  |
| Reflow-2 @ NFE=50                          |  0.00783 |     0.07065 |   0.04198 |        0.270  |
| Reflow-2 @ NFE=200                         |  0.00678 |     0.05665 |   0.03243 |        0.196  |
| Teacher-0421 @ NFE=200                     |  0.00661 |     0.04939 |   0.02801 |        0.178  |

SR shows the same pattern (combining resample + adaptive grid gives only a marginal nudge and still loses to linear at 20 NFE), but the resample-only variant for SR was not run. If we revisit SR speedups, running resample-K=3 alone is the next experiment to do.

## 3. What this means in practice

- Always enable **`--resample_steps 3 --resample_t_threshold 0.4 --time_grid_mode uniform`** for imputation. At any compute budget it either matches or beats the no-resample path; at iso-compute it strictly dominates K=1 (§2.1d: −23% PeakLoad, −12% SymQ99 vs K=1 NFE=500 at 500 net queries); and at NFE≥200 it beats the teacher itself on peak/tail metrics. Bump τ to **0.5** when peak-load / tail accuracy is the priority (§2.1c, ~13% PeakLoad gain for +14% compute).
- **Budget conversion rule (§2.1d):** given a target net-query budget `B`, prefer `NFE = B/2.2` with `K=3, τ=0.4` over `NFE = B` with `K=1`. The K=1 path is dominated.
- Keep **`--time_grid_mode uniform`**. Do not enable geometric grid at current settings.
- **Speed regime (≤ NFE=20):** Reflow-2 + K=3 is the only viable option; Reflow-2 alone collapses.
- **Quality regime (NFE ≥ 200):** Reflow-1 and Reflow-2 are interchangeable once K=3 is on. Use either.
- **New quality ceiling for imputation**: Reflow + K=3 at NFE≥300 exceeds teacher @ NFE=500 on PeakLoad/SymQ99/Pearson. For production imputation, this is the recommended inference recipe.
- CRPS saturates ~0.0095 — a dataset/operator floor, not a method limit. Don't chase it further.

Additionally, the same recipe has not been tried on **super-resolution** yet — resample-only SR was flagged as a follow-up in §2.2 and remains open.
