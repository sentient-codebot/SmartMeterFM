# Metrics Review: `evaluate_samples.py`, `imputation_demo.py`, `super_resolution_demo.py`

This review walks through every metric computed by the three showcase scripts,
explains exactly how the numbers are produced (shapes, reductions, what is
compared against what), and flags the caveats — cases where the number is
computed correctly but *measures something different from what the name
suggests*.

The underlying primitives live in `src/smartmeterfm/utils/eval.py`. All
references below use line numbers from that file.

---

## 1. Shared data shape conventions

Understanding the shapes is half the review:

- Datasets use patchified storage: a single sample is `[num_patches, patch_size]`.
  For LCL (30-min resolution, monthly segment) this is `[93, 16]` → 1488
  real-time time steps.
- **Generation / evaluation** (`evaluate_samples.py`) works on batches of
  patchified tensors `[N, num_patches, patch_size]`.
- **Imputation / SR demos** unpatchify per-sample and work in the flat real
  time domain: imputed/SR samples are `[num_samples, seq_len]`, targets are
  `[seq_len]`. So the two pipelines feed metric primitives with very
  different shape semantics.

`tensor.flatten(1)` on a `[N, 93, 16]` tensor yields `[N, 1488]` and
**preserves temporal order** (patch index is outer, within-patch inner),
which matters for any metric that looks at temporal structure.

---

## 2. `evaluate_samples.py`

Tensors entering `MultiMetric(...)`:
- `source` (generated): `[1000, 93, 16]` per month
- `target` (real): `[~1350, 93, 16]` per month (2013-filtered)

### 2.1 `MkMMD` (`eval.py:77`)

**What it computes.** Multi-kernel MMD with `num_kernel=1`, `kernel_mul=2.0`,
`coefficient="auto"` (single RBF with Hamming weight 1.0 after
normalization → effectively one kernel).

**Shape handling.** `flatten_to_2d` turns `[N, 93, 16]` into `[N, 1488]` by
flattening the last two dims. Each sample is treated as a single 1488-dim
vector. Bandwidth is auto-computed from target–target pairwise L2².

**Caveats.**
- Bandwidth is set from `target`-only self-distances (`eval.py:150`), not
  from the median-heuristic over the pooled set. Fine for single-kernel
  use, but sensitive to target batch size.
- `num_kernel=1` means the "Mk" in MkMMD is a misnomer here — you are
  measuring single-RBF MMD. The metric itself is valid; the name overstates
  what's happening.
- The RBF operates in a 1488-dimensional space; MMD magnitudes are not
  directly comparable across datasets with different `seq_len`.

### 2.2 `DirectFD` (`calculate_frechet`, `eval.py:447`) — ⚠ subtle behavior

**What it computes.** Fréchet distance between two Gaussians fitted to
flattened samples, `FD = ||μ₁−μ₂||² + tr(Σ₁+Σ₂ − 2(Σ₁Σ₂)^{1/2})`.

**Shape handling (the caveat).** When inputs are 3D, it does:
```
source = source[:, shape[1] // 2, :]    # keeps ONE patch slice
```
With `[N, 93, 16]` this takes patch index **46** only, collapsing the
comparison to an `[N, 16]` distribution. The name `DirectFD` suggests the
whole trajectory is compared, but only a single mid-sequence 16-sample
window is used.

**Implication.** The FD value is a Fréchet distance of **one specific
half-hour patch in the middle of the month**, not of the full profile. If
you want full-sequence Fréchet, flatten to `[N, 1488]` before calling, or
patch the helper to use `.flatten(1)` instead of slicing.

### 2.3 `kl_divergence` (`eval.py:302`), `ws_distance` (`eval.py:358`), `ks_test_d/p` (`eval.py:402`)

**What they compute.** All three **fully flatten** both tensors (`flatten()`
with no dim arg → 1D) and compare the resulting scalar-valued empirical
distributions of every number in the tensor.

- `kl_divergence`: 200-bin histogram KL(source || target) with zero-bin
  masking and renormalization.
- `ws_distance`: scipy `wasserstein_distance` over the two 1D arrays.
- `ks_test_d/p`: scipy `ks_2samp` → D statistic and p-value.

**Caveats.**
- These measure only the **marginal value distribution** of "normalized
  power" across all samples and all times. They are insensitive to
  temporal structure, inter-sample variability, daily/seasonal cycles,
  correlation, etc. Two models with very different trajectories can score
  identical KL/WS/KS if marginals match.
- The KL has a silent left-out-mass issue: bins with `p_source = 0 ∧ p_target > 0`
  (or vice-versa) are zeroed then the remaining mass is renormalized
  (`eval.py:338-345`). That means support mismatches disappear instead of
  producing +∞.
- `ks_test_p` is effectively always 0 at the sample sizes used here
  (1k × 1488 values vs 1.3k × 1488) — you will see `0.000000` regardless
  of quality (visible in the log for every month).

### 2.4 `source_mean / source_std / target_mean / target_std`

Scalar mean/std over every element of the tensor. Useful only as a sanity
check that global scale has not drifted. Not a distributional metric.

### 2.5 `CRPS` via `crps_metric` (script line 108)

```python
crps_empirical_vs_empirical(target.flatten(1), source.flatten(1))
```

**What it computes.** Per-dimension CRPS between two empirical
distributions, then averaged.

- Inputs are `[N_target, 1488]` and `[N_source, 1488]`.
- `crps_empirical_vs_empirical` (`eval.py:734`) iterates over
  `D = 1488` positions and for each position computes
  `E|X − Y| − ½(E|X − X'| + E|Y − Y'|)` between the two empirical
  marginals at that time step.
- The result is averaged to a single scalar.

**Caveats.**
- This is the CRPS between **marginal distributions at each time step**,
  averaged over time. It is a legitimate quantity, but it is *not* the
  classical per-trajectory CRPS (source is a batch of forecasts, target
  is a single truth). Two models with different joint structure but
  matching marginals will look identical.
- Cost is O(D · (N_src² + N_src·N_tgt + N_tgt²)); with D=1488 and
  N≈1000 this is by far the most expensive metric — and it explains the
  ~5-minute "Overall" step in the log (`16:04:44` minus `15:59:47`).

### 2.6 `PearsonR_diff` via `pearsonr_metric` (script line 117) — ⚠ naming

**What it computes.**
1. `metric_pearsonr(source, target, shift=1)` (`eval.py:838`) flattens each
   tensor to `[N, 1488]`, concatenates into `[N_src+N_tgt, 1488]`, and for
   every row runs scipy `pearsonr(x[:−1], x[1:])` → lag-1 autocorrelation
   per sample.
2. Script filters NaNs (see §5), takes the mean across samples of each
   side, and returns `|mean_src_acorr − mean_tgt_acorr|`.

**Caveats.**
- The "PearsonR difference" is actually the **absolute difference of
  mean sample-level lag-1 autocorrelation**. It is a summary of temporal
  smoothness, not a per-sample Pearson comparison.
- Patch boundaries are crossed: index 15 → 16 steps from "last element
  of patch 0" to "first element of patch 1". Because the patchification
  is a reshape of a contiguous time sequence, this lag-1 is still a real
  temporal lag-1; no caveat there.
- For LCL with `seq_len=1488`, taking lag-1 of the whole flattened profile
  mixes intra-day and inter-day correlation into one scalar.

### 2.7 `PeakLoadError` via `peak_load_error_metric` (script line 134)

```python
s_peaks = src.flatten(1).max(dim=1)  # [N_src, 1]
t_peaks = tgt.flatten(1).max(dim=1)  # [N_tgt, 1]
pos = crps_empirical_vs_empirical(t_peaks, s_peaks).mean()
neg = ... (on -src, -tgt)
return pos + neg
```

**What it computes.** CRPS between the distribution of per-sample maxima
(and, on negated values, per-sample minima) of source vs target. Two
empirical distributions of peaks → legitimate CRPS.

**Caveat.** This is the distribution-level peak error and matches the
many-real-samples setting. Note that the *local* implementation here
(script line 134) differs in semantics from the imported
`peak_load_error` in `eval.py:891`, which expects a **single** target
trajectory — see §3.3.

### 2.8 `SymQuantileError_99` via `sym_quantile_metric` (script line 150)

```python
s_q = torch.quantile(src_2d, 0.99, dim=1, keepdim=True)  # [N_src, 1]
t_q = torch.quantile(tgt_2d, 0.99, dim=1, keepdim=True)  # [N_tgt, 1]
upper = crps_empirical_vs_empirical(t_q, s_q).mean()
lower = ... (0.01 quantile)
return upper + lower
```

**What it computes.** CRPS between per-sample 99% quantiles (and 1%
quantiles) of the two distributions. Same "many targets" shape story as
`PeakLoadError`.

**Caveat.** The docstring claims this avoids a "dim=0 issue in
`sym_quantile_error` with 2D targets". That is accurate: the imported
`sym_quantile_error` treats the target as a single sequence and uses
`dim=0`, which silently does the wrong thing for 2D real-target
tensors. **This local version is the correct one for the
generation/evaluation setting**; do not replace it with the imported
helper.

### 2.9 Overall vs per-month aggregation

`compute_metrics_per_month` runs the full metric stack on each month
separately, then again on the concatenation of all months
(`all_gen = cat(...)`, `all_real = cat(...)`). The "Overall" numbers are
therefore **pooled-distribution** metrics, not averages of per-month
metrics — the difference matters for nonlinear metrics like KL and MMD.

---

## 3. `imputation_demo.py`

Tensors per test series:
- `original_real` (HR): `[seq_len]`
- `imputed_real`: `[num_samples, seq_len]` (default 10)
- `baseline_real`: `[seq_len]` (linear interpolation)
- `mask`: `[seq_len]` (1=observed, 0=missing)

### 3.1 MSE / MAE / RMSE (at missing positions)

```python
mean_imputed = imputed.mean(dim=0)          # [seq_len]
original_missing  = original[missing_positions]
mean_imp_missing  = mean_imputed[missing_positions]
mse = ((original_missing - mean_imp_missing)**2).mean()
```

**What it computes.** Errors of the posterior-mean reconstruction at
missing positions only. RMSE = sqrt(MSE) per series; then averaged across
series in the final summary.

**Caveats.**
- Using the **mean of samples** discards all probabilistic information
  the flow model provides. A model with larger but well-calibrated
  variance can score worse on MSE/MAE than a deterministic baseline
  while producing better CRPS — this is why the user memory note prefers
  CRPS / quantile / peak-load metrics.
- Final aggregation is an arithmetic mean of per-series scalars
  (script line 543). A series with 50% missing and one with 10% missing
  contribute equally to the average.

### 3.2 Uncertainty

```python
imputed[:, missing].std(dim=0).mean()
```

Std across samples at each missing position, averaged over missing
positions. Correct.

### 3.3 CRPS at missing positions

```python
crps_per_dim = crps_empirical_dimwise(imputed, original_flat)
crps_missing = crps_per_dim[missing_flat].mean()
```

**What it computes.** `crps_empirical_dimwise` (`eval.py:697`) computes
per-dim CRPS where the target is a **single value** per dim and the
source is an empirical distribution over `num_samples` values:
`CRPS_d = 2·mean_i [ τ_i·Δ⁺ + (τ_i−1)·Δ⁻ ]` with `τ_i = (2i−1)/(2B)`.

At `B=10` this gives a coarse CRPS; at `B=1` (the baseline batch,
`baseline_batch = baseline.flatten().unsqueeze(0)`), `τ = 0.5` and the
formula collapses to `CRPS = |source − target|` — i.e. the "baseline CRPS"
is exactly the **MAE of the baseline**, not a sharpness-aware score.
This is a property of the formula with B=1, not a bug, but it does mean
that "CRPS_baseline" is just MAE at the same positions. Keep that in
mind when reporting "CRPS vs baseline" — the baseline cannot benefit
from calibration because it has no distribution.

### 3.4 `peak_load_error` (imported, `eval.py:891`) — ⚠ per-series semantics

```python
ple = peak_load_error(imputed, original_flat)        # [num_samples, seq_len] vs [seq_len]
ple_baseline = peak_load_error(baseline_batch, original_flat)  # [1, seq_len] vs [seq_len]
```

**What it computes.** Inside, it compares:
- `source_pos_peaks` (per-sample max): `[num_samples, 1]`
- `target_pos_peak`  (scalar max): `[1, 1]`

via `crps_empirical_dimwise(source, target.squeeze(0))` → a CRPS where
the "truth" is the single scalar peak of the single ground-truth
trajectory and the "forecast" is the empirical distribution of
sample peaks. Sum of positive-peak and negative-peak CRPS.

**Caveat.** This is a **peak CRPS for a single trajectory**, not a
distribution-of-peaks comparison. That matches the imputation setting
(one ground truth per series) but is categorically different from the
`PeakLoadError` in `evaluate_samples.py` (§2.7). Do not cross-compare the
two numbers across scripts.

### 3.5 `sym_quantile_error` (imported, `eval.py:937`) — ⚠ quirk with 2D targets

The imported helper does:
```python
target_pos_quantile = torch.quantile(target, q, dim=0, keepdim=True)
```
With `target = original_flat` (shape `[seq_len]`), `dim=0` is the only
axis so `quantile(target, 0.99, dim=0)` is the scalar 99th percentile of
the trajectory — **correct here**.

But if you ever passed a 2D target (e.g. `[N, seq_len]`) into this
helper, `dim=0` would take the quantile **across samples at each time
position**, giving a `[seq_len]` vector — a very different object. This
is the bug the local `sym_quantile_metric` in `evaluate_samples.py`
works around (§2.8). Fine in the imputation/SR demos because target is
always 1D; risky if the helper is reused elsewhere.

### 3.6 Pearson R diff

```python
source_acorr   = calculate_pearsonr_per_sample(imputed, shift=1).mean()   # over samples
target_acorr   = calculate_pearsonr_per_sample(original_flat.unsqueeze(0), shift=1).item()
pearsonr_diff  = abs(source_acorr - target_acorr)
```

**What it computes.** Mean lag-1 autocorrelation across the 10 imputed
samples, minus the single lag-1 autocorrelation of the target, absolute
value. Same shape story as §2.6 but per series.

**Caveat.** The imputed samples usually preserve observed positions
exactly (the mask is enforced), so their autocorrelation is dominated by
the observed skeleton; this makes the metric less sensitive to what the
model actually filled in. You may want a masked-only variant.

---

## 4. `super_resolution_demo.py`

Tensors per test series:
- `original_real`: `[seq_len]`
- `sr_samples_real`: `[num_samples, seq_len]` (default 10)
- `baseline_real`: `[seq_len]` (F.interpolate linear)

### 4.1 MSE / MAE / RMSE (full sequence)

Unlike imputation, SR evaluates the **whole** sequence (not just
upsampled positions), because every position is a model output:

```python
mean_sr = sr_samples.mean(dim=0)
mse_sr  = ((original_hr - mean_sr)**2).mean()
```

Reasonable for SR: observed LR points are reconstructed by averaging
each HR block, so there isn't a clean "observed" vs "unobserved"
partition.

### 4.2 Uncertainty and Improvement

- `uncertainty = sr_samples.std(dim=0).mean()` — same as imputation.
- `improvement_mse_pct = (mse_baseline − mse_sr) / mse_baseline · 100`.
  A single per-series percentage that is then **averaged** across series
  — this is **not** the percentage improvement of the average MSE (which
  would require averaging MSEs first, then taking the ratio). With
  highly skewed per-series MSE the two differ noticeably.

### 4.3 CRPS / Peak / SymQuantile / Pearson

Implementations identical to imputation_demo (§3.3–3.6). All the same
caveats apply, including:
- Baseline CRPS = MAE at B=1.
- `peak_load_error` = per-series peak CRPS (single truth, sample
  distribution) — different semantics from `evaluate_samples.py` (§2.7).
- `sym_quantile_error` with 1D target is safe.

### 4.4 Aggregation

`avg_metrics[key] = sum(m[key]) / len(all_metrics)` — simple arithmetic
mean across test series. No weighting, no per-month breakdown.

---

## 5. The `ConstantInputWarning` / PearsonR NaN warning

From the log (`slurm_22081445.log`):

```
eval.py:820: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
WARNING - PearsonR: 0/1000 source and 1/1372 target samples produced NaN (constant sequences), filtering them out
...
WARNING - PearsonR: 0/12000 source and 17/15873 target samples produced NaN (constant sequences), filtering them out
```

### What is happening

`calculate_pearsonr_per_sample` (`eval.py:779`) reshapes each flattened
sample into two lag-shifted arrays `x1 = x[:, :-1]`, `x2 = x[:, 1:]` and
calls `scipy.stats.pearsonr(x1[i], x2[i])` per row. `pearsonr` is
undefined when either input has zero variance (constant array); scipy
emits `ConstantInputWarning` once and returns `NaN`.

At `seq_len = 1488` for LCL monthly, "constant" means the **entire
1488-step trajectory** is constant. Realistically that only happens when:
- a household meter was offline the whole month and the dataset
  imputation filled with a single value;
- a household had zero consumption for the full month (vacant);
- a normalization/preprocessing issue collapsed a series to a constant.

Counts in the log bear this out: **0 source samples** are ever constant
(flow generations always have variance), and **1–4 real target samples
per month** are constant, summing to 17/15873 overall — a ~0.1% rate
consistent with sporadic meter faults.

### How the script handles it

`pearsonr_metric` in `evaluate_samples.py` filters NaN entries before
averaging:

```python
src_mean = src[~torch.isnan(src)].mean()
tgt_mean = tgt[~torch.isnan(tgt)].mean()
```

So the reported `PearsonR_diff` is computed only from non-constant
samples. Correct.

### Why the `ConstantInputWarning` appears only once

scipy filters `ConstantInputWarning` via `warnings.warn` with the
default filter, which usually shows the warning **once per call site**.
That's why you see the one warning line followed by the summary
"1/1372 target samples produced NaN" rather than dozens of warnings.

### Recommendations

1. **Fine as-is for evaluation.** The filtering is correct; the result
   is not biased beyond the loss of a few constant samples.
2. **Optional root-cause cleanup.** Add a filter at data-load time to
   drop series with `std < eps` from the LCL test set — this would
   remove the warning entirely and make counts stable across runs.
3. **Silencing.** If the log spam is undesirable, wrap the scipy call in
   `with warnings.catch_warnings(): warnings.simplefilter("ignore", ConstantInputWarning)`
   around `pearsonr` in `calculate_pearsonr_per_sample`.
4. **Per-dim alternative.** For the evaluate script, consider comparing
   the *distributions* of per-sample autocorrelations (e.g. CRPS or
   Wasserstein on the two acorr vectors) instead of the absolute
   difference of means — this retains information about the spread of
   temporal structure, not just its average.

---

## 6. Cross-script inconsistencies to be aware of

| Metric name           | `evaluate_samples.py` semantics                     | `imputation_demo.py` / `super_resolution_demo.py` semantics |
|-----------------------|------------------------------------------------------|--------------------------------------------------------------|
| `PeakLoadError`       | CRPS between two empirical distributions of peaks   | CRPS of sample-peak distribution vs single target peak       |
| `SymQuantileError_99` | CRPS between two empirical distributions of q99     | CRPS of sample-q99 distribution vs single target q99         |
| `CRPS`                | Per-time-step marginal CRPS between two populations | Per-position CRPS of `num_samples` vs single truth           |
| `PearsonR diff`       | \|mean acorr(gen) − mean acorr(real)\|              | \|mean acorr(samples) − acorr(single truth)\|                |
| `baseline CRPS`       | n/a                                                  | Equals MAE at B=1 — not a sharpness metric                   |

These are consistent within each script but measure different things
across scripts. Do **not** plot them on the same axis.

---

## 7. Summary of potential issues (in priority order)

1. **`DirectFD` silently collapses a 3D tensor to a single mid-sequence
   patch slice** (`eval.py:457-462`). The reported Fréchet distance is
   over `[N, 16]`, not `[N, 1488]`. Consider flattening instead, or
   renaming to `DirectFD_midpatch` to reflect the actual semantics.
2. **`ks_test_p` is effectively always 0** at the sample sizes used. It
   adds noise to the output without information; drop it or replace
   with an effect size.
3. **"MkMMD" with `num_kernel=1`** is single-RBF MMD. Either increase
   `num_kernel` (e.g. 5 with `coefficient="auto"` which defaults the
   Hamming weighting) or rename.
4. **KL/WS/KS are marginal-only** — they ignore temporal structure. They
   are fine as light-weight sanity checks but should not be load-bearing
   for model selection.
5. **"Improvement %" in the demos is averaged-of-ratios**, not
   ratio-of-averages. Document or change.
6. **`improvement_mse_pct` degenerate when `mse_baseline == 0`**:
   returns 0. Unlikely but worth noting.
7. **Pearson NaN handling.** Already correct (filtered), but the
   `ConstantInputWarning` log noise can be silenced and a preprocessing
   filter for zero-variance series would be cleaner (§5).
8. **Per-series arithmetic averaging** in the demos weights all series
   equally regardless of missing-rate / length. Acceptable default but
   should be stated in any publication.
