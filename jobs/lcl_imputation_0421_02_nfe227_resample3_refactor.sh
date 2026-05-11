#!/bin/bash

#SBATCH --job-name="lcl-imput-0421-02-nfe227-K3-refactor"
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=n.lin@tudelft.nl

# Verify the FMSolver / AutocastForwardWrapper refactor:
#   1) Run the unit tests covering the refactored sampling pipeline
#      (FMSolver Euler bit-equivalence with ODESolver, time-grid helpers,
#      RePaint resample call counts, AutocastForwardWrapper fp32 contract,
#      and the existing month-mask suite to catch regressions).
#   2) Re-run the iso-compute MNAR imputation experiment from
#      lcl_imputation_0421_02_nfe227_resample3.sh with the new code and
#      write to a NEW output dir so the previous bf16-everywhere results
#      stay available for direct comparison.

module load 2025

cd ~/projects/SmartMeterFM

# init venv
source .venv/bin/activate
uv sync

TIMEID="LCL-0421-02"
CKPT="checkpoints/${TIMEID}/last.ckpt"

# ---------------------------------------------------------------------------
# 1) Unit tests (fast — should complete in < 3 min)
# ---------------------------------------------------------------------------
echo "**************** [${TIMEID}] running unit tests (FMSolver + AutocastForwardWrapper) **************************"
uv run pytest tests/test_fmsolver.py tests/test_month_mask.py -v
TEST_EXIT=$?
if [ $TEST_EXIT -ne 0 ]; then
    echo "**************** [${TIMEID}] UNIT TESTS FAILED (exit $TEST_EXIT). Aborting before real run. **************************"
    exit $TEST_EXIT
fi
echo "**************** [${TIMEID}] unit tests PASSED. Proceeding to real experiment. **************************"

# ---------------------------------------------------------------------------
# 2) Real experiment: NFE=227 K=3 MNAR imputation with the refactored code
#    Same params as jobs/lcl_imputation_0421_02_nfe227_resample3.sh.
#    Output dir suffixed with `_refactor` so you can diff against the existing
#    `mnar_20_nfe227_resample3/` directory produced by the bf16-everywhere
#    code path.
#    Iso-compute reminder: net queries = 227 * (1 + 3*0.4) = 499.4 ~= NFE=500 K=1.
# ---------------------------------------------------------------------------
echo "**************** [${TIMEID}] starting real imputation run (refactored: surgical bf16 + FMSolver) **************************"

uv run python scripts/showcase/imputation_demo.py \
    --checkpoint $CKPT \
    --dataset lcl_electricity \
    --data_root data/lcl_electricity/ \
    --imputation_type mnar_consecutive \
    --missing_rate 0.2 \
    --num_samples 10 \
    --num_test_series 100 \
    --num_steps 227 \
    --resample_steps 3 \
    --resample_t_threshold 0.4 \
    --time_grid_mode uniform \
    --output_dir results/imputation/${TIMEID}/mnar_20_nfe227_resample3_refactor

echo "**************** [${TIMEID}] MNAR imputation NFE=227 K=3 (refactor: surgical bf16 + FMSolver) completed. **************************"
echo ""
echo "Compare against existing results:"
echo "  diff -r results/imputation/${TIMEID}/mnar_20_nfe227_resample3 \\"
echo "          results/imputation/${TIMEID}/mnar_20_nfe227_resample3_refactor"
echo "Or load both summary CSVs and compare CRPS / quantile error / peak load err / pearsonr."
