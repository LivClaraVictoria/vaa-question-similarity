#!/bin/bash
# Launcher: model selection alpha sweep — compares base vs cloned pipeline across 21 alpha values.
# Submits one SLURM array job (21 workers, one per alpha), then a dependent collect job.
# Override CONFIG_A / CONFIG_B by setting env vars before calling.
#
# Run with: bash jobs/perfect_clones/launch_model_selection.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"
SWEEP_DIR="${PROJECT_DIR}/experiment_results/exp1/model_alpha_sweep/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

# Default configs — override via env vars before calling this script
export CONFIG_A="${CONFIG_A:-configs/base_pipeline/pipeline_e5_ZH.py}"
export CONFIG_B="${CONFIG_B:-configs/experiments/perfect_clones_model_selection/identical_highcandvar_n10_e5_ZH.py}"
export SWEEP_DIR

N_ALPHAS=21
MAX_IDX=$((N_ALPHAS - 1))   # 21 alphas (indices 0–20): 0.01, 0.1–1.5 step 0.1, 1.8–3.0 step 0.3

echo "=== Model Selection Alpha Sweep ==="
echo "  Config A: ${CONFIG_A}"
echo "  Config B: ${CONFIG_B}"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Alphas: ${N_ALPHAS} (array 0-${MAX_IDX})"

# Workers: one per alpha value
SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-${MAX_IDX} \
    --job-name="model_sel_sweep" \
    "${SCRIPT_DIR}/job_model_selection_worker.sh")
echo "  Workers submitted: job array ${SWEEP_JOB} (${N_ALPHAS} tasks)"

# Collect: aggregates per-alpha CSVs + plots (depends on all workers)
COLLECT_JOB=$(sbatch --parsable --export=ALL \
    --dependency=afterok:${SWEEP_JOB} \
    "${SCRIPT_DIR}/job_model_selection_collect.sh")
echo "  Collect submitted:  job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo ""
echo "  Monitor: squeue -u \$USER"
