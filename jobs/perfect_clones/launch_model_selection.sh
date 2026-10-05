#!/bin/bash
# Launcher: model selection alpha sweep — compares base vs cloned pipeline across the alpha grid.
# Submits one SLURM array job (one worker per alpha), then a dependent collect job.
# Override CONFIG_A / CONFIG_B by setting env vars before calling.
# Override the alpha grid with ALPHAS="0.01,0.1,0.2" (comma-separated); default = DEFAULT_ALPHAS in experiments/_common.py.
#
# Run with: bash jobs/perfect_clones/launch_model_selection.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../_lib/common.sh"
cd "${PROJECT_DIR}"
SWEEP_DIR="$(results_root)/exp1/model_alpha_sweep/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

# Default configs — override via env vars before calling this script
export CONFIG_A="${CONFIG_A:-configs/base_pipeline/pipeline_e5_ZH.py}"
export CONFIG_B="${CONFIG_B:-configs/experiments/perfect_clones_model_selection/identical_highcandvar_n10_e5_ZH.py}"
export SWEEP_DIR

export ALPHAS   # empty = model_selection.py falls back to DEFAULT_ALPHAS

activate_env
# Resolve the alpha list exactly as model_selection.py does, so array size always matches.
N_ALPHAS=$(python -c "
import os
from experiments._common import DEFAULT_ALPHAS
raw = os.environ.get('ALPHAS', '')
print(len([a for a in raw.split(',') if a.strip()]) if raw.strip() else len(DEFAULT_ALPHAS))
")
MAX_IDX=$((N_ALPHAS - 1))

echo "=== Model Selection Alpha Sweep ==="
echo "  Config A: ${CONFIG_A}"
echo "  Config B: ${CONFIG_B}"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Alphas: ${N_ALPHAS} (array 0-${MAX_IDX}): ${ALPHAS:-<DEFAULT_ALPHAS>}"

# Workers: one per alpha value
SWEEP_JOB=$(submit --array=0-${MAX_IDX} \
    --job-name="model_sel_sweep" \
    "${SCRIPT_DIR}/job_model_selection_worker.sh")
echo "  Workers submitted: job array ${SWEEP_JOB} (${N_ALPHAS} tasks)"

# Collect: aggregates per-alpha CSVs + plots (depends on all workers)
COLLECT_JOB=$(submit \
    --dependency=afterok:${SWEEP_JOB} \
    "${SCRIPT_DIR}/job_model_selection_collect.sh")
echo "  Collect submitted:  job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo ""
echo "  Monitor: squeue -u \$USER"
