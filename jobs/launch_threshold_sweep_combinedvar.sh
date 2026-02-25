#!/bin/bash
# Launcher: threshold alpha sweep for base E5 ZH vs identical_combinedvar_n10.
# Computes the number of alpha values dynamically from the min non-clone distance,
# then submits a job array (one per alpha) + a dependent collect job.
# Run with: bash jobs/launch_threshold_sweep_combinedvar.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"
SWEEP_DIR="${PROJECT_DIR}/experiment_results/threshold_alpha_sweep_results/workers_combinedvar_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export CONFIG_A="configs/full_pipeline/base_data/pipeline_e5_ZH.py"
export CONFIG_B="configs/full_pipeline/cloned/identical_combinedvar_n10_e5_ZH.py"
export SWEEP_DIR

# Determine number of alpha values from the threshold
cd "${PROJECT_DIR}"

# Activate conda for the Python call
[[ -f /itet-stor/liweiss/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/liweiss/net_scratch/conda/bin/conda shell.bash hook)"
conda activate bachelor-thesis

N_ALPHAS=$(python -c "
from threshold_alpha_sweep_main import compute_threshold_and_alphas
from main import load_config
from pathlib import Path
config_b = load_config(Path('${CONFIG_B}'))
_, alphas = compute_threshold_and_alphas(config_b)
print(len(alphas))
" 2>&1 | tail -1)
MAX_IDX=$((N_ALPHAS - 1))

echo "=== Threshold Alpha Sweep: combinedvar ==="
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Alpha count: ${N_ALPHAS} (array 0-${MAX_IDX})"

SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-${MAX_IDX} "${SCRIPT_DIR}/job_threshold_sweep_worker.sh")
echo "  Workers submitted: job array ${SWEEP_JOB} (${N_ALPHAS} tasks)"

COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} "${SCRIPT_DIR}/job_threshold_sweep_collect.sh")
echo "  Collect submitted:  job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo "  Monitor: squeue -u \$USER"
