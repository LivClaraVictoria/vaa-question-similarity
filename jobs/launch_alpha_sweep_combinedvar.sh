#!/bin/bash
# Launcher: alpha sweep for base E5 ZH vs identical_combinedvar_n10.
# Submits a 21-task job array (one per alpha) + a dependent collect job.
# Run with: bash jobs/launch_alpha_sweep_combinedvar.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWEEP_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/alpha_sweep_results/workers_combinedvar_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export CONFIG_A="configs/full_pipeline/base_data/pipeline_e5_ZH.py"
export CONFIG_B="configs/full_pipeline/cloned/identical_combinedvar_n10_e5_ZH.py"
export SWEEP_DIR

echo "=== Alpha Sweep: combinedvar ==="
echo "  Sweep dir: ${SWEEP_DIR}"

# 21 alpha values (indices 0-20)
SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-20 "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
echo "  Workers submitted: job array ${SWEEP_JOB} (21 tasks)"

COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
echo "  Collect submitted:  job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo "  Monitor: squeue -u \$USER"
