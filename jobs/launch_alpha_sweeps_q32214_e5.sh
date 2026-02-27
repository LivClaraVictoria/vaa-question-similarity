#!/bin/bash
# Launcher: Alpha sweeps for E5 model vs all Q32214 clone types.
# Assumes clone datasets already exist (run launch_create_clones_q32214.sh first).
# Run with: bash jobs/launch_alpha_sweeps_q32214_e5.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/alpha_sweep_results"

export CONFIG_A="configs/full_pipeline/base_data/pipeline_e5_ZH.py"

CLONE_TYPES=("identical_q32214_n10" "easy_paraphrase_q32214_n10" "hard_paraphrase_q32214_n10" "negation_q32214_n10" "negation_easy_q32214_n10" "negation_hard_q32214_n10" "natural_mixed_q32214_n5")

echo "=== Alpha Sweeps: E5 vs Q32214 clones ==="
echo ""

for CLONE in "${CLONE_TYPES[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    export CONFIG_B="configs/full_pipeline/cloned/${CLONE}_e5_ZH.py"
    SUBFOLDER="alpha_sweep_pipeline_e5_ZH_vs_${CLONE}_e5_ZH"
    export SWEEP_DIR="${RESULTS_BASE}/${SUBFOLDER}/workers_${TIMESTAMP}"
    mkdir -p "${SWEEP_DIR}"

    SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-20 --job-name="asw_${CLONE}_e5" "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
    COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} --job-name="asc_${CLONE}_e5" "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
    echo "  ${CLONE}: sweep=${SWEEP_JOB}, collect=${COLLECT_JOB}"
    sleep 1  # ensure unique timestamps
done

echo ""
echo "=== 7 alpha sweep batches submitted (E5) ==="
echo "  Monitor: squeue -u \$USER"
