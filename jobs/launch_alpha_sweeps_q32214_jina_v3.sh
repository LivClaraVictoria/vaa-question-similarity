#!/bin/bash
# Launcher: Alpha sweeps for Jina v3 model vs all Q32214 clone types.
# Assumes clone datasets already exist (run launch_create_clones_q32214.sh first).
# Run with: bash jobs/launch_alpha_sweeps_q32214_jina_v3.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/alpha_sweep_results"

export CONFIG_A="configs/full_pipeline/base_data/pipeline_jina_v3_ZH.py"

CLONE_TYPES=("identical_q32214_n10" "easy_paraphrase_q32214_n10" "hard_paraphrase_q32214_n10" "negation_q32214_n10" "negation_easy_q32214_n10" "negation_hard_q32214_n10" "natural_mixed_q32214_n5")

echo "=== Alpha Sweeps: Jina v3 vs Q32214 clones ==="
echo ""

for CLONE in "${CLONE_TYPES[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    export CONFIG_B="configs/full_pipeline/cloned/${CLONE}_jina_v3_ZH.py"
    export SWEEP_DIR="${RESULTS_BASE}/workers_${CLONE}_jina_v3_${TIMESTAMP}"
    mkdir -p "${SWEEP_DIR}"

    SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-20 --job-name="asw_${CLONE}_jina" "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
    COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} --job-name="asc_${CLONE}_jina" "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
    echo "  ${CLONE}: sweep=${SWEEP_JOB}, collect=${COLLECT_JOB}"
    sleep 1  # ensure unique timestamps
done

echo ""
echo "=== 7 alpha sweep batches submitted (Jina v3) ==="
echo "  Monitor: squeue -u \$USER"
