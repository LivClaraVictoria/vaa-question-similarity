#!/bin/bash
# Launcher: Alpha sweeps for all 4 models vs all Q32214 n5 clone types.
# Assumes n5 clone datasets already exist (run launch_create_clones_q32214_n5.sh first).
# Run with: bash jobs/launch_alpha_sweeps_q32214_n5.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/alpha_sweep_results"

echo "=========================================="
echo "  Q32214 n5 Alpha Sweeps (4 models x 6 clone types)"
echo "=========================================="
echo ""

MODELS=("e5" "jina_v3" "e5_instruct" "qwen3")
BASE_CONFIGS=(
    "configs/full_pipeline/base_data/pipeline_e5_ZH.py"
    "configs/full_pipeline/base_data/pipeline_jina_v3_ZH.py"
    "configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py"
    "configs/full_pipeline/base_data/pipeline_qwen3_ZH.py"
)
CLONE_TYPES=("negation_q32214_n5" "easy_paraphrase_q32214_n5" "hard_paraphrase_q32214_n5" "negation_easy_q32214_n5" "negation_hard_q32214_n5" "natural_mixed_q32214_n5")

SWEEP_COUNT=0
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    export CONFIG_A="${BASE_CONFIGS[$i]}"

    for CLONE in "${CLONE_TYPES[@]}"; do
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        export CONFIG_B="configs/full_pipeline/cloned/${CLONE}_${MODEL}_ZH.py"
        SUBFOLDER="alpha_sweep_pipeline_${MODEL}_ZH_vs_${CLONE}_${MODEL}_ZH"
        export SWEEP_DIR="${RESULTS_BASE}/${SUBFOLDER}/workers_${TIMESTAMP}"
        mkdir -p "${SWEEP_DIR}"

        SWEEP_JOB=$(sbatch --parsable --export=ALL \
            --array=0-20 --job-name="asw_${CLONE:0:8}_${MODEL:0:4}" \
            "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
        COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} \
            --job-name="asc_${CLONE:0:8}_${MODEL:0:4}" \
            "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
        echo "  ${MODEL} x ${CLONE}: sweep=${SWEEP_JOB}, collect=${COLLECT_JOB}"
        SWEEP_COUNT=$((SWEEP_COUNT + 1))
        sleep 1  # ensure unique timestamps
    done
done

echo ""
echo "=== Experiment submitted ==="
echo "  Alpha sweeps: ${SWEEP_COUNT} batches (each: 21 workers + 1 collect)"
echo "  Total SLURM jobs: $((SWEEP_COUNT * 22))"
echo "  Monitor: squeue -u \$USER"
