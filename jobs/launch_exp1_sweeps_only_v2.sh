#!/bin/bash
# Launcher: Experiment 1 v2 — Alpha sweeps ONLY (clone datasets already exist).
# 3 sequential batches (1 per clone type) to limit NFS IO.
#
# Run with: bash jobs/launch_exp1_sweeps_only_v2.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/exp1/model_alpha_sweep/top5impact_v2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "  Experiment 1 v2: Alpha Sweeps (no clone creation)"
echo "  Top-5 questions: 32214, 32228, 32234, 32240, 32261"
echo "  3 clone conditions x 10 models, 3 sequential batches"
echo "  Results: ${RESULTS_BASE}"
echo "  Timestamp: ${TIMESTAMP}"
echo "=========================================="
echo ""

MODELS=(       "sbert"   "e5"   "e5_asym"   "e5_instruct"   "e5_asym_instruct"   "jina_v3"   "bge_m3"   "gte"   "nomic_v2"   "qwen3")
BASE_CONFIGS=(
    "configs/full_pipeline/base_data/pipeline_sbert_ZH.py"
    "configs/full_pipeline/base_data/pipeline_e5_ZH.py"
    "configs/full_pipeline/base_data/pipeline_e5_asym_ZH.py"
    "configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py"
    "configs/full_pipeline/base_data/pipeline_e5_asym_instruct_ZH.py"
    "configs/full_pipeline/base_data/pipeline_jina_v3_ZH.py"
    "configs/full_pipeline/base_data/pipeline_bge_m3_ZH.py"
    "configs/full_pipeline/base_data/pipeline_gte_ZH.py"
    "configs/full_pipeline/base_data/pipeline_nomic_v2_ZH.py"
    "configs/full_pipeline/base_data/pipeline_qwen3_ZH.py"
)

CLONE_CONDITIONS=("hard_paraphrase_top5impact_v2_n4" "negation_hard_top5impact_v2_n4" "perfect_mix_top5impact_v2_n4")

# First batch has no dependency
BATCH_DEPENDENCY=""
TOTAL_SWEEPS=0

for CLONE in "${CLONE_CONDITIONS[@]}"; do
    if [ -z "${BATCH_DEPENDENCY}" ]; then
        echo "  === Batch: ${CLONE} (no dependency — starts immediately) ==="
    else
        echo "  === Batch: ${CLONE} (depends on ${BATCH_DEPENDENCY}) ==="
    fi
    COLLECT_JOB_IDS=""

    for i in "${!MODELS[@]}"; do
        MODEL="${MODELS[$i]}"
        export CONFIG_A="${BASE_CONFIGS[$i]}"
        export CONFIG_B="configs/full_pipeline/cloned/${CLONE}_${MODEL}_ZH.py"
        SUBFOLDER="alpha_sweep_pipeline_${MODEL}_ZH_vs_${CLONE}_${MODEL}_ZH"
        export SWEEP_DIR="${RESULTS_BASE}/${SUBFOLDER}/workers_${TIMESTAMP}"
        export OUTPUT_DIR="${RESULTS_BASE}"
        mkdir -p "${SWEEP_DIR}"

        DEP_FLAG=""
        [ -n "${BATCH_DEPENDENCY}" ] && DEP_FLAG="--dependency=${BATCH_DEPENDENCY}"

        SWEEP_JOB=$(sbatch --parsable --export=ALL ${DEP_FLAG} \
            --array=0-20 --job-name="asw2_${MODEL:0:8}_${CLONE:0:6}" \
            "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
        COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} \
            --job-name="asc2_${MODEL:0:8}_${CLONE:0:6}" \
            "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")

        echo "    ${MODEL}: sweep=${SWEEP_JOB} (21 tasks), collect=${COLLECT_JOB}"

        if [ -z "${COLLECT_JOB_IDS}" ]; then
            COLLECT_JOB_IDS="${COLLECT_JOB}"
        else
            COLLECT_JOB_IDS="${COLLECT_JOB_IDS}:${COLLECT_JOB}"
        fi

        TOTAL_SWEEPS=$((TOTAL_SWEEPS + 1))
    done

    BATCH_DEPENDENCY="afterok:${COLLECT_JOB_IDS}"
    echo ""
done

echo "=========================================="
echo "  Submitted ${TOTAL_SWEEPS} sweeps in 3 sequential batches"
echo "  Per batch: 10 models x 21 workers (parallel) + 10 collects"
echo "  Total SLURM jobs: $((TOTAL_SWEEPS * 22))"
echo "  Results dir: ${RESULTS_BASE}"
echo "  Monitor: squeue -u \$USER"
echo "=========================================="
