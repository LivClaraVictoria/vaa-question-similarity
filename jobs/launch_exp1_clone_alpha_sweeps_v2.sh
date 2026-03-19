#!/bin/bash
# Launcher: Experiment 1 v2 — Alpha sweeps for all embedding models x 5 clone conditions.
# Uses updated top-5 impact questions (32214, 32228, 32234, 32240, 32261) with 4 clones each.
#
# Phase 1: Creates 3 NEW cloned datasets (serial, paraphrase cache safety).
#   easy_paraphrase and negation_easy datasets already exist from prior run.
# Phase 2: Submits 30 alpha sweep batches in 3 SEQUENTIAL BATCHES (1 per clone type)
#   to reduce peak NFS IO. Each batch = 10 models × 21 workers running in parallel.
#   Existing easy_paraphrase + negation_easy results (20 sweeps) are kept.
#
# Results saved to experiment_results/exp1/model_alpha_sweep/top5impact_v2/
#
# Requires OPENAI_API_KEY in environment for paraphrase generation.
# Run with: bash jobs/launch_exp1_clone_alpha_sweeps_v2.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/exp1/model_alpha_sweep/top5impact_v2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "  Experiment 1 v2: Clone Alpha Sweeps"
echo "  Top-5 questions: 32214, 32228, 32234, 32240, 32261"
echo "  3 NEW clone conditions x 10 embedding models"
echo "  3 sequential batches to limit NFS IO"
echo "  (easy_paraphrase + negation_easy already done)"
echo "  Results: ${RESULTS_BASE}"
echo "  Timestamp: ${TIMESTAMP}"
echo "=========================================="
echo ""

# ============================================================
# Phase 1: Create clone datasets (serial due to paraphrase cache)
# ============================================================

echo "--- Phase 1: Creating clone datasets (serial chain) ---"

export CLONE_CONFIG="configs/create_clones/hard_paraphrase_top5impact_v2_n4.py"
CLONE1=$(sbatch --parsable --export=ALL --job-name=clone_hp_v2 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  hard_paraphrase_top5impact_v2_n4:  job ${CLONE1}"

export CLONE_CONFIG="configs/create_clones/negation_hard_top5impact_v2_n4.py"
CLONE2=$(sbatch --parsable --export=ALL --dependency=afterok:${CLONE1} --job-name=clone_nh_v2 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation_hard_top5impact_v2_n4:    job ${CLONE2} (after ${CLONE1})"

export CLONE_CONFIG="configs/create_clones/perfect_mix_top5impact_v2_n4.py"
CLONE3=$(sbatch --parsable --export=ALL --dependency=afterok:${CLONE2} --job-name=clone_pm_v2 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  perfect_mix_top5impact_v2_n4:      job ${CLONE3} (after ${CLONE2})"

echo ""

# ============================================================
# Phase 2: Alpha sweeps in 3 sequential batches (1 per clone type)
# Each batch: 10 models in parallel, each with 21-task array + collect
# Next batch starts only after ALL collects in previous batch finish
# ============================================================

echo "--- Phase 2: Submitting alpha sweeps in 3 batches ---"
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

# Start first batch after clone creation
BATCH_DEPENDENCY="afterok:${CLONE3}"
TOTAL_SWEEPS=0

for CLONE in "${CLONE_CONDITIONS[@]}"; do
    echo "  === Batch: ${CLONE} (depends on ${BATCH_DEPENDENCY}) ==="
    COLLECT_JOB_IDS=""

    for i in "${!MODELS[@]}"; do
        MODEL="${MODELS[$i]}"
        export CONFIG_A="${BASE_CONFIGS[$i]}"
        export CONFIG_B="configs/full_pipeline/cloned/${CLONE}_${MODEL}_ZH.py"
        SUBFOLDER="alpha_sweep_pipeline_${MODEL}_ZH_vs_${CLONE}_${MODEL}_ZH"
        export SWEEP_DIR="${RESULTS_BASE}/${SUBFOLDER}/workers_${TIMESTAMP}"
        export OUTPUT_DIR="${RESULTS_BASE}"
        mkdir -p "${SWEEP_DIR}"

        SWEEP_JOB=$(sbatch --parsable --export=ALL --dependency=${BATCH_DEPENDENCY} \
            --array=0-20 --job-name="asw2_${MODEL:0:8}_${CLONE:0:6}" \
            "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
        COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} \
            --job-name="asc2_${MODEL:0:8}_${CLONE:0:6}" \
            "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")

        echo "    ${MODEL}: sweep=${SWEEP_JOB} (21 tasks), collect=${COLLECT_JOB}"

        # Accumulate collect job IDs for batch dependency
        if [ -z "${COLLECT_JOB_IDS}" ]; then
            COLLECT_JOB_IDS="${COLLECT_JOB}"
        else
            COLLECT_JOB_IDS="${COLLECT_JOB_IDS}:${COLLECT_JOB}"
        fi

        TOTAL_SWEEPS=$((TOTAL_SWEEPS + 1))
    done

    # Next batch depends on ALL collects of this batch completing
    BATCH_DEPENDENCY="afterok:${COLLECT_JOB_IDS}"
    echo ""
done

echo "=========================================="
echo "  Experiment 1 v2 (additional sweeps) submitted!"
echo "  Clone creation: 3 jobs (serial chain, last=${CLONE3})"
echo "  Alpha sweeps:   ${TOTAL_SWEEPS} sweeps in 3 sequential batches"
echo "  Per batch: 10 models x 21 workers (parallel) + 10 collects"
echo "  Total SLURM jobs: $((3 + TOTAL_SWEEPS * 22))"
echo "  Results dir: ${RESULTS_BASE}"
echo "  Monitor: squeue -u \$USER"
echo "=========================================="
