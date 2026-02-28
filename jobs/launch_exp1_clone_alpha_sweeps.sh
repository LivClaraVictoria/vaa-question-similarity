#!/bin/bash
# Launcher: Experiment 1 — Alpha sweeps for all embedding models x 3 clone conditions.
# Clones top-5 impact questions (32214, 32261, 32228, 32234, 32222) with 4 clones each.
# Phase 1: Creates 3 cloned datasets (serial, paraphrase cache safety).
# Phase 2: Submits 30 alpha sweep batches (3 conditions x 10 models, parallel).
# Requires OPENAI_API_KEY in environment for paraphrase generation.
# Run with: bash jobs/launch_exp1_clone_alpha_sweeps.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/alpha_sweep_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "  Experiment 1: Clone Alpha Sweeps"
echo "  3 clone conditions x 10 embedding models"
echo "  Timestamp: ${TIMESTAMP}"
echo "=========================================="
echo ""

# ============================================================
# Phase 1: Create clone datasets (serial due to paraphrase cache)
# ============================================================

echo "--- Phase 1: Creating clone datasets (serial chain) ---"

export CLONE_CONFIG="configs/create_clones/easy_paraphrase_top5impact_n4.py"
CLONE1=$(sbatch --parsable --export=ALL --job-name=clone_ep_top5 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  easy_paraphrase_top5impact_n4:  job ${CLONE1}"

export CLONE_CONFIG="configs/create_clones/negation_easy_top5impact_n4.py"
CLONE2=$(sbatch --parsable --export=ALL --dependency=afterok:${CLONE1} --job-name=clone_ne_top5 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation_easy_top5impact_n4:    job ${CLONE2} (after ${CLONE1})"

export CLONE_CONFIG="configs/create_clones/mixed_top5impact_n4.py"
CLONE3=$(sbatch --parsable --export=ALL --dependency=afterok:${CLONE2} --job-name=clone_mix_top5 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  mixed_top5impact_n4:            job ${CLONE3} (after ${CLONE2})"

echo ""

# ============================================================
# Phase 2: Alpha sweeps (all depend on clone creation completing)
# ============================================================

echo "--- Phase 2: Submitting alpha sweeps (after clone job ${CLONE3}) ---"
echo ""

# Model arrays (parallel indices)
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

CLONE_CONDITIONS=("easy_paraphrase_top5impact_n4" "negation_easy_top5impact_n4" "mixed_top5impact_n4")

SWEEP_COUNT=0
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    export CONFIG_A="${BASE_CONFIGS[$i]}"

    for CLONE in "${CLONE_CONDITIONS[@]}"; do
        export CONFIG_B="configs/full_pipeline/cloned/${CLONE}_${MODEL}_ZH.py"
        SUBFOLDER="alpha_sweep_pipeline_${MODEL}_ZH_vs_${CLONE}_${MODEL}_ZH"
        export SWEEP_DIR="${RESULTS_BASE}/${SUBFOLDER}/workers_${TIMESTAMP}"
        mkdir -p "${SWEEP_DIR}"

        SWEEP_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${CLONE3} \
            --array=0-20 --job-name="asw_${MODEL:0:8}_${CLONE:0:6}" \
            "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
        COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} \
            --job-name="asc_${MODEL:0:8}_${CLONE:0:6}" \
            "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")

        echo "  ${MODEL} x ${CLONE}:"
        echo "    sweep=${SWEEP_JOB} (21 tasks), collect=${COLLECT_JOB}"

        SWEEP_COUNT=$((SWEEP_COUNT + 1))
    done
done

echo ""
echo "=========================================="
echo "  Experiment 1 submitted!"
echo "  Clone creation: 3 jobs (serial chain, last=${CLONE3})"
echo "  Alpha sweeps:   ${SWEEP_COUNT} batches (each: 21 workers + 1 collect)"
echo "  Total SLURM jobs: $((3 + SWEEP_COUNT * 22))"
echo "  Monitor: squeue -u \$USER"
echo "=========================================="
