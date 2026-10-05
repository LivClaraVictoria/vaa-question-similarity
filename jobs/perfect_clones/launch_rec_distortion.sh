#!/bin/bash
# Launcher: per-question alpha sweep across ALL 5 clone types.
# Submits 5 SLURM job arrays (one per clone type, 75 workers each)
# into a shared sweep directory, then a single collect job.
#
# Optional env overrides (defaults = the full 75-question sweep on the default alpha grid):
#   Q_IDS="32214,32228"   restrict to these question IDs (array size = number of IDs)
#   ALPHAS="0.01,0.1,0.2" alpha grid (comma-separated; default = DEFAULT_ALPHAS in experiments/_common.py)
#   PIPELINE_CONFIG, N_CLONES, SWEEP_TAG (suffix for the sweep dir name)
#
# Run with: bash jobs/perfect_clones/launch_rec_distortion.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../_lib/common.sh"
cd "${PROJECT_DIR}"
SWEEP_DIR="$(results_root)/exp1/question_alpha_sweep/workers_allct${SWEEP_TAG:+_${SWEEP_TAG}}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/base_pipeline/pipeline_e5_instruct_ZH_a04.py}"
export SWEEP_DIR
export N_CLONES="${N_CLONES:-4}"
export ALPHAS Q_IDS   # empty = defaults (all alphas / all questions)

CLONE_TYPES=("easy_paraphrase" "hard_paraphrase" "negation_easy" "negation_hard" "perfect_mix")

echo "=== Per-Question Alpha Sweep (All Clone Types) ==="
echo "  Config: ${PIPELINE_CONFIG}"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Clone types: ${CLONE_TYPES[*]}"
echo "  N clones: ${N_CLONES}"
echo "  Alphas: ${ALPHAS:-<DEFAULT_ALPHAS>}"
echo "  Questions: ${Q_IDS:-<all>}"

# Step 1: Determine question count
activate_env

if [ -n "${Q_IDS}" ]; then
    N_QUESTIONS=$(echo "${Q_IDS}" | tr ',' '\n' | grep -c .)
else
    N_QUESTIONS=$(python -c "
import pandas as pd
df = pd.read_parquet('${PROJECT_DIR}/data/cleaned/df_questions.parquet')
print(len(df[df['ID_question'] < 9_000_000]))
")
fi
MAX_IDX=$((N_QUESTIONS - 1))

echo ""
echo "  Questions: ${N_QUESTIONS} (array 0-${MAX_IDX})"

# Step 2: Submit one array per clone type
echo ""
echo "--- Submitting worker arrays ---"
ALL_JOB_IDS=""

for CT in "${CLONE_TYPES[@]}"; do
    export CLONE_TYPE="${CT}"
    JOB_ID=$(submit --array=0-${MAX_IDX} \
        --job-name="qa_sweep_${CT}" \
        "${SCRIPT_DIR}/job_question_alpha_sweep_worker_ct.sh")
    echo "  ${CT}: job array ${JOB_ID} (${N_QUESTIONS} tasks)"

    if [ -z "${ALL_JOB_IDS}" ]; then
        ALL_JOB_IDS="${JOB_ID}"
    else
        ALL_JOB_IDS="${ALL_JOB_IDS}:${JOB_ID}"
    fi
done

# Step 3: Submit collect job (depends on all worker arrays)
echo ""
echo "--- Submitting collect job ---"
COLLECT_JOB=$(submit \
    --dependency=afterok:${ALL_JOB_IDS} \
    "${SCRIPT_DIR}/job_question_alpha_sweep_collect.sh")
echo "  Collect submitted: job ${COLLECT_JOB} (depends on ${ALL_JOB_IDS})"

echo ""
echo "  Total workers: $((N_QUESTIONS * ${#CLONE_TYPES[@]})) (${N_QUESTIONS} questions × ${#CLONE_TYPES[@]} clone types)"
echo "  Monitor: squeue -u \$USER"
