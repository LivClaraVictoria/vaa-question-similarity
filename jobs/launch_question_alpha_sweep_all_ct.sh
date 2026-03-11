#!/bin/bash
# Launcher: per-question alpha sweep across ALL 5 clone types.
# Submits 5 SLURM job arrays (one per clone type, 75 workers each)
# into a shared sweep directory, then a single collect job.
#
# Run with: bash jobs/launch_question_alpha_sweep_all_ct.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"
SWEEP_DIR="${PROJECT_DIR}/experiment_results/exp1/question_alpha_sweep/workers_allct_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a04.py"
export SWEEP_DIR
export N_CLONES=4

CLONE_TYPES=("easy_paraphrase" "hard_paraphrase" "negation_easy" "negation_hard" "perfect_mix")

echo "=== Per-Question Alpha Sweep (All Clone Types) ==="
echo "  Config: ${PIPELINE_CONFIG}"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Clone types: ${CLONE_TYPES[*]}"
echo "  N clones: ${N_CLONES}"

# Step 1: Determine question count
cd "${PROJECT_DIR}"
[[ -f /itet-stor/liweiss/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/liweiss/net_scratch/conda/bin/conda shell.bash hook)"
conda activate bachelor-thesis

N_QUESTIONS=$(python -c "
import pandas as pd
df = pd.read_parquet('${PROJECT_DIR}/data/cleaned/df_questions.parquet')
print(len(df[df['ID_question'] < 9_000_000]))
")
MAX_IDX=$((N_QUESTIONS - 1))

echo ""
echo "  Questions: ${N_QUESTIONS} (array 0-${MAX_IDX})"

# Step 2: Submit one array per clone type
echo ""
echo "--- Submitting worker arrays ---"
ALL_JOB_IDS=""

for CT in "${CLONE_TYPES[@]}"; do
    export CLONE_TYPE="${CT}"
    JOB_ID=$(sbatch --parsable --export=ALL --array=0-${MAX_IDX} \
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
COLLECT_JOB=$(sbatch --parsable --export=ALL \
    --dependency=afterok:${ALL_JOB_IDS} \
    "${SCRIPT_DIR}/job_question_alpha_sweep_collect.sh")
echo "  Collect submitted: job ${COLLECT_JOB} (depends on ${ALL_JOB_IDS})"

echo ""
echo "  Total workers: $((N_QUESTIONS * ${#CLONE_TYPES[@]})) (${N_QUESTIONS} questions × ${#CLONE_TYPES[@]} clone types)"
echo "  Monitor: squeue -u \$USER"
