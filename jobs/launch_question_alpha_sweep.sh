#!/bin/bash
# Launcher: per-question alpha sweep with easy paraphrases.
# Step 1: Generate paraphrases (interactive, requires OPENAI_API_KEY)
# Step 2: Submit SLURM worker array (one per question)
# Step 3: Submit dependent collect job
#
# Run with: bash jobs/launch_question_alpha_sweep.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"
SWEEP_DIR="${PROJECT_DIR}/experiment_results/exp1/question_alpha_sweep/workers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py"
export SWEEP_DIR

echo "=== Per-Question Alpha Sweep ==="
echo "  Config: ${PIPELINE_CONFIG}"
echo "  Sweep dir: ${SWEEP_DIR}"

# Step 1: Prepare paraphrases (interactive, before SLURM)
echo ""
echo "--- Step 1: Generating paraphrases (if needed) ---"
cd "${PROJECT_DIR}"

# Activate conda for interactive paraphrase generation
[[ -f /itet-stor/liweiss/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/liweiss/net_scratch/conda/bin/conda shell.bash hook)"
conda activate bachelor-thesis

python -u -m experiments.synthetic_clones.rec_change.question_alpha_sweep \
    --config "${PIPELINE_CONFIG}" \
    --mode prepare

# Step 2: Determine question count
N_QUESTIONS=$(python -c "
import pandas as pd
df = pd.read_parquet('${PROJECT_DIR}/data/cleaned/df_questions.parquet')
print(len(df[df['ID_question'] < 9_000_000]))
")
MAX_IDX=$((N_QUESTIONS - 1))

echo ""
echo "--- Step 2: Submitting workers ---"
echo "  Questions: ${N_QUESTIONS} (array 0-${MAX_IDX})"

SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-${MAX_IDX} "${SCRIPT_DIR}/job_question_alpha_sweep_worker.sh")
echo "  Workers submitted: job array ${SWEEP_JOB} (${N_QUESTIONS} tasks)"

# Step 3: Collect
echo ""
echo "--- Step 3: Submitting collect job ---"
COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} "${SCRIPT_DIR}/job_question_alpha_sweep_collect.sh")
echo "  Collect submitted:  job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo ""
echo "  Monitor: squeue -u \$USER"
