#!/bin/bash
# Launcher: per-question clone type sweep with CRW (E5-INSTRUCT, alpha=0.3).
# Submits one job per question (75 workers) + a dependent collect job.
# Each worker processes all 5 clone types for one question.
# Run with: bash jobs/launch_question_clone_sweep.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"
SWEEP_DIR="${PROJECT_DIR}/experiment_results/question_impact/clone_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py"
export CLONE_TYPES="easy_paraphrase,hard_paraphrase,negation_easy,negation_hard,perfect_mix"
export N_CLONES=4
export SWEEP_DIR

# Determine question count
N_QUESTIONS=$(python -c "
import pandas as pd
df = pd.read_parquet('${PROJECT_DIR}/data/cleaned/df_questions.parquet')
print(len(df[df['ID_question'] < 9_000_000]))
")
MAX_IDX=$((N_QUESTIONS - 1))

echo "=== Question Clone Type Sweep (CRW) ==="
echo "  Config: ${PIPELINE_CONFIG}"
echo "  Clone types: ${CLONE_TYPES}"
echo "  Clones per question: ${N_CLONES}"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Questions: ${N_QUESTIONS} (array 0-${MAX_IDX})"

SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-${MAX_IDX} "${SCRIPT_DIR}/job_question_clone_sweep_worker.sh")
echo "  Workers submitted: job array ${SWEEP_JOB} (${N_QUESTIONS} tasks)"

COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} "${SCRIPT_DIR}/job_question_clone_sweep_collect.sh")
echo "  Collect submitted:  job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Expected: ~30 min per worker (75 parallel)"
