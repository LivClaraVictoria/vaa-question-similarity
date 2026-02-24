#!/bin/bash
# Launcher: question impact sweep for base E5 ZH.
# Submits one job per question (array) + a dependent collect job.
# Run with: bash jobs/launch_question_impact.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"
SWEEP_DIR="${PROJECT_DIR}/experiment_results/question_impact_results/workers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_ZH.py"
export SWEEP_DIR

# Determine question count by reading the questions parquet directly
N_QUESTIONS=$(python -c "
import pandas as pd
df = pd.read_parquet('${PROJECT_DIR}/data/cleaned/df_questions.parquet')
print(len(df[df['ID_question'] < 9_000_000]))
")
MAX_IDX=$((N_QUESTIONS - 1))

echo "=== Question Impact Sweep ==="
echo "  Config: ${PIPELINE_CONFIG}"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Questions: ${N_QUESTIONS} (array 0-${MAX_IDX})"

SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-${MAX_IDX} "${SCRIPT_DIR}/job_question_impact_worker.sh")
echo "  Workers submitted: job array ${SWEEP_JOB} (${N_QUESTIONS} tasks)"

COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} "${SCRIPT_DIR}/job_question_impact_collect.sh")
echo "  Collect submitted:  job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo "  Monitor: squeue -u \$USER"
