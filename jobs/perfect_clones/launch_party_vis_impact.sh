#!/bin/bash
# Launcher: party visibility impact analysis.
# Array of workers (one per question) + dependent collect job.
# Run with: bash jobs/launch_party_vis_impact.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"
SWEEP_DIR="${PROJECT_DIR}/experiment_results/question_impact/party_visibility_impact/workers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export PIPELINE_CONFIG="configs/base_pipeline/pipeline_e5_ZH.py"
export SWEEP_DIR

# Determine question count
N_QUESTIONS=$(python -c "
import pandas as pd
df = pd.read_parquet('${PROJECT_DIR}/data/cleaned/df_questions.parquet')
print(len(df[df['ID_question'] < 9_000_000]))
")
MAX_IDX=$((N_QUESTIONS - 1))

echo "=== Party Visibility Impact Analysis ==="
echo "  Config: ${PIPELINE_CONFIG}"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Questions: ${N_QUESTIONS} (array 0-${MAX_IDX})"

# Workers: one per question
SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-${MAX_IDX} "${SCRIPT_DIR}/job_party_vis_impact_worker.sh")
echo "  Workers submitted: job array ${SWEEP_JOB} (${N_QUESTIONS} tasks)"

# Collect: aggregate + enrich + all plots
COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} "${SCRIPT_DIR}/job_party_vis_impact_collect.sh")
echo "  Collect submitted:  job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo ""
echo "  Monitor: squeue -u \$USER"
