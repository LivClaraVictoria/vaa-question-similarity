#!/bin/bash
# Launcher: party impact analysis for base E5 ZH.
# Phase 1: one job per question (array) + collect
# Phase 2: CRW for top-5 (depends on collect)
# Run with: bash jobs/launch_party_impact.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"
SWEEP_DIR="${PROJECT_DIR}/experiment_results/party_impact/workers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_ZH.py"
export SWEEP_DIR
export TOP_K=5
export TARGET_PARTY="Centre"

# Determine question count
N_QUESTIONS=$(python -c "
import pandas as pd
df = pd.read_parquet('${PROJECT_DIR}/data/cleaned/df_questions.parquet')
print(len(df[df['ID_question'] < 9_000_000]))
")
MAX_IDX=$((N_QUESTIONS - 1))

echo "=== Party Impact Analysis ==="
echo "  Config: ${PIPELINE_CONFIG}"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Questions: ${N_QUESTIONS} (array 0-${MAX_IDX})"
echo "  Phase 2 top-k: ${TOP_K}"
echo "  Target party: ${TARGET_PARTY}"

# Phase 1: worker array
SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-${MAX_IDX} "${SCRIPT_DIR}/job_party_impact_worker.sh")
echo "  Workers submitted: job array ${SWEEP_JOB} (${N_QUESTIONS} tasks)"

# Phase 1: collect (depends on all workers)
COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} "${SCRIPT_DIR}/job_party_impact_collect.sh")
echo "  Collect submitted:  job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

# Phase 2: CRW correction (depends on collect)
PHASE2_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${COLLECT_JOB} "${SCRIPT_DIR}/job_party_impact_phase2.sh")
echo "  Phase 2 submitted:  job ${PHASE2_JOB} (depends on ${COLLECT_JOB})"

echo ""
echo "  Monitor: squeue -u \$USER"
