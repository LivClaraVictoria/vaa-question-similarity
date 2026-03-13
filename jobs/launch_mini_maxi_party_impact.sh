#!/bin/bash
# Launcher: mini vs maxi party impact analysis.
# Phase 1: one job per full-only question (array 0-44) + collect
# Phase 2: CRW correction for each major party (depends on collect)
# Run with: bash jobs/launch_mini_maxi_party_impact.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"
SWEEP_DIR="${PROJECT_DIR}/experiment_results/party_impact/mini_maxi/phase1/workers/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py"
export SWEEP_DIR
export TOP_K=5

PARTIES=("SP" "Green" "GLP" "Centre" "FDP" "SVP")

# Determine full-only question count (total - mini)
N_FULL_ONLY=$(python -c "
import pandas as pd
df = pd.read_parquet('${PROJECT_DIR}/data/cleaned/df_questions.parquet')
real = df[df['ID_question'] < 9_000_000]
print(len(real[real['rapide'] != 1]))
")
MAX_IDX=$((N_FULL_ONLY - 1))

echo "=== Mini vs Maxi Party Impact Analysis ==="
echo "  Config: ${PIPELINE_CONFIG}"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Full-only questions: ${N_FULL_ONLY} (array 0-${MAX_IDX})"
echo "  Phase 2 top-k: ${TOP_K}"
echo "  Parties: ${PARTIES[*]}"

# Phase 1: worker array
SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-${MAX_IDX} "${SCRIPT_DIR}/job_mini_maxi_worker.sh")
echo "  Workers submitted: job array ${SWEEP_JOB} (${N_FULL_ONLY} tasks)"

# Phase 1: collect (depends on all workers)
COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} "${SCRIPT_DIR}/job_mini_maxi_collect.sh")
echo "  Collect submitted:  job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

# Phase 2: one job per party (all depend on collect, run in parallel)
echo ""
echo "--- Phase 2: submitting one job per party (depend on collect ${COLLECT_JOB}) ---"
PHASE2_JOBS=()
for PARTY in "${PARTIES[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --export=ALL,TARGET_PARTY="${PARTY}" \
        --dependency=afterok:${COLLECT_JOB} \
        "${SCRIPT_DIR}/job_mini_maxi_phase2.sh")
    PHASE2_JOBS+=("${JOB_ID}")
    echo "  ${PARTY}: job ${JOB_ID}"
done

DEP_STR=$(IFS=:; echo "${PHASE2_JOBS[*]}")

echo ""
echo "--- Compile: aggregate all Phase 2 results (depends on all Phase 2 jobs) ---"
COMPILE_JOB=$(sbatch --parsable \
    --export=ALL \
    --dependency=afterok:${DEP_STR} \
    "${SCRIPT_DIR}/job_mini_maxi_compile.sh")
echo "  Compile: job ${COMPILE_JOB}"

echo ""
echo "  Phase 2 jobs: ${PHASE2_JOBS[*]}"
echo "  Compile job:  ${COMPILE_JOB}"
echo "  Monitor: squeue -u \$USER"
