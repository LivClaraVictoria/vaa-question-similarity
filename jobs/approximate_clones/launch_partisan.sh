#!/bin/bash
# Launcher: mini vs maxi party impact with corr-weighted question selection.
# Reuses existing Phase 1 results (no re-computation needed).
# Phase 2: one job per party (corr_weighted selection) + compile.
# Run with: bash jobs/launch_mini_maxi_corr_weighted.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PIPELINE_CONFIG="configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py"
export TOP_K=5
export SELECTION_MODE="corr_weighted"

PARTIES=("SP" "Green" "GLP" "Centre" "FDP" "SVP")

echo "=== Mini vs Maxi Party Impact — Corr-Weighted Selection ==="
echo "  Config: ${PIPELINE_CONFIG}"
echo "  Selection mode: ${SELECTION_MODE}"
echo "  Phase 2 top-k: ${TOP_K}"
echo "  Parties: ${PARTIES[*]}"
echo ""
echo "  (Reuses existing Phase 1 results — no Phase 1 jobs submitted)"

# Phase 2: one job per party (all independent, run in parallel)
echo ""
echo "--- Phase 2: submitting one job per party ---"
PHASE2_JOBS=()
for PARTY in "${PARTIES[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --export=ALL,TARGET_PARTY="${PARTY}" \
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
