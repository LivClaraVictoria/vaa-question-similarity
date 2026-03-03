#!/bin/bash
# Launch party sweep: run Phase 2 for ALL 6 major parties in parallel.
# Requires a Phase 1 CSV (already computed, model-independent).
#
# Flow:
#   1. Pre-generate paraphrases for ALL parties' top-K questions (serial, safe)
#   2. Submit 6 Phase 2 jobs in parallel (one per party, each depends on step 1)
#
# Usage: bash jobs/launch_party_sweep.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ---
export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py"
export PHASE1_CSV="experiment_results/party_impact_results/party_impact_pipeline_e5_ZH_0302_0057.csv"
export TOP_K=5

PARTIES=("SP" "Green" "GLP" "Centre" "FDP" "SVP")

echo "=== Party Sweep: Phase 2 for all ${#PARTIES[@]} parties ==="
echo "  Config:    ${PIPELINE_CONFIG}"
echo "  Phase1 CSV: ${PHASE1_CSV}"
echo "  Top-K:     ${TOP_K}"
echo "  Parties:   ${PARTIES[*]}"
echo ""

# Step 1: Pre-generate paraphrases (serial)
echo "--- Step 1: Pre-generating paraphrases ---"
PRE_JOB=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/job_party_impact_pre_paraphrases.sh")
echo "  Submitted pre-paraphrase job: ${PRE_JOB}"

# Step 2: Submit 6 Phase 2 jobs in parallel (each depends on pre-paraphrase)
echo ""
echo "--- Step 2: Submitting Phase 2 jobs (parallel, depend on ${PRE_JOB}) ---"
PHASE2_JOBS=()
for PARTY in "${PARTIES[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --export=ALL,TARGET_PARTY="${PARTY}" \
        --dependency=afterok:${PRE_JOB} \
        "${SCRIPT_DIR}/job_party_impact_phase2.sh")
    PHASE2_JOBS+=("${JOB_ID}")
    echo "  ${PARTY}: job ${JOB_ID}"
done

echo ""
echo "=== All jobs submitted ==="
echo "  Pre-paraphrase: ${PRE_JOB}"
echo "  Phase 2 jobs:   ${PHASE2_JOBS[*]}"
echo ""
echo "Monitor with: squeue -u \$USER"
