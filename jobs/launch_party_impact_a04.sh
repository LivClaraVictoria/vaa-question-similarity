#!/bin/bash
# Launcher: party impact Phase 2 with E5-INSTRUCT alpha=0.4
# Runs Phase 2 for each major party (top-5 questions per party).
# Reuses existing Phase 1 CSV (alpha/model-independent).
# Run with: bash jobs/launch_party_impact_a04.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a04.py"
export TOP_K=5
export PHASE1_CSV="experiment_results/party_impact_results/party_impact_pipeline_e5_ZH_0302_0057.csv"

PARTIES=("SP" "Green" "GLP" "Centre" "FDP" "SVP")

echo "=== Party Impact Phase 2 (E5-INSTRUCT, alpha=0.4) ==="
echo "  Config: ${PIPELINE_CONFIG}"
echo "  Phase 1 CSV: ${PHASE1_CSV}"
echo "  Top-k: ${TOP_K}"
echo "  Parties: ${PARTIES[*]}"
echo ""

for PARTY in "${PARTIES[@]}"; do
    export TARGET_PARTY="${PARTY}"
    JOB_ID=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/job_party_impact_phase2.sh")
    echo "  ${PARTY}: job ${JOB_ID}"
done

echo ""
echo "  Monitor: squeue -u \$USER"
