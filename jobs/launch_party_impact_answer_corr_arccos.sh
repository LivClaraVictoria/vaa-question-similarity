#!/bin/bash
# Launcher: party impact Phase 2 with ANSWER-CORRELATION-ARCCOS metric
# Runs Phase 2 for each major party (top-5 questions per party)
# at alpha=1.5 (highest alpha with near-perfect correction).
# Reuses existing Phase 1 CSV (alpha/model-independent).
# Run with: bash jobs/launch_party_impact_answer_corr_arccos.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export TOP_K=5
export PHASE1_CSV="experiment_results/party_impact/high_impact/phase1/pipeline_e5_ZH/party_impact_pipeline_e5_ZH_0302_0057.csv"

PARTIES=("SP" "Green" "GLP" "Centre" "FDP" "SVP")

echo "=== Party Impact Phase 2 (ANSWER-CORRELATION-ARCCOS) ==="
echo "  Phase 1 CSV: ${PHASE1_CSV}"
echo "  Top-k: ${TOP_K}"
echo "  Parties: ${PARTIES[*]}"
echo ""

# --- Alpha = 1.5 ---
export PIPELINE_CONFIG="configs/base_pipeline/pipeline_answer_corr_arccos_ZH_a15.py"

echo "--- Alpha=1.5: Pre-generating paraphrases ---"
PRE_JOB=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/job_party_impact_pre_paraphrases.sh")
echo "  Submitted pre-paraphrase job: ${PRE_JOB}"

echo ""
echo "--- Alpha=1.5: Submitting Phase 2 jobs (parallel, depend on ${PRE_JOB}) ---"
PHASE2_JOBS=()
for PARTY in "${PARTIES[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --export=ALL,TARGET_PARTY="${PARTY}" \
        --dependency=afterok:${PRE_JOB} \
        "${SCRIPT_DIR}/job_party_impact_phase2.sh")
    PHASE2_JOBS+=("${JOB_ID}")
    echo "  ${PARTY}: job ${JOB_ID}"
done

DEP_STR=$(IFS=:; echo "${PHASE2_JOBS[*]}")

echo ""
echo "--- Alpha=1.5: Compile job (depends on all Phase 2 jobs) ---"
COMPILE_JOB=$(sbatch --parsable \
    --export=ALL \
    --dependency=afterok:${DEP_STR} \
    "${SCRIPT_DIR}/job_party_impact_compile.sh")
echo "  Compile: job ${COMPILE_JOB}"

echo ""
echo "=== All jobs submitted ==="
echo "  Pre-paraphrase:    ${PRE_JOB}"
echo "  Alpha=1.5 Phase 2: ${PHASE2_JOBS[*]}"
echo "  Alpha=1.5 Compile: ${COMPILE_JOB}"
echo ""
echo "Monitor with: squeue -u \$USER"
