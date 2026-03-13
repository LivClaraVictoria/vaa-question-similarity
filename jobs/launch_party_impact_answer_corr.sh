#!/bin/bash
# Launcher: party impact Phase 2 with ANSWER-CORRELATION metric
# Runs Phase 2 for each major party (top-5 questions per party)
# at alpha=0.3 and alpha=0.4.
# Reuses existing Phase 1 CSV (alpha/model-independent).
# Run with: bash jobs/launch_party_impact_answer_corr.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export TOP_K=5
export PHASE1_CSV="experiment_results/party_impact/phase1/pipeline_e5_ZH/party_impact_pipeline_e5_ZH_0302_0057.csv"

PARTIES=("SP" "Green" "GLP" "Centre" "FDP" "SVP")

echo "=== Party Impact Phase 2 (ANSWER-CORRELATION) ==="
echo "  Phase 1 CSV: ${PHASE1_CSV}"
echo "  Top-k: ${TOP_K}"
echo "  Parties: ${PARTIES[*]}"
echo ""

# --- Alpha = 0.3 ---
export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_answer_corr_ZH_a03.py"

echo "--- Alpha=0.3: Pre-generating paraphrases ---"
PRE_JOB_A03=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/job_party_impact_pre_paraphrases.sh")
echo "  Submitted pre-paraphrase job: ${PRE_JOB_A03}"

echo ""
echo "--- Alpha=0.3: Submitting Phase 2 jobs (parallel, depend on ${PRE_JOB_A03}) ---"
PHASE2_JOBS_A03=()
for PARTY in "${PARTIES[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --export=ALL,TARGET_PARTY="${PARTY}" \
        --dependency=afterok:${PRE_JOB_A03} \
        "${SCRIPT_DIR}/job_party_impact_phase2.sh")
    PHASE2_JOBS_A03+=("${JOB_ID}")
    echo "  ${PARTY}: job ${JOB_ID}"
done

DEP_STR_A03=$(IFS=:; echo "${PHASE2_JOBS_A03[*]}")

echo ""
echo "--- Alpha=0.3: Compile job (depends on all Phase 2 jobs) ---"
COMPILE_JOB_A03=$(sbatch --parsable \
    --export=ALL \
    --dependency=afterok:${DEP_STR_A03} \
    "${SCRIPT_DIR}/job_party_impact_compile.sh")
echo "  Compile: job ${COMPILE_JOB_A03}"

# --- Alpha = 0.4 ---
export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_answer_corr_ZH_a04.py"

echo ""
echo "--- Alpha=0.4: Submitting Phase 2 jobs (parallel, depend on ${PRE_JOB_A03}) ---"
PHASE2_JOBS_A04=()
for PARTY in "${PARTIES[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --export=ALL,TARGET_PARTY="${PARTY}" \
        --dependency=afterok:${PRE_JOB_A03} \
        "${SCRIPT_DIR}/job_party_impact_phase2.sh")
    PHASE2_JOBS_A04+=("${JOB_ID}")
    echo "  ${PARTY}: job ${JOB_ID}"
done

DEP_STR_A04=$(IFS=:; echo "${PHASE2_JOBS_A04[*]}")

echo ""
echo "--- Alpha=0.4: Compile job (depends on all Phase 2 jobs) ---"
COMPILE_JOB_A04=$(sbatch --parsable \
    --export=ALL \
    --dependency=afterok:${DEP_STR_A04} \
    "${SCRIPT_DIR}/job_party_impact_compile.sh")
echo "  Compile: job ${COMPILE_JOB_A04}"

echo ""
echo "=== All jobs submitted ==="
echo "  Pre-paraphrase:     ${PRE_JOB_A03}"
echo "  Alpha=0.3 Phase 2:  ${PHASE2_JOBS_A03[*]}"
echo "  Alpha=0.3 Compile:  ${COMPILE_JOB_A03}"
echo "  Alpha=0.4 Phase 2:  ${PHASE2_JOBS_A04[*]}"
echo "  Alpha=0.4 Compile:  ${COMPILE_JOB_A04}"
echo ""
echo "Monitor with: squeue -u \$USER"
