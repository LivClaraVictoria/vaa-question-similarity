#!/bin/bash
# Phase 2: cumulative CRW correction — clone top-K questions simultaneously, compare visibility.
# Env vars: PIPELINE_CONFIG, TARGET_PARTY, TOP_K, PHASE1_CSV.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8

# Party impact Phase 2 — CRW correction analysis for top-K questions.
# Requires embedding model + full CRW pipeline for each question.
# Launched with --dependency by launch_partisan.sh.
# Expects: PIPELINE_CONFIG, TOP_K (via --export).
# Optional: PHASE1_CSV, TARGET_PARTY (auto-detected/omitted if not set).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

PHASE2_ARGS="--mode phase2 --config ${PIPELINE_CONFIG}"
[[ -n "${PHASE1_CSV}" ]] && PHASE2_ARGS="${PHASE2_ARGS} --phase1-csv ${PHASE1_CSV}"
[[ -n "${TOP_K}" ]] && PHASE2_ARGS="${PHASE2_ARGS} --top-k ${TOP_K}"
[[ -n "${TARGET_PARTY}" ]] && PHASE2_ARGS="${PHASE2_ARGS} --target-party ${TARGET_PARTY}"

python -u -m experiments.perfect_clones.partisan_distortion ${PHASE2_ARGS}

echo "Finished at: $(date)"
exit 0
