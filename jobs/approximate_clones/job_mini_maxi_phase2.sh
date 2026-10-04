#!/bin/bash
# Phase 2: cumulative CRW correction for mini vs maxi party visibility analysis.
# Env vars: PIPELINE_CONFIG, TARGET_PARTY, TOP_K, PHASE1_CSV, SELECTION_MODE.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8

# Mini-maxi Phase 2 — CRW correction with multiple distance metrics.
# Runs 5 (metric, alpha) combinations sequentially.
# Launched with --dependency by launch_partisan_full.sh.
# Expects: PIPELINE_CONFIG, TOP_K (via --export).
# Optional: PHASE1_CSV, TARGET_PARTY.

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

PHASE2_ARGS="--mode phase2 --config ${PIPELINE_CONFIG}"
[[ -n "${PHASE1_CSV}" ]] && PHASE2_ARGS="${PHASE2_ARGS} --phase1-csv ${PHASE1_CSV}"
[[ -n "${TOP_K}" ]] && PHASE2_ARGS="${PHASE2_ARGS} --top-k ${TOP_K}"
[[ -n "${TARGET_PARTY}" ]] && PHASE2_ARGS="${PHASE2_ARGS} --target-party ${TARGET_PARTY}"
[[ -n "${SELECTION_MODE}" ]] && PHASE2_ARGS="${PHASE2_ARGS} --selection-mode ${SELECTION_MODE}"

python -u -m experiments.approximate_clones.partisan_distortion ${PHASE2_ARGS}

echo "Finished at: $(date)"
exit 0
