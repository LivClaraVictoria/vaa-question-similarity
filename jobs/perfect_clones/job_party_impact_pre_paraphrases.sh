#!/bin/bash
# Pre-generate paraphrases for the top-K party impact questions before Phase 2 starts.
# Env vars: PIPELINE_CONFIG, TARGET_PARTY, TOP_K, PHASE1_CSV.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2

# Pre-generate paraphrases for ALL parties' top-K questions.
# Primes the JSON cache so that parallel Phase 2 jobs only read (no race condition).
# Expects: PIPELINE_CONFIG, PHASE1_CSV, TOP_K (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

PRE_ARGS="--mode pre-paraphrases --config ${PIPELINE_CONFIG}"
[[ -n "${PHASE1_CSV}" ]] && PRE_ARGS="${PRE_ARGS} --phase1-csv ${PHASE1_CSV}"
[[ -n "${TOP_K}" ]] && PRE_ARGS="${PRE_ARGS} --top-k ${TOP_K}"

python -u -m experiments.perfect_clones.partisan_distortion ${PRE_ARGS}

echo "Finished at: $(date)"
exit 0
