#!/bin/bash
# Prepare: generate paraphrases for all 75 questions × 4 clone types (once).
# Must run serially — paraphrase cache has a read/rewrite race condition.
# Env vars: PIPELINE_CONFIG.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m main noise-slider \
    --mode prepare \
    --config "${PIPELINE_CONFIG}"

echo "Finished at: $(date)"
exit 0
