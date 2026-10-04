#!/bin/bash
# Worker: generate a single cloned dataset from a clone config and write it to data/cloned/.
# Env vars: CLONE_CONFIG (path to clone config .py file).
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2

# Generic single clone-creation runner. Expects CLONE_CONFIG env var.

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Config: ${CLONE_CONFIG}"

python -u -m main generate-clones --config "${CLONE_CONFIG}"

echo "Finished at: $(date)"
exit 0
