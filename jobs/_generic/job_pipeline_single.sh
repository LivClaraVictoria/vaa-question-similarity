#!/bin/bash
# Worker: run a single full pipeline (distances → CRW → recommendations) for one config.
# Env vars: PIPELINE_CONFIG (path to pipeline config), PIPELINE_OVERRIDES (optional key=value args).
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8

# Generic single-pipeline runner. Expects PIPELINE_CONFIG env var.

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Config: ${PIPELINE_CONFIG}"

python -u -m main pipeline --config "${PIPELINE_CONFIG}" ${PIPELINE_OVERRIDES:-}

echo "Finished at: $(date)"
exit 0
