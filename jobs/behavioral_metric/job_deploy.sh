#!/bin/bash
# Worker: behavioral-metric out-of-sample deployment simulation (all seeds, sequential).
# Estimates the answer-based distance from a 5% voter pilot, measures held-out recommendation
# distortion + cross-seed stability. CPU only (no embedding model).
# Env vars: DEPLOY_CONFIG (config path; default behavioral L1 ZH),
#           DEPLOY_SEEDS  (optional, e.g. "0,1,2,3,4"),
#           DEPLOY_ALPHAS (optional comma list; default auto-calibrated from the distances).
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

ARGS="--config ${DEPLOY_CONFIG:-configs/base_pipeline/pipeline_behavioral_l1_ZH.py}"
[[ -n "${DEPLOY_SEEDS}" ]]  && ARGS="${ARGS} --seeds ${DEPLOY_SEEDS}"
[[ -n "${DEPLOY_ALPHAS}" ]] && ARGS="${ARGS} --alphas ${DEPLOY_ALPHAS}"

echo "Args: ${ARGS}"
python -u -m main behavioral-deploy ${ARGS}

echo "Finished at: $(date)"
exit 0
