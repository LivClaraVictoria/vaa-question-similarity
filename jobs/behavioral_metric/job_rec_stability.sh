#!/bin/bash
# Worker: per-voter inter-pilot recommendation stability (a few reference alphas, 5 pilots).
# Recomputes CRW recommendations (the expensive step) so it runs on a compute node, not the login.
# Env vars: RECSTAB_CONFIG (config path; default behavioral L1 ZH),
#           RECSTAB_ALPHAS (optional comma list; default 0.13,0.25,0.37),
#           RECSTAB_SEEDS  (optional, e.g. "0,1,2,3,4").
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

ARGS="--config ${RECSTAB_CONFIG:-configs/base_pipeline/pipeline_behavioral_l1_ZH.py}"
[[ -n "${RECSTAB_ALPHAS}" ]] && ARGS="${ARGS} --alphas ${RECSTAB_ALPHAS}"
[[ -n "${RECSTAB_SEEDS}" ]]  && ARGS="${ARGS} --seeds ${RECSTAB_SEEDS}"

echo "Args: ${ARGS}"
python -u -m main behavioral-rec-stability ${ARGS}

echo "Finished at: $(date)"
exit 0
