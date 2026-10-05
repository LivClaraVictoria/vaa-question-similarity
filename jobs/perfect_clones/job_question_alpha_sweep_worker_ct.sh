#!/bin/bash
# Worker (SLURM array): run alpha sweep for one question × one clone type (75 q × 5 types).
# Env vars: PIPELINE_CONFIG, SWEEP_DIR, CLONE_TYPE; optional ALPHAS, Q_IDS (comma-separated).
# Array index selects the question (within Q_IDS if set, else among all questions).
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8

# Question alpha sweep worker — processes a single question for a single clone type.
# Expects: SLURM_ARRAY_TASK_ID, SWEEP_DIR, PIPELINE_CONFIG, CLONE_TYPE (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}, SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "CLONE_TYPE: ${CLONE_TYPE}"

python -u -m experiments.perfect_clones.recommendation_distortion \
    --mode worker \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --config "${PIPELINE_CONFIG}" \
    --sweep-dir "${SWEEP_DIR}" \
    --clone-type "${CLONE_TYPE}" \
    --n-clones "${N_CLONES:-5}" \
    ${ALPHAS:+--alphas "${ALPHAS}"} \
    ${Q_IDS:+--q-ids "${Q_IDS}"}

echo "Finished at: $(date)"
exit 0
