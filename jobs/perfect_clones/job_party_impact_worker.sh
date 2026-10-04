#!/bin/bash
# Worker (SLURM array): Phase 1 — compute party visibility delta when one question is cloned 4×.
# Env vars: PIPELINE_CONFIG, SWEEP_DIR. Array index selects the question.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4

# Party impact worker — processes a single question (Phase 1).
# Launched as a SLURM job array by launch_partisan.sh.
# Expects: SLURM_ARRAY_TASK_ID, SWEEP_DIR, PIPELINE_CONFIG (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}, SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

python -u -m experiments.perfect_clones.partisan_distortion \
    --mode worker \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --config "${PIPELINE_CONFIG}" \
    --sweep-dir "${SWEEP_DIR}"

echo "Finished at: $(date)"
exit 0
