#!/bin/bash
# Worker (SLURM array): run all 5 clone types for one question with CRW (E5-INSTRUCT, alpha=0.3).
# Env vars: PIPELINE_CONFIG, SWEEP_DIR. Array index selects the question.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4

# Question clone-type sweep worker — processes a single question across all clone types.
# Launched as a SLURM job array by launch_question_clone_sweep.sh.
# Expects: SLURM_ARRAY_TASK_ID, SWEEP_DIR, PIPELINE_CONFIG, CLONE_TYPES, N_CLONES (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}, SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "CLONE_TYPES: ${CLONE_TYPES}"
echo "N_CLONES: ${N_CLONES}"

python -u -m experiments.explanatory.question_impact \
    --mode worker \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --config "${PIPELINE_CONFIG}" \
    --sweep-dir "${SWEEP_DIR}" \
    --clone-types "${CLONE_TYPES}" \
    --n-clones "${N_CLONES}"

echo "Finished at: $(date)"
exit 0
