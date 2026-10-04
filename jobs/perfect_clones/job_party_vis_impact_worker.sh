#!/bin/bash
# Worker (SLURM array): compute party visibility shift for one question cloned 4× identical.
# Env vars: PIPELINE_CONFIG, SWEEP_DIR. Array index selects the question.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4

# Party visibility impact worker — processes a single question.
# Launched as a SLURM job array by launch_party_vis_impact.sh.
# Expects: SLURM_ARRAY_TASK_ID, SWEEP_DIR, PIPELINE_CONFIG (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}, SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

python -u -m experiments.perfect_clones.party_visibility_impact \
    --mode worker \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --config "${PIPELINE_CONFIG}" \
    --sweep-dir "${SWEEP_DIR}"

echo "Finished at: $(date)"
exit 0
