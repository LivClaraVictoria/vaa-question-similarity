#!/bin/bash
# Worker (SLURM array): run threshold-based CRW for one alpha value (verification experiment).
# Env vars: CONFIG_A, CONFIG_B, SWEEP_DIR. Array index selects the alpha.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4

# Threshold alpha sweep worker — processes a single alpha value.
# Launched as a SLURM job array by launch_threshold_sweep.sh.
# Expects: SLURM_ARRAY_TASK_ID, SWEEP_DIR, CONFIG_A, CONFIG_B (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}, SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

python -u -m experiments.verification.threshold_alpha_sweep \
    --mode worker \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --config_a "${CONFIG_A}" \
    --config_b "${CONFIG_B}" \
    --sweep-dir "${SWEEP_DIR}"

echo "Finished at: $(date)"
exit 0
