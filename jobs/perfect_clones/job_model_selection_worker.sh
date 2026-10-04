#!/bin/bash
# Worker (SLURM array): run CRW + recommendations for one alpha value in the model selection sweep.
# Env vars: CONFIG_A, CONFIG_B, SWEEP_DIR. Array index selects the alpha.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4

# Model selection worker — processes a single alpha value.
# Launched as a SLURM job array by launch_model_selection.sh.
# Expects: SLURM_ARRAY_TASK_ID, SWEEP_DIR, CONFIG_A, CONFIG_B (via --export).
# Optional: OUTPUT_DIR — if set, passes --output-dir to model_selection.

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}, SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

OUTPUT_DIR_FLAG=""
[[ -n "${OUTPUT_DIR}" ]] && OUTPUT_DIR_FLAG="--output-dir ${OUTPUT_DIR}"

python -u -m experiments.perfect_clones.model_selection \
    --mode worker \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --config_a "${CONFIG_A}" \
    --config_b "${CONFIG_B}" \
    --sweep-dir "${SWEEP_DIR}" \
    ${OUTPUT_DIR_FLAG}

echo "Finished at: $(date)"
exit 0
