#!/bin/bash
# Worker (SLURM array): run the λ × seed grid for a single source question.
# Env vars: PIPELINE_CONFIG, SWEEP_DIR, ALPHA, N_SEEDS, LAMBDAS.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4

# Noise slider worker — processes a single source question.
# Runs λ × seed grid: 8 λ × 20 seeds = 160 rec pairs per task.
# Launched as a SLURM array by launch_robustness.sh.
# Expects: SLURM_ARRAY_TASK_ID, SWEEP_DIR, PIPELINE_CONFIG, ALPHA, N_SEEDS, LAMBDAS
# (via --export=ALL).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}, SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

EXTRA_ARGS=()
if [[ -n "${LAMBDAS}" ]]; then
    EXTRA_ARGS+=("--lambdas" "${LAMBDAS}")
fi
if [[ -n "${SUBSET_N}" ]]; then
    EXTRA_ARGS+=("--subset-n" "${SUBSET_N}")
fi

python -u -m main noise-slider \
    --mode worker \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --config "${PIPELINE_CONFIG}" \
    --sweep-dir "${SWEEP_DIR}" \
    --alpha "${ALPHA}" \
    --n-seeds "${N_SEEDS}" \
    "${EXTRA_ARGS[@]}"

echo "Finished at: $(date)"
exit 0
