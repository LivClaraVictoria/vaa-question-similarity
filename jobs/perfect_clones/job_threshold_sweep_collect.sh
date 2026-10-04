#!/bin/bash
# Collect: aggregate per-alpha threshold CRW sweep CSVs into combined result with plots.
# Env vars: CONFIG_A, CONFIG_B, SWEEP_DIR.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4

# Threshold alpha sweep collect — aggregates per-alpha worker CSVs and generates plots.
# Launched with --dependency by launch_threshold_sweep.sh.
# Expects: SWEEP_DIR, CONFIG_A, CONFIG_B (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m experiments.verification.threshold_alpha_sweep \
    --mode collect \
    --config_a "${CONFIG_A}" \
    --config_b "${CONFIG_B}" \
    --sweep-dir "${SWEEP_DIR}"

echo "Finished at: $(date)"
exit 0
