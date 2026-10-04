#!/bin/bash
# Collect: aggregate noise-slider worker CSVs → master + aggregated + plot + report.
# Env vars: PIPELINE_CONFIG, SWEEP_DIR, ALPHA.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2

# Noise slider collect — aggregates per-question worker CSVs, computes trimmed stats,
# writes master/aggregated CSVs + plot + report.
# Launched with --dependency by launch_robustness.sh.
# Expects: SWEEP_DIR, PIPELINE_CONFIG, ALPHA (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m main noise-slider \
    --mode collect \
    --config "${PIPELINE_CONFIG}" \
    --sweep-dir "${SWEEP_DIR}" \
    --alpha "${ALPHA}"

echo "Finished at: $(date)"
exit 0
