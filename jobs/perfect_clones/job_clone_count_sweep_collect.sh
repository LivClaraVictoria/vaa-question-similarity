#!/bin/bash
# Collect: aggregate per-question clone count sweep CSVs into a combined result with plots.
# Env vars: PIPELINE_CONFIG, SWEEP_DIR.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2

# Clone count sweep collect — aggregates per-question worker CSVs and generates
# plots + report.
# Launched with --dependency by launch_clone_count_sweep.sh.
# Expects: SWEEP_DIR, PIPELINE_CONFIG (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m experiments.perfect_clones.clone_count_sweep \
    --mode collect \
    --config "${PIPELINE_CONFIG}" \
    --sweep-dir "${SWEEP_DIR}"

echo "Finished at: $(date)"
exit 0
