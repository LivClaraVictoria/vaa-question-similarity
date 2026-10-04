#!/bin/bash
# Collect: aggregate per-question impact CSVs + run correlation analysis + produce ranking plots.
# Env vars: PIPELINE_CONFIG, SWEEP_DIR.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4

# Question impact collect — aggregates per-question worker CSVs, computes
# correlation analysis, and generates plots.
# Launched with --dependency by launch_question_impact.sh.
# Expects: SWEEP_DIR, PIPELINE_CONFIG (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m experiments.explanatory.question_impact \
    --mode collect \
    --config "${PIPELINE_CONFIG}" \
    --sweep-dir "${SWEEP_DIR}"

echo "Finished at: $(date)"
exit 0
