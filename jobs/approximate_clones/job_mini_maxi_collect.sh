#!/bin/bash
# Collect: aggregate mini/maxi Phase 1 per-question CSVs into a summary with heatmap and per-party plots.
# Env vars: PIPELINE_CONFIG, SWEEP_DIR, TOP_K, TARGET_PARTY.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2

# Mini-maxi collect — aggregates per-question worker CSVs and generates
# Phase 1 plots (heatmap, per-party ranking, scatter, report).
# Launched with --dependency by launch_partisan_full.sh.
# Expects: SWEEP_DIR, PIPELINE_CONFIG (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m experiments.approximate_clones.partisan_distortion \
    --mode collect \
    --config "${PIPELINE_CONFIG}" \
    --sweep-dir "${SWEEP_DIR}"

echo "Finished at: $(date)"
exit 0
