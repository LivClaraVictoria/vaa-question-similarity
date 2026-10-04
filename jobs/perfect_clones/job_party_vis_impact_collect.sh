#!/bin/bash
# Collect: aggregate party visibility impact per-question CSVs and produce ranked summary plots.
# Env vars: PIPELINE_CONFIG, SWEEP_DIR.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4

# Party visibility impact collect — aggregates per-question worker CSVs,
# computes answer correlations, and generates all plots + reports.
# Launched with --dependency by launch_party_vis_impact.sh.
# Expects: SWEEP_DIR, PIPELINE_CONFIG (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m experiments.perfect_clones.party_visibility_impact \
    --mode collect \
    --config "${PIPELINE_CONFIG}" \
    --sweep-dir "${SWEEP_DIR}"

echo "Finished at: $(date)"
exit 0
