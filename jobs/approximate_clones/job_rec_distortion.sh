#!/bin/bash
# Approximate clone alpha sweep: tests CRW on top-5 correlated full-only questions added to mini.
# Env vars: PIPELINE_CONFIG (mini config), SWEEP_DIR.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8

# Approximate clone alpha sweep: adds top-5 most correlated full-only questions
# to the mini questionnaire and tests CRW correction with 3 metrics.

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m experiments.approximate_clones.recommendation_distortion \
    --config configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py \
    --top-k 5

echo "Finished at: $(date)"
exit 0
