#!/bin/bash
# Collect: aggregate per-question alpha sweep worker CSVs into compiled results with plots.
# Env vars: PIPELINE_CONFIG, SWEEP_DIR.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2

# Question alpha sweep collect — aggregates per-question worker CSVs and generates plots.
# Launched with --dependency by launch_question_alpha_sweep.sh.
# Expects: SWEEP_DIR, PIPELINE_CONFIG (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m experiments.perfect_clones.recommendation_distortion \
    --mode collect \
    --config "${PIPELINE_CONFIG}" \
    --sweep-dir "${SWEEP_DIR}" \
    ${ALPHAS:+--alphas "${ALPHAS}"}

echo "Finished at: $(date)"
exit 0
