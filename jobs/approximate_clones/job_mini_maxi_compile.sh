#!/bin/bash
# Compile: enrich mini/maxi Phase 1 CSV with corr-weighted scores and selection indicators.
# No env vars required; reads from experiment_results/party_impact/mini_maxi/.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2

# Mini-maxi compile — aggregate all per-party Phase 2 CSVs into compiled outputs.
# Launched with --dependency by launch_partisan_full.sh.
# Expects: PIPELINE_CONFIG (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

COMPILE_ARGS="--mode compile --config ${PIPELINE_CONFIG}"
[[ -n "${SELECTION_MODE}" ]] && COMPILE_ARGS="${COMPILE_ARGS} --selection-mode ${SELECTION_MODE}"

python -u -m experiments.approximate_clones.partisan_distortion ${COMPILE_ARGS}

echo "Finished at: $(date)"
exit 0
