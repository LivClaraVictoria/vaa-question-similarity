#!/bin/bash
# Compile: merge party impact Phase 2 results across multiple models/conditions into one report.
# No env vars required; reads from experiment_results/party_impact/high_impact/.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2

# Party impact compile — aggregate all per-party Phase 2 CSVs into compiled outputs.
# Expects: PIPELINE_CONFIG (via --export).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m experiments.perfect_clones.partisan_distortion --mode compile --config ${PIPELINE_CONFIG}

echo "Finished at: $(date)"
exit 0
