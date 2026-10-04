#!/bin/bash
# Worker: run the model benchmark evaluation on fake benchmark distance CSVs.
# Env vars: DIST_CONFIG (path to distance-only config for fake data).
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2

# Evaluate all embedding models on the fake benchmark.
# Reads distance CSVs from experiment_results/pipeline_outputs/distance_metrics/fake_results/
# Outputs comparison CSV + plot to experiment_results/model_benchmark/

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m experiments.explanatory.model_benchmark

echo "Finished at: $(date)"
exit 0
