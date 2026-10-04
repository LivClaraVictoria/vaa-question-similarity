#!/bin/bash
# Test run: smoke-test a single clone count sweep worker with a small subset to verify setup.
# Env vars: PIPELINE_CONFIG, SWEEP_DIR.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4

# Quick test: clone count sweep for a single question with fewer n_values.

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m experiments.perfect_clones.clone_count_sweep \
    --config configs/base_pipeline/pipeline_e5_ZH.py \
    --mode worker --task-id 0 --n-values 1,5,10 \
    --sweep-dir experiment_results/clone_count_sweep/test_workers

echo "Finished at: $(date)"
exit 0
