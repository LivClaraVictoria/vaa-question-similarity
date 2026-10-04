#!/bin/bash
# Worker: compare two recommendation parquets and write a Jaccard/Spearman/rank-shift report.
# Env vars: REC_A (baseline parquet path), REC_B (cloned/modified parquet path).
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4

# Generic single comparator runner. Expects REC_A and REC_B env vars (parquet paths).

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Comparing: ${REC_A} vs ${REC_B}"

python -u -m main compare "${REC_A}" "${REC_B}"

echo "Finished at: $(date)"
exit 0
