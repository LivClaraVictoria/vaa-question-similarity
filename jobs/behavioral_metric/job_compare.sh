#!/bin/bash
# Worker: compare BEHAVIORAL-L1 vs ANSWER-CORRELATION-ARCCOS distance matrices on V∪C.
# Lightweight CPU job (two answer-based distance matrices + merge + scatter + report).
# Env vars: COMPARE_CONFIG (config path; default behavioral L1 ZH).
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

python -u -m main behavioral-compare --config "${COMPARE_CONFIG:-configs/base_pipeline/pipeline_behavioral_l1_ZH.py}"

echo "Finished at: $(date)"
exit 0
