#!/bin/bash
# Collect: aggregate per-alpha model selection CSVs into a combined sweep result with plots.
# Env vars: CONFIG_A, CONFIG_B, SWEEP_DIR.
#SBATCH --mail-type=NONE
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2

# Model selection collect — aggregates per-alpha worker CSVs and generates plots.
# Launched with --dependency by launch_model_selection.sh.
# Expects: SWEEP_DIR, CONFIG_A, CONFIG_B (via --export).
# Optional: OUTPUT_DIR — if set, passes --output-dir to model_selection.

source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
job_preamble

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

OUTPUT_DIR_FLAG=""
[[ -n "${OUTPUT_DIR}" ]] && OUTPUT_DIR_FLAG="--output-dir ${OUTPUT_DIR}"

python -u -m experiments.perfect_clones.model_selection \
    --mode collect \
    --config_a "${CONFIG_A}" \
    --config_b "${CONFIG_B}" \
    --sweep-dir "${SWEEP_DIR}" \
    ${OUTPUT_DIR_FLAG}

echo "Finished at: $(date)"
exit 0
