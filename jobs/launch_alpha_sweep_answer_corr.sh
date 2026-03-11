#!/bin/bash
# Launcher: alpha sweep for answer-correlation metric.
# Base (75q) vs easy_paraphrase_top5impact_n4 (95q).
# Submits a 21-task job array (one per alpha) + a dependent collect job.
# Run with: bash jobs/launch_alpha_sweep_answer_corr.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBFOLDER="alpha_sweep_pipeline_answer_corr_ZH_vs_easy_paraphrase_top5impact_n4_answer_corr_ZH"
SWEEP_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/alpha_sweep_results/${SUBFOLDER}/workers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export CONFIG_A="configs/full_pipeline/base_data/pipeline_answer_corr_ZH.py"
export CONFIG_B="configs/full_pipeline/cloned/easy_paraphrase_top5impact_n4_answer_corr_ZH.py"
export SWEEP_DIR

echo "=== Alpha Sweep: answer-correlation (easy_paraphrase_top5impact_n4) ==="
echo "  Sweep dir: ${SWEEP_DIR}"

# 21 alpha values (indices 0-20)
# Reduced resources: no GPU, no embedding model — just Pearson correlations + recs
SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-20 \
    --mem=16G --time=01:00:00 --cpus-per-task=4 \
    "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
echo "  Workers submitted: job array ${SWEEP_JOB} (21 tasks)"

COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
echo "  Collect submitted:  job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo "  Monitor: squeue -u \$USER"
