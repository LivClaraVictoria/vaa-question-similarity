#!/bin/bash
# Launcher: Experiment 2 test — two alpha sweeps with question removal.
#   1) Health top-3 impact removed (3 questions, 72 remaining)
#   2) Health + Welfare top-3 each removed (6 questions, 69 remaining)
# Both run as 21-task job arrays + dependent collect jobs.
# Results go to experiment_results/question_removal_results/.
#
# Run with: bash jobs/launch_question_removal_test.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export OUTPUT_DIR="experiment_results/question_removal_results"

CONFIG_A="configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py"

# --- Run 1: Health top-3 ---
CONFIG_B1="configs/full_pipeline/removed/removed_health_top3_e5_instruct_ZH.py"
SUBFOLDER1="alpha_sweep_pipeline_e5_instruct_ZH_vs_removed_health_top3_e5_instruct_ZH"
SWEEP_DIR1="/itet-stor/liweiss/net_scratch/vaa-question-similarity/${OUTPUT_DIR}/${SUBFOLDER1}/workers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR1}"

export CONFIG_A CONFIG_B="${CONFIG_B1}" SWEEP_DIR="${SWEEP_DIR1}"
SWEEP_JOB1=$(sbatch --parsable --export=ALL --array=0-20 "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
echo "=== Run 1: Health top-3 removed ==="
echo "  Workers: job array ${SWEEP_JOB1} (21 tasks)"

COLLECT_JOB1=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB1} "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
echo "  Collect: job ${COLLECT_JOB1} (depends on ${SWEEP_JOB1})"

# --- Run 2: Health + Welfare top-3 each ---
CONFIG_B2="configs/full_pipeline/removed/removed_health_welfare_top6_e5_instruct_ZH.py"
SUBFOLDER2="alpha_sweep_pipeline_e5_instruct_ZH_vs_removed_health_welfare_top6_e5_instruct_ZH"
SWEEP_DIR2="/itet-stor/liweiss/net_scratch/vaa-question-similarity/${OUTPUT_DIR}/${SUBFOLDER2}/workers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR2}"

export CONFIG_B="${CONFIG_B2}" SWEEP_DIR="${SWEEP_DIR2}"
SWEEP_JOB2=$(sbatch --parsable --export=ALL --array=0-20 "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
echo ""
echo "=== Run 2: Health + Welfare top-6 removed ==="
echo "  Workers: job array ${SWEEP_JOB2} (21 tasks)"

COLLECT_JOB2=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB2} "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
echo "  Collect: job ${COLLECT_JOB2} (depends on ${SWEEP_JOB2})"

echo ""
echo "Total: 42 workers + 2 collect jobs"
echo "Monitor: squeue -u \$USER"
