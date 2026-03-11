#!/bin/bash
# Launcher: Extreme removal test — 37 of 75 questions removed (top half by impact).
# Sanity check to verify the pipeline shows a large effect with massive removal.
#
# Run with: bash jobs/launch_question_removal_extreme.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export OUTPUT_DIR="experiment_results/exp2_question_removal"

export CONFIG_A="configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py"
export CONFIG_B="configs/full_pipeline/removed/removed_top37_e5_instruct_ZH.py"

SUBFOLDER="alpha_sweep_pipeline_e5_instruct_ZH_vs_removed_top37_e5_instruct_ZH"
export SWEEP_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity/${OUTPUT_DIR}/${SUBFOLDER}/workers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

echo "=== Extreme Removal Test: 37 of 75 questions removed ==="
echo "  Sweep dir: ${SWEEP_DIR}"

SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-20 "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
echo "  Workers: job array ${SWEEP_JOB} (21 tasks)"

COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
echo "  Collect: job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo "  Monitor: squeue -u \$USER"
