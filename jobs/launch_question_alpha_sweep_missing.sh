#!/bin/bash
# Launcher: resubmit only the missing question_alpha_sweep workers
# into the existing sweep directory, then run collect.
#
# Run with: bash jobs/launch_question_alpha_sweep_missing.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"

# Reuse the existing sweep directory (don't create a new one)
export SWEEP_DIR="${PROJECT_DIR}/experiment_results/question_alpha_sweep_results/workers_allct_20260304_002148"
export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py"
export N_CLONES=4

echo "=== Resubmitting missing question_alpha_sweep workers ==="
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Config: ${PIPELINE_CONFIG}"
echo ""

ALL_JOB_IDS=""

submit_worker() {
    local task_id=$1
    local clone_type=$2
    export CLONE_TYPE="${clone_type}"

    JOB_ID=$(sbatch --parsable --export=ALL \
        --array="${task_id}" \
        --job-name="qa_fix_${clone_type}_${task_id}" \
        "${SCRIPT_DIR}/job_question_alpha_sweep_worker_ct.sh")
    echo "  ${clone_type} task-id=${task_id}: job ${JOB_ID}"

    if [ -z "${ALL_JOB_IDS}" ]; then
        ALL_JOB_IDS="${JOB_ID}"
    else
        ALL_JOB_IDS="${ALL_JOB_IDS}:${JOB_ID}"
    fi
}

# easy_paraphrase: 8 missing
for tid in 2 5 6 7 9 11 14 17; do
    submit_worker $tid easy_paraphrase
done

# hard_paraphrase: 3 missing
for tid in 35 37 50; do
    submit_worker $tid hard_paraphrase
done

# negation_easy: 1 missing
submit_worker 42 negation_easy

# negation_hard: 1 missing
submit_worker 55 negation_hard

# perfect_mix: 5 missing
for tid in 39 56 60 61 63; do
    submit_worker $tid perfect_mix
done

echo ""
echo "--- Submitting collect job ---"
COLLECT_JOB=$(sbatch --parsable --export=ALL \
    --dependency=afterok:${ALL_JOB_IDS} \
    "${SCRIPT_DIR}/job_question_alpha_sweep_collect.sh")
echo "  Collect: job ${COLLECT_JOB} (depends on all workers)"

echo ""
echo "  Total: 18 workers + 1 collect job"
echo "  Monitor: squeue -u \$USER"
