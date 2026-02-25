#!/bin/bash
# Launcher: create 6 cloned datasets (one per clone type) using high_voter_variance selector.
# Jobs run in series to avoid paraphrase cache race conditions.
# Run with: bash jobs/launch_create_clones_highvotervar.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Create Clones: highvotervar 2q x 5 (6 types, serial) ==="

JOB1=$(CLONE_CONFIG="configs/create_clones/identical_highvotervar_2q_n5.py" \
    sbatch --parsable --job-name=clone_identical_hvv \
    "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  [1/6] identical        -> job $JOB1"

JOB2=$(CLONE_CONFIG="configs/create_clones/easy_paraphrase_highvotervar_2q_n5.py" \
    sbatch --parsable --job-name=clone_easy_hvv \
    --dependency=afterok:${JOB1} \
    "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  [2/6] easy_paraphrase  -> job $JOB2 (after $JOB1)"

JOB3=$(CLONE_CONFIG="configs/create_clones/hard_paraphrase_highvotervar_2q_n5.py" \
    sbatch --parsable --job-name=clone_hard_hvv \
    --dependency=afterok:${JOB2} \
    "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  [3/6] hard_paraphrase  -> job $JOB3 (after $JOB2)"

JOB4=$(CLONE_CONFIG="configs/create_clones/negation_highvotervar_2q_n5.py" \
    sbatch --parsable --job-name=clone_neg_hvv \
    --dependency=afterok:${JOB3} \
    "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  [4/6] negation         -> job $JOB4 (after $JOB3)"

JOB5=$(CLONE_CONFIG="configs/create_clones/negation_easy_highvotervar_2q_n5.py" \
    sbatch --parsable --job-name=clone_neg_easy_hvv \
    --dependency=afterok:${JOB4} \
    "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  [5/6] negation_easy    -> job $JOB5 (after $JOB4)"

JOB6=$(CLONE_CONFIG="configs/create_clones/negation_hard_highvotervar_2q_n5.py" \
    sbatch --parsable --job-name=clone_neg_hard_hvv \
    --dependency=afterok:${JOB5} \
    "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  [6/6] negation_hard    -> job $JOB6 (after $JOB5)"

echo ""
echo "All 6 jobs queued in series. Monitor: squeue -u \$USER"
