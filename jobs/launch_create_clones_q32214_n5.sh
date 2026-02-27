#!/bin/bash
# Launcher: Create all q32214 n5 cloned datasets (serial — paraphrase cache race condition).
# Also regenerates natural_mixed_q32214_n5 since the underlying data changed.
# Run with: bash jobs/launch_create_clones_q32214_n5.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Clone Creation: q32214 n5 variants (serial) ==="
echo ""

export CLONE_CONFIG="configs/create_clones/negation_q32214_n5.py"
JOB1=$(sbatch --parsable --export=ALL --job-name=clone_neg5 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation_n5:      ${JOB1}"

export CLONE_CONFIG="configs/create_clones/easy_paraphrase_q32214_n5.py"
JOB2=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB1} --job-name=clone_ep5 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  easy_paraphrase_n5: ${JOB2} (after ${JOB1})"

export CLONE_CONFIG="configs/create_clones/hard_paraphrase_q32214_n5.py"
JOB3=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB2} --job-name=clone_hp5 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  hard_paraphrase_n5: ${JOB3} (after ${JOB2})"

export CLONE_CONFIG="configs/create_clones/negation_easy_q32214_n5.py"
JOB4=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB3} --job-name=clone_ne5 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation_easy_n5: ${JOB4} (after ${JOB3})"

export CLONE_CONFIG="configs/create_clones/negation_hard_q32214_n5.py"
JOB5=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB4} --job-name=clone_nh5 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation_hard_n5: ${JOB5} (after ${JOB4})"

export CLONE_CONFIG="configs/create_clones/natural_mixed_q32214_n5.py"
JOB6=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB5} --job-name=clone_nat5 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  natural_mixed_n5: ${JOB6} (after ${JOB5})"

echo ""
echo "=== 6 clone creation jobs submitted (serial chain) ==="
echo "  Chain: ${JOB1} -> ${JOB2} -> ${JOB3} -> ${JOB4} -> ${JOB5} -> ${JOB6}"
echo "  Monitor: squeue -u \$USER"
