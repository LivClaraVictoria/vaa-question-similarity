#!/bin/bash
# Launcher: Create all allq (all 75 questions) cloned datasets (serial — paraphrase cache race condition).
# Run with: bash jobs/launch_create_clones_allq.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Clone Creation: allq n4 variants (serial) ==="
echo ""

export CLONE_CONFIG="configs/create_clones/easy_paraphrase_allq_n4.py"
JOB1=$(sbatch --parsable --export=ALL --job-name=clone_ep_allq "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  easy_paraphrase_allq_n4:  ${JOB1}"

export CLONE_CONFIG="configs/create_clones/hard_paraphrase_allq_n4.py"
JOB2=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB1} --job-name=clone_hp_allq "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  hard_paraphrase_allq_n4:  ${JOB2} (after ${JOB1})"

export CLONE_CONFIG="configs/create_clones/negation_easy_allq_n4.py"
JOB3=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB2} --job-name=clone_ne_allq "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation_easy_allq_n4:    ${JOB3} (after ${JOB2})"

export CLONE_CONFIG="configs/create_clones/negation_hard_allq_n4.py"
JOB4=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB3} --job-name=clone_nh_allq "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation_hard_allq_n4:    ${JOB4} (after ${JOB3})"

export CLONE_CONFIG="configs/create_clones/perfect_mix_allq_n4.py"
JOB5=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB4} --job-name=clone_pm_allq "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  perfect_mix_allq_n4:      ${JOB5} (after ${JOB4})"

echo ""
echo "=== 5 clone creation jobs submitted (serial chain) ==="
echo "  Chain: ${JOB1} -> ${JOB2} -> ${JOB3} -> ${JOB4} -> ${JOB5}"
echo "  Monitor: squeue -u \$USER"
