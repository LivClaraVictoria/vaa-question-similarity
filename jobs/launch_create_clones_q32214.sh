#!/bin/bash
# Launcher: Create Q32214 clone datasets (serial due to paraphrase cache race condition).
# Paraphrase types need OPENAI_API_KEY in environment.
# identical_q32214_n10 already exists — only creates the 6 new datasets.
# Run with: bash jobs/launch_create_clones_q32214.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Create Q32214 Clone Datasets (serial) ==="
echo ""

# First job: easy_paraphrase (triggers paraphrase generation for Q32214)
export CLONE_CONFIG="configs/create_clones/easy_paraphrase_q32214_n10.py"
JOB1=$(sbatch --parsable --export=ALL --job-name=clone_ep_q32214 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  easy_paraphrase:  ${JOB1}"

# Chain remaining jobs sequentially (paraphrase cache race condition)
export CLONE_CONFIG="configs/create_clones/hard_paraphrase_q32214_n10.py"
JOB2=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB1} --job-name=clone_hp_q32214 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  hard_paraphrase:  ${JOB2} (after ${JOB1})"

export CLONE_CONFIG="configs/create_clones/negation_q32214_n10.py"
JOB3=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB2} --job-name=clone_neg_q32214 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation:         ${JOB3} (after ${JOB2})"

export CLONE_CONFIG="configs/create_clones/negation_easy_q32214_n10.py"
JOB4=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB3} --job-name=clone_ne_q32214 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation_easy:    ${JOB4} (after ${JOB3})"

export CLONE_CONFIG="configs/create_clones/negation_hard_q32214_n10.py"
JOB5=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB4} --job-name=clone_nh_q32214 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation_hard:    ${JOB5} (after ${JOB4})"

export CLONE_CONFIG="configs/create_clones/natural_mixed_q32214_n5.py"
JOB6=$(sbatch --parsable --export=ALL --dependency=afterok:${JOB5} --job-name=clone_nat_q32214 "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  natural_mixed:    ${JOB6} (after ${JOB5})"

echo ""
echo "=== 6 clone creation jobs submitted (serial chain) ==="
echo "  Last job ID: ${JOB6} (all done when this completes)"
echo "  Monitor: squeue -u \$USER"
