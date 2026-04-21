#!/bin/bash
# Launcher: create 3 identical cloned datasets in parallel.
# Run with: bash jobs/_generic/launch_create_clones.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Create Clones (3 parallel jobs) ==="

CLONE_CONFIG="configs/create_clones/identical_q32214_n10.py" \
    sbatch --parsable --job-name=clone_q32214 "${SCRIPT_DIR}/job_create_clone_single.sh"

CLONE_CONFIG="configs/create_clones/identical_highcandvar_n10.py" \
    sbatch --parsable --job-name=clone_highcandvar "${SCRIPT_DIR}/job_create_clone_single.sh"

CLONE_CONFIG="configs/create_clones/identical_combinedvar_n10.py" \
    sbatch --parsable --job-name=clone_combinedvar "${SCRIPT_DIR}/job_create_clone_single.sh"

echo "  3 clone jobs submitted. Monitor: squeue -u \$USER"
