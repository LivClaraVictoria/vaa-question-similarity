#!/bin/bash
# Launcher: Run allq question sweeps (5 clone types, E5-INSTRUCT, alpha=0.3).
# Assumes allq clone datasets already exist (run launch_create_clones_allq.sh first).
# Run with: bash jobs/launch_allq_sweeps_e5_instruct.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  All-Questions Sweep (E5-INSTRUCT, alpha=0.3)"
echo "=========================================="
echo ""

export CONFIG_A="configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py"

CLONE_TYPES=("easy_paraphrase_allq_n4" "hard_paraphrase_allq_n4" "negation_easy_allq_n4" "negation_hard_allq_n4" "perfect_mix_allq_n4")

SWEEP_COUNT=0
for CLONE in "${CLONE_TYPES[@]}"; do
    export CONFIG_B="configs/full_pipeline/cloned/${CLONE}_e5_instruct_ZH.py"

    JOB=$(sbatch --parsable --export=ALL \
        --job-name="allq_${CLONE:0:12}" \
        "${SCRIPT_DIR}/job_allq_sweep_single.sh")
    echo "  ${CLONE}: job=${JOB}"
    SWEEP_COUNT=$((SWEEP_COUNT + 1))
done

echo ""
echo "=== Experiment submitted ==="
echo "  Comparisons: ${SWEEP_COUNT} (parallel, each ~30-60 min)"
echo "  Monitor: squeue -u \$USER"
