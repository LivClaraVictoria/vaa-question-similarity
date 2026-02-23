#!/bin/bash
# Launcher: Run base + 5 clone pipelines in parallel (alpha=0.9, E5, ZH).
# Assumes all clone datasets already exist in data/cloned/.
# Run with: bash jobs/launch_clone_type_pipelines.sh
# Then run: bash jobs/launch_clone_type_comparisons.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CLONE_TYPES=(easy_paraphrase hard_paraphrase negation negation_easy negation_hard)

echo "=== Clone Type Pipelines (alpha=0.9, E5, ZH) ==="
echo ""

export PIPELINE_OVERRIDES="alpha=0.9"

export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_ZH.py"
JOB_BASE=$(sbatch --parsable --export=ALL --job-name=pipe_base_a09 "${SCRIPT_DIR}/job_pipeline_single.sh")
echo "  Base pipeline: ${JOB_BASE}"

for TYPE in "${CLONE_TYPES[@]}"; do
    export PIPELINE_CONFIG="configs/full_pipeline/cloned/${TYPE}_combinedvar_n10_e5_ZH.py"
    JOB=$(sbatch --parsable --export=ALL --job-name="pipe_${TYPE}_a09" "${SCRIPT_DIR}/job_pipeline_single.sh")
    echo "  Pipeline ${TYPE}: ${JOB}"
done

echo ""
echo "=== 6 pipeline jobs submitted (all parallel) ==="
echo "  Monitor: squeue -u \$USER"
echo "  When done, run: bash jobs/launch_clone_type_comparisons.sh"
