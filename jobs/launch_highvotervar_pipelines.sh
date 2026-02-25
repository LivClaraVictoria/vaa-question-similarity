#!/bin/bash
# Launcher: Run base + 6 highvotervar clone pipelines in parallel.
# Assumes all clone datasets already exist in data/cloned/.
# Run with: bash jobs/launch_highvotervar_pipelines.sh
# Then run: bash jobs/launch_highvotervar_comparisons.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CLONE_TYPES=(identical easy_paraphrase hard_paraphrase negation negation_easy negation_hard)
SUFFIX="highvotervar_2q_n5"

echo "=== Highvotervar Pipelines (E5, ZH, default alpha) ==="
echo ""

export PIPELINE_OVERRIDES=""

export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_ZH.py"
JOB_BASE=$(sbatch --parsable --export=ALL --job-name="pipe_base_hv" "${SCRIPT_DIR}/job_pipeline_single.sh")
echo "  Base pipeline: ${JOB_BASE}"

for TYPE in "${CLONE_TYPES[@]}"; do
    export PIPELINE_CONFIG="configs/full_pipeline/cloned/${TYPE}_${SUFFIX}_e5_ZH.py"
    JOB=$(sbatch --parsable --export=ALL --job-name="pipe_${TYPE}_hv" "${SCRIPT_DIR}/job_pipeline_single.sh")
    echo "  Pipeline ${TYPE}: ${JOB}"
done

echo ""
echo "=== 7 pipeline jobs submitted (all parallel) ==="
echo "  Monitor: squeue -u \$USER"
echo "  When done, run: bash jobs/launch_highvotervar_comparisons.sh"
