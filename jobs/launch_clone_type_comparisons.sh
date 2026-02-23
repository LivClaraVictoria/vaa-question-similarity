#!/bin/bash
# Launcher: Compare base vs each clone type (alpha=0.9, E5, ZH).
# Run after launch_clone_type_pipelines.sh has completed.
# Run with: bash jobs/launch_clone_type_comparisons.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIRECTORY="/itet-stor/liweiss/net_scratch/vaa-question-similarity"

RECS_CLEANED="${DIRECTORY}/experiment_results/recommendation_results/cleaned"
RECS_CLONED="${DIRECTORY}/experiment_results/recommendation_results/cloned"

CONDA_INIT='[ -f /itet-stor/liweiss/net_scratch/conda/bin/conda ] && eval "$(/itet-stor/liweiss/net_scratch/conda/bin/conda shell.bash hook)" && conda activate bachelor-thesis'

SBATCH_COMPARE="--mem=16G --time=01:00:00 --cpus-per-task=4 --gres=gpu:0 --nodes=1 --exclude=tikgpu10,tikgpu[06-09]"

CLONE_TYPES=(easy_paraphrase hard_paraphrase negation negation_easy negation_hard)

echo "=== Clone Type Comparisons (alpha=0.9, E5, ZH) ==="
echo ""

for TYPE in "${CLONE_TYPES[@]}"; do
    CMP_JOB=$(sbatch --parsable \
        ${SBATCH_COMPARE} \
        --job-name="cmp_${TYPE}_a09" \
        --output="${DIRECTORY}/jobs/out/%j.out" \
        --error="${DIRECTORY}/jobs/out/%j.err" \
        --wrap="
set -o errexit
cd ${DIRECTORY}
${CONDA_INIT}

BASE_REC=\$(ls -t ${RECS_CLEANED}/recs_pipeline_e5_ZH_alpha~0.9*.parquet 2>/dev/null | head -1)
CLONE_REC=\$(ls -t ${RECS_CLONED}/recs_${TYPE}_combinedvar_n10_e5_ZH_alpha~0.9*.parquet 2>/dev/null | head -1)

if [ -z \"\${BASE_REC}\" ] || [ -z \"\${CLONE_REC}\" ]; then
    echo \"ERROR: Could not find recommendation parquets for ${TYPE}\" >&2
    echo \"  BASE_REC=\${BASE_REC}\" >&2
    echo \"  CLONE_REC=\${CLONE_REC}\" >&2
    exit 1
fi

echo \"Comparing: \${BASE_REC} vs \${CLONE_REC}\"
python -u -m comparator_main \"\${BASE_REC}\" \"\${CLONE_REC}\"
echo \"Finished at: \$(date)\"
")
    echo "  Compare base vs ${TYPE}: ${CMP_JOB}"
done

echo ""
echo "=== 5 comparator jobs submitted (all parallel) ==="
echo "  Monitor: squeue -u \$USER"
