#!/bin/bash
# Launcher: Compare base vs each highvotervar clone type (E5, ZH, default alpha).
# Run after launch_highvotervar_pipelines_compare.sh has completed.
# Run with: bash jobs/launch_highvotervar_comparisons.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIRECTORY="/itet-stor/liweiss/net_scratch/vaa-question-similarity"

RECS_CLEANED="${DIRECTORY}/experiment_results/pipeline_outputs/recommendations/cleaned"
RECS_CLONED="${DIRECTORY}/experiment_results/pipeline_outputs/recommendations/cloned"

CONDA_INIT='[ -f /itet-stor/liweiss/net_scratch/conda/bin/conda ] && eval "$(/itet-stor/liweiss/net_scratch/conda/bin/conda shell.bash hook)" && conda activate bachelor-thesis'

SBATCH_COMPARE="--mem=16G --time=01:00:00 --cpus-per-task=4 --gres=gpu:0 --nodes=1 --exclude=tikgpu10,tikgpu[06-09]"

CLONE_TYPES=(identical easy_paraphrase hard_paraphrase negation negation_easy negation_hard)
SUFFIX="highvotervar_2q_n5"

echo "=== Highvotervar Comparisons (E5, ZH, default alpha) ==="
echo ""

for TYPE in "${CLONE_TYPES[@]}"; do
    CMP_JOB=$(sbatch --parsable \
        ${SBATCH_COMPARE} \
        --job-name="cmp_${TYPE}_hv" \
        --output="${DIRECTORY}/jobs/out/%j.out" \
        --error="${DIRECTORY}/jobs/out/%j.err" \
        --wrap="
set -o errexit
cd ${DIRECTORY}
${CONDA_INIT}

BASE_REC=\$(ls -t ${RECS_CLEANED}/recs_pipeline_e5_ZH*.parquet 2>/dev/null | head -1)
CLONE_REC=\$(ls -t ${RECS_CLONED}/${TYPE}_${SUFFIX}/recs_${TYPE}_${SUFFIX}_e5_ZH*.parquet 2>/dev/null | head -1)

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
echo "=== 6 comparator jobs submitted (all parallel) ==="
echo "  Monitor: squeue -u \$USER"
