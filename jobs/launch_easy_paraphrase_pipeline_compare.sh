#!/bin/bash
# Launcher: run base + easy paraphrase pipelines in parallel, then compare.
# Run with: bash jobs/launch_easy_paraphrase_pipeline_compare.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIRECTORY="/itet-stor/liweiss/net_scratch/vaa-question-similarity"

echo "=== Easy Paraphrase: pipeline + compare ==="

# Step 1: Submit both pipelines in parallel
export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_ZH.py"
JOB_BASE=$(sbatch --parsable --export=ALL --job-name=pipe_base "${SCRIPT_DIR}/job_pipeline_single.sh")
echo "  Base pipeline: ${JOB_BASE}"

export PIPELINE_CONFIG="configs/full_pipeline/cloned/easy_paraphrase_combinedvar_n10_e5_ZH.py"
JOB_CLONE=$(sbatch --parsable --export=ALL --job-name=pipe_easy_para "${SCRIPT_DIR}/job_pipeline_single.sh")
echo "  Clone pipeline: ${JOB_CLONE}"

# Step 2: Submit comparator that depends on both pipelines
# The comparator job needs to find the parquet files itself, so we use a small inline script.
COMPARE_JOB=$(sbatch --parsable --dependency=afterok:${JOB_BASE}:${JOB_CLONE} \
    --job-name=cmp_easy_para \
    --mem=16G --time=01:00:00 --cpus-per-task=4 --gres=gpu:0 --nodes=1 \
    --exclude=tikgpu10,tikgpu[06-09] \
    --output="${DIRECTORY}/jobs/out/%j.out" \
    --error="${DIRECTORY}/jobs/out/%j.err" \
    --wrap="
cd ${DIRECTORY}
[[ -f /itet-stor/liweiss/net_scratch/conda/bin/conda ]] && eval \"\$(/itet-stor/liweiss/net_scratch/conda/bin/conda shell.bash hook)\"
conda activate bachelor-thesis

RECS_CLEANED='${DIRECTORY}/experiment_results/pipeline_outputs/recommendations/cleaned'
RECS_CLONED='${DIRECTORY}/experiment_results/pipeline_outputs/recommendations/cloned'
BASE_REC=\$(ls -t \${RECS_CLEANED}/recs_pipeline_e5_ZH*.parquet 2>/dev/null | head -1)
CLONE_REC=\$(ls -t \${RECS_CLONED}/recs_easy_paraphrase_combinedvar_n10_e5_ZH*.parquet 2>/dev/null | head -1)

if [[ -z \"\${BASE_REC}\" || -z \"\${CLONE_REC}\" ]]; then
    echo 'ERROR: Could not find recommendation parquets' >&2
    exit 1
fi

echo \"Comparing: \${BASE_REC} vs \${CLONE_REC}\"
python -u -m comparator_main \"\${BASE_REC}\" \"\${CLONE_REC}\"
")
echo "  Comparator: ${COMPARE_JOB} (depends on ${JOB_BASE} + ${JOB_CLONE})"

echo ""
echo "  Pipelines run in parallel, comparator waits for both."
echo "  Monitor: squeue -u \$USER"
