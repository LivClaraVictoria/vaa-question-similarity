#!/bin/bash
# Launcher: Run base + 6 highvotervar clone pipelines in parallel, then compare each vs base.
# Assumes all clone datasets already exist in data/cloned/.
# Run with: bash jobs/launch_highvotervar_pipelines_compare.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIRECTORY="/itet-stor/liweiss/net_scratch/vaa-question-similarity"

CLONE_TYPES=(identical easy_paraphrase hard_paraphrase negation negation_easy negation_hard)
SUFFIX="highvotervar_2q_n5"

RECS_CLEANED="${DIRECTORY}/experiment_results/recommendation_results/cleaned"
RECS_CLONED="${DIRECTORY}/experiment_results/recommendation_results/cloned"

CONDA_INIT='[ -f /itet-stor/liweiss/net_scratch/conda/bin/conda ] && eval "$(/itet-stor/liweiss/net_scratch/conda/bin/conda shell.bash hook)" && conda activate bachelor-thesis'

echo "=== Highvotervar Pipelines + Comparisons (E5, ZH, default alpha) ==="
echo ""

# --- Step 1: Submit base + 6 cloned pipelines in parallel ---

export PIPELINE_OVERRIDES=""
export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_ZH.py"
JOB_BASE=$(sbatch --parsable --export=ALL --job-name="pipe_base_hv" "${SCRIPT_DIR}/job_pipeline_single.sh")
echo "  Base pipeline: ${JOB_BASE}"

PIPE_DEPS="${JOB_BASE}"
declare -A PIPE_JOBS

for TYPE in "${CLONE_TYPES[@]}"; do
    export PIPELINE_CONFIG="configs/full_pipeline/cloned/${TYPE}_${SUFFIX}_e5_ZH.py"
    JOB=$(sbatch --parsable --export=ALL --job-name="pipe_${TYPE}_hv" "${SCRIPT_DIR}/job_pipeline_single.sh")
    echo "  Pipeline ${TYPE}: ${JOB}"
    PIPE_DEPS="${PIPE_DEPS}:${JOB}"
    PIPE_JOBS[$TYPE]="${JOB}"
done

echo ""
echo "  7 pipeline jobs submitted in parallel."
echo "  Dependency string: ${PIPE_DEPS}"
echo ""

# --- Step 2: Submit 6 comparator jobs, all depend on all pipelines completing ---

SBATCH_COMPARE="--mem=16G --time=01:00:00 --cpus-per-task=4 --gres=gpu:0 --nodes=1 --exclude=tikgpu10,tikgpu[06-09]"

echo "=== Submitting comparator jobs (depend on all pipelines) ==="
echo ""

for TYPE in "${CLONE_TYPES[@]}"; do
    CMP_JOB=$(sbatch --parsable \
        --dependency=afterok:${PIPE_DEPS} \
        ${SBATCH_COMPARE} \
        --job-name="cmp_${TYPE}_hv" \
        --output="${DIRECTORY}/jobs/out/%j.out" \
        --error="${DIRECTORY}/jobs/out/%j.err" \
        --wrap="
set -o errexit
cd ${DIRECTORY}
${CONDA_INIT}

BASE_REC=\$(ls -t ${RECS_CLEANED}/recs_pipeline_e5_ZH_alpha~0.6*.parquet 2>/dev/null | head -1)
CLONE_REC=\$(ls -t ${RECS_CLONED}/recs_${TYPE}_${SUFFIX}_e5_ZH_alpha~0.6*.parquet 2>/dev/null | head -1)

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
echo "=== All jobs submitted ==="
echo "  Monitor: squeue -u \$USER"
