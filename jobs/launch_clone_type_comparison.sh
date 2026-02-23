#!/bin/bash
# Launcher: Compare 5 clone types against baseline, all with alpha=0.9, E5, ZH.
# Clone types: easy_paraphrase, hard_paraphrase, negation, negation_easy, negation_hard
# Each: 10 clones of 1 question (combinedvar selector).
#
# Dependency structure:
#   Phase 1a: base pipeline + easy_paraphrase pipeline (parallel, no deps)
#   Phase 1b: 4 clone creations in SERIES (to avoid JSON race condition on paraphrase cache)
#   Phase 2:  4 clone pipelines, each depends on its own clone creation
#   Phase 3:  5 comparators, each depends on base pipeline + its clone pipeline
#
# Prerequisites: OPENAI_API_KEY must be set (needed for paraphrase generation).
# Run with: bash jobs/launch_clone_type_comparison.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIRECTORY="/itet-stor/liweiss/net_scratch/vaa-question-similarity"

RECS_CLEANED="${DIRECTORY}/experiment_results/recommendation_results/cleaned"
RECS_CLONED="${DIRECTORY}/experiment_results/recommendation_results/cloned"

# Conda activation snippet (reused in --wrap comparator jobs)
CONDA_INIT='[ -f /itet-stor/liweiss/net_scratch/conda/bin/conda ] && eval "$(/itet-stor/liweiss/net_scratch/conda/bin/conda shell.bash hook)" && conda activate bachelor-thesis'

# Comparator sbatch resource options
SBATCH_COMPARE="--mem=16G --time=01:00:00 --cpus-per-task=4 --gres=gpu:0 --nodes=1 --exclude=tikgpu10,tikgpu[06-09]"

CLONE_TYPES=(easy_paraphrase hard_paraphrase negation negation_easy negation_hard)

# Clone types that need dataset creation (easy_paraphrase already exists in data/cloned/)
CREATE_TYPES=(hard_paraphrase negation negation_easy negation_hard)

# Check OPENAI_API_KEY
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "WARNING: OPENAI_API_KEY is not set. Clone creation for paraphrase types will fail."
    echo "Set it with: export OPENAI_API_KEY=<key>"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

echo "=== Clone Type Comparison (alpha=0.9, E5, ZH, 5 types) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 1a: Base pipeline with alpha=0.9
# ---------------------------------------------------------------------------
export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_ZH.py"
export PIPELINE_OVERRIDES="alpha=0.9"
JOB_BASE=$(sbatch --parsable --export=ALL --job-name=pipe_base_a09 "${SCRIPT_DIR}/job_pipeline_single.sh")
echo "  Base pipeline (alpha=0.9): ${JOB_BASE}"

# ---------------------------------------------------------------------------
# Phase 1b: Create 4 clone datasets in SERIES to avoid race on paraphrase cache JSON
# ---------------------------------------------------------------------------
declare -A JOB_CREATE
PREV_CREATE_JOB=""

for TYPE in "${CREATE_TYPES[@]}"; do
    export CLONE_CONFIG="configs/create_clones/${TYPE}_combinedvar_n10.py"
    if [[ -z "${PREV_CREATE_JOB}" ]]; then
        JOB_CREATE[${TYPE}]=$(sbatch --parsable --export=ALL \
            --job-name="clone_${TYPE}" \
            "${SCRIPT_DIR}/job_create_clone_single.sh")
    else
        JOB_CREATE[${TYPE}]=$(sbatch --parsable --export=ALL \
            --dependency=afterok:${PREV_CREATE_JOB} \
            --job-name="clone_${TYPE}" \
            "${SCRIPT_DIR}/job_create_clone_single.sh")
    fi
    PREV_CREATE_JOB=${JOB_CREATE[${TYPE}]}
    echo "  Create ${TYPE}: ${JOB_CREATE[${TYPE}]}"
done

# ---------------------------------------------------------------------------
# Phase 1c: easy_paraphrase pipeline starts immediately (dataset already exists)
# ---------------------------------------------------------------------------
declare -A JOB_PIPE

export PIPELINE_CONFIG="configs/full_pipeline/cloned/easy_paraphrase_combinedvar_n10_e5_ZH.py"
export PIPELINE_OVERRIDES="alpha=0.9"
JOB_PIPE[easy_paraphrase]=$(sbatch --parsable --export=ALL \
    --job-name=pipe_easy_para_a09 \
    "${SCRIPT_DIR}/job_pipeline_single.sh")
echo "  Pipeline easy_paraphrase (alpha=0.9): ${JOB_PIPE[easy_paraphrase]}"

echo ""

# ---------------------------------------------------------------------------
# Phase 2: Clone pipelines with alpha=0.9 (each depends on its clone creation)
# ---------------------------------------------------------------------------
export PIPELINE_OVERRIDES="alpha=0.9"

for TYPE in "${CLONE_TYPES[@]}"; do
    DEP_ID=${JOB_CREATE[${TYPE}]}
    export PIPELINE_CONFIG="configs/full_pipeline/cloned/${TYPE}_combinedvar_n10_e5_ZH.py"
    JOB_PIPE[${TYPE}]=$(sbatch --parsable --export=ALL \
        --dependency=afterok:${DEP_ID} \
        --job-name="pipe_${TYPE}_a09" \
        "${SCRIPT_DIR}/job_pipeline_single.sh")
    echo "  Pipeline ${TYPE} (alpha=0.9): ${JOB_PIPE[${TYPE}]} (after clone ${DEP_ID})"
done

echo ""

# ---------------------------------------------------------------------------
# Phase 3: Comparators (each depends on base pipeline + respective clone pipeline)
# Uses ls -t to find the most recent parquet matching the alpha~0.9 pattern.
# ---------------------------------------------------------------------------
for TYPE in "${CLONE_TYPES[@]}"; do
    PIPE_DEP=${JOB_PIPE[${TYPE}]}
    CMP_JOB=$(sbatch --parsable \
        --dependency=afterok:${JOB_BASE}:${PIPE_DEP} \
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
    echo "  Compare base vs ${TYPE}: ${CMP_JOB} (after base ${JOB_BASE} + pipe ${PIPE_DEP})"
done

echo ""
echo "=== All 16 jobs submitted ==="
echo "  Phase 1a: base pipeline + easy_paraphrase pipeline (parallel)"
echo "  Phase 1b: 4 clone creations in series (avoids paraphrase cache race condition)"
echo "  Phase 2:  4 clone pipelines (each waits on its clone creation)"
echo "  Phase 3:  5 comparators (each waits on base + its clone pipeline)"
echo "  Monitor: squeue -u \$USER"
