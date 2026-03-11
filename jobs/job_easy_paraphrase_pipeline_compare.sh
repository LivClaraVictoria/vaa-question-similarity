#!/bin/bash
#SBATCH --mail-type=NONE

# --- LOGGING PATHS ---
#SBATCH --output=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.out
#SBATCH --error=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.err

# --- MEMORY (RAM) ---
#SBATCH --mem=32G

#SBATCH --nodes=1

# Time Limit
#SBATCH --time=04:00:00

#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --exclude=tikgpu10,tikgpu[06-09]
#CommentSBATCH --nodelist=tikgpu01
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='titan_rtx|tesla_v100|titan_xp|a100_80gb'


ETH_USERNAME=liweiss
PROJECT_NAME="vaa-question-similarity"
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=bachelor-thesis
mkdir -p ${DIRECTORY}/jobs

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d -p /tmp)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

cd "${TMPDIR}" || exit 1

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

# ── 1. Base pipeline (cleaned data, E5, ZH) ──────────────────────────────────
echo ""
echo "=== Running base pipeline ==="
python -u -m main --config configs/full_pipeline/base_data/pipeline_e5_ZH.py

# ── 2. Cloned pipeline (easy_paraphrase_combinedvar_n10, E5, ZH) ─────────────
echo ""
echo "=== Running cloned pipeline: easy_paraphrase_combinedvar_n10 ==="
python -u -m main --config configs/full_pipeline/cloned/easy_paraphrase_combinedvar_n10_e5_ZH.py

# ── 3. Compare base vs cloned ─────────────────────────────────────────────────
# Helper: find the most recent parquet matching a prefix in a directory.
find_rec() {
    local dir=$1
    local prefix=$2
    ls -t "${dir}/${prefix}"*.parquet 2>/dev/null | head -1
}

RECS_CLEANED="${DIRECTORY}/experiment_results/pipeline_outputs/recommendations/cleaned"
RECS_CLONED="${DIRECTORY}/experiment_results/pipeline_outputs/recommendations/cloned"

BASE_REC=$(find_rec "${RECS_CLEANED}" "recs_pipeline_e5_ZH")
if [[ -z "${BASE_REC}" ]]; then
    echo "ERROR: Could not find base recommendation parquet in ${RECS_CLEANED}" >&2
    exit 1
fi
echo "Base rec parquet: ${BASE_REC}"

CLONE_REC=$(find_rec "${RECS_CLONED}" "recs_easy_paraphrase_combinedvar_n10_e5_ZH")
if [[ -z "${CLONE_REC}" ]]; then
    echo "ERROR: Could not find recommendation parquet for easy_paraphrase_combinedvar_n10" >&2
    exit 1
fi
echo "Clone rec parquet: ${CLONE_REC}"

echo ""
echo "=== Comparing base vs easy_paraphrase_combinedvar_n10 ==="
python -u -m comparator_main "${BASE_REC}" "${CLONE_REC}"

echo ""
echo "Finished at: $(date)"
exit 0
