#!/bin/bash
#SBATCH --mail-type=NONE

# --- LOGGING PATHS ---
#SBATCH --output=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.out
#SBATCH --error=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.err

# --- MEMORY (RAM) ---
#SBATCH --mem=32G

#SBATCH --nodes=1

# Time Limit (alpha sweep: 15 alphas × ~same cost as one pipeline run)
#SBATCH --time=08:00:00

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

# Alpha sweep: base E5 ZH vs identical_q32214_n10 E5 ZH
# Alphas default to 0.1 through 1.5 in 0.1 steps.
# n defaults to 36 (ZH seat count, inferred from config).
python -u -m experiments.perfect_clones.model_selection \
    --config_a configs/full_pipeline/base_data/pipeline_e5_ZH.py \
    --config_b configs/full_pipeline/cloned/identical_combinedvar_n10_e5_ZH.py

echo "Finished at: $(date)"
exit 0