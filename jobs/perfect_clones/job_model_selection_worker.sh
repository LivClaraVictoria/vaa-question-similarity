#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%A_%a.out
#SBATCH --error=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%A_%a.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --exclude=tikgpu10,tikgpu[06-09],arton[10-11]

# Alpha sweep worker — processes a single alpha value.
# Launched as a SLURM job array by launch_alpha_sweep.sh.
# Expects: SLURM_ARRAY_TASK_ID, SWEEP_DIR, CONFIG_A, CONFIG_B (via --export).
# Optional: OUTPUT_DIR — if set, passes --output-dir to alpha_sweep_main.py.

ETH_USERNAME=liweiss
PROJECT_NAME="vaa-question-similarity"
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=bachelor-thesis

set -o errexit

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
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}, SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}

# Redirect HuggingFace cache to data-scratch to avoid itet-stor quota issues
export HF_HOME=/usr/itetnas04/data-scratch-01/${ETH_USERNAME}/data/.cache/huggingface

cd ${DIRECTORY}

OUTPUT_DIR_FLAG=""
[[ -n "${OUTPUT_DIR}" ]] && OUTPUT_DIR_FLAG="--output-dir ${OUTPUT_DIR}"

python -u -m experiments.perfect_clones.model_selection \
    --mode worker \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --config_a "${CONFIG_A}" \
    --config_b "${CONFIG_B}" \
    --sweep-dir "${SWEEP_DIR}" \
    ${OUTPUT_DIR_FLAG}

echo "Finished at: $(date)"
exit 0
