#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.out
#SBATCH --error=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.err
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --exclude=tikgpu10,tikgpu[06-09]

# Evaluate all embedding models on the fake benchmark.
# Reads distance CSVs from experiment_results/distance_metric/fake_results/
# Outputs comparison CSV + plot to experiment_results/model_benchmark/

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
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}

# Redirect HuggingFace cache to data-scratch to avoid itet-stor quota issues
export HF_HOME=/usr/itetnas04/data-scratch-01/${ETH_USERNAME}/data/.cache/huggingface

cd ${DIRECTORY}

python -u scripts/evaluate_embedding_models.py

echo "Finished at: $(date)"
exit 0
