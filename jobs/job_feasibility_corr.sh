#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.out
#SBATCH --error=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --exclude=tikgpu10,tikgpu[06-09]

# Feasibility check: correlation-driven question addition.
# Single job — only 3 parquet reads, rest is in-memory CPU.
# Run with: sbatch jobs/job_feasibility_corr.sh

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
cd ${DIRECTORY}

python -u scripts/feasibility_corr_addition.py

echo "Finished at: $(date)"
exit 0
