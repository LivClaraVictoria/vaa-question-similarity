#!/bin/bash
# Worker: per-voter inter-pilot recommendation stability (a few reference alphas, 5 pilots).
# Recomputes CRW recommendations (the expensive step) so it runs on a compute node, not the login.
# Env vars: RECSTAB_CONFIG (config path; default behavioral L1 ZH),
#           RECSTAB_ALPHAS (optional comma list; default 0.13,0.25,0.37),
#           RECSTAB_SEEDS  (optional, e.g. "0,1,2,3,4").
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.out
#SBATCH --error=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --exclude=tikgpu10,tikgpu[06-09]

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

export HF_HOME=/usr/itetnas04/data-scratch-01/${ETH_USERNAME}/data/.cache/huggingface

cd ${DIRECTORY}

ARGS="--config ${RECSTAB_CONFIG:-configs/base_pipeline/pipeline_behavioral_l1_ZH.py}"
[[ -n "${RECSTAB_ALPHAS}" ]] && ARGS="${ARGS} --alphas ${RECSTAB_ALPHAS}"
[[ -n "${RECSTAB_SEEDS}" ]]  && ARGS="${ARGS} --seeds ${RECSTAB_SEEDS}"

echo "Args: ${ARGS}"
python -u -m main behavioral-rec-stability ${ARGS}

echo "Finished at: $(date)"
exit 0
