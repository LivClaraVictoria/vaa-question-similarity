#!/bin/bash
# Phase 2: cumulative CRW correction for mini vs maxi party visibility analysis.
# Env vars: PIPELINE_CONFIG, TARGET_PARTY, TOP_K, PHASE1_CSV, SELECTION_MODE.
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.out
#SBATCH --error=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.err
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --exclude=tikgpu10,tikgpu[06-09]

# Mini-maxi Phase 2 — CRW correction with multiple distance metrics.
# Runs 5 (metric, alpha) combinations sequentially.
# Launched with --dependency by launch_partisan_full.sh.
# Expects: PIPELINE_CONFIG, TOP_K (via --export).
# Optional: PHASE1_CSV, TARGET_PARTY.

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

PHASE2_ARGS="--mode phase2 --config ${PIPELINE_CONFIG}"
[[ -n "${PHASE1_CSV}" ]] && PHASE2_ARGS="${PHASE2_ARGS} --phase1-csv ${PHASE1_CSV}"
[[ -n "${TOP_K}" ]] && PHASE2_ARGS="${PHASE2_ARGS} --top-k ${TOP_K}"
[[ -n "${TARGET_PARTY}" ]] && PHASE2_ARGS="${PHASE2_ARGS} --target-party ${TARGET_PARTY}"
[[ -n "${SELECTION_MODE}" ]] && PHASE2_ARGS="${PHASE2_ARGS} --selection-mode ${SELECTION_MODE}"

python -u -m experiments.approximate_clones.partisan_distortion ${PHASE2_ARGS}

echo "Finished at: $(date)"
exit 0
