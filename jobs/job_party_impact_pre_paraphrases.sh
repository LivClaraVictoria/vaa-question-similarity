#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.out
#SBATCH --error=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.err
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --exclude=tikgpu10,tikgpu[06-09]

# Pre-generate paraphrases for ALL parties' top-K questions.
# Primes the JSON cache so that parallel Phase 2 jobs only read (no race condition).
# Expects: PIPELINE_CONFIG, PHASE1_CSV, TOP_K (via --export).

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

PRE_ARGS="--mode pre-paraphrases --config ${PIPELINE_CONFIG}"
[[ -n "${PHASE1_CSV}" ]] && PRE_ARGS="${PRE_ARGS} --phase1-csv ${PHASE1_CSV}"
[[ -n "${TOP_K}" ]] && PRE_ARGS="${PRE_ARGS} --top-k ${TOP_K}"

python -u -m party_impact_main ${PRE_ARGS}

echo "Finished at: $(date)"
exit 0
