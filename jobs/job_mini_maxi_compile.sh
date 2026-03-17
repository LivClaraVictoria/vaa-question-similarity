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

# Mini-maxi compile — aggregate all per-party Phase 2 CSVs into compiled outputs.
# Launched with --dependency by launch_mini_maxi_party_impact.sh.
# Expects: PIPELINE_CONFIG (via --export).

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

COMPILE_ARGS="--mode compile --config ${PIPELINE_CONFIG}"
[[ -n "${SELECTION_MODE}" ]] && COMPILE_ARGS="${COMPILE_ARGS} --selection-mode ${SELECTION_MODE}"

python -u -m mini_maxi_party_impact_main ${COMPILE_ARGS}

echo "Finished at: $(date)"
exit 0
