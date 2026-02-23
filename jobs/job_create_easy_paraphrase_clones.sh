#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL

# --- LOGGING PATHS ---
# %j is automatically replaced by the specific Job ID number.
# CRITICAL: The directory '.../jobs' MUST exist before running, or the job will fail silently.
#SBATCH --output=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.out
#SBATCH --error=/itet-stor/liweiss/net_scratch/vaa-question-similarity/jobs/out/%j.err

# --- MEMORY (RAM) ---
#SBATCH --mem=32G

# Always keep this at 1 for standard training scripts.
#SBATCH --nodes=1

# Time Limit
#SBATCH --time=04:00:00

# Standard rule: 4 CPUs per 1 GPU.
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --exclude=tikgpu10,tikgpu[06-09]
#CommentSBATCH --nodelist=tikgpu01 # Specify that it should run on this particular node
#CommentSBATCH --account=tik-internal
# request specific GPU models (if available on the cluster)
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

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"


[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

# Execute your code
python -u clone_main.py --config configs/create_clones/easy_paraphrase_combinedvar_n10.py


# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
