# Machine-specific settings for the job scripts. Copy to jobs/cluster.local.sh (gitignored) and edit.
# Sourced by jobs/cluster.conf before the profile is loaded; every variable is optional.

# Profile to use: jobs/clusters/<name>.sh, or a private one in jobs/clusters/local/<name>.sh.
# CLUSTER="slurm"

# Python environment.
# CONDA_ROOT="${HOME}/miniforge3"
# CONDA_ENV="vaa"

# Slurm log directory and HuggingFace cache.
# LOG_DIR="/path/to/scratch/vaa-question-similarity/logs"
# HF_HOME="/path/to/scratch/.cache/huggingface"

# Extra sbatch arguments for every submission.
# SBATCH_EXTRA_ARGS=(--account=<account> --exclude=<nodes>)

# Command run at the start of every job (e.g. loading a module).
# CLUSTER_SETUP_CMD="module load <module>"

# Data directory if it is not <repo>/data (read by configs/base_constants.py).
# export CLUSTER_DATA_PATH="/path/to/data"
