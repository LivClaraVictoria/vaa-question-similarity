# Cluster profile: D-ITET TIK cluster.
# Sourced by jobs/_lib/common.sh. Every profile must define the same variables and cluster_setup().
# Values reproduce the settings that were previously hardcoded in every job script.

# Python environment (conda/mamba install root + env name).
CONDA_ROOT="${CONDA_ROOT:-/itet-stor/${USER}/net_scratch/conda}"
CONDA_ENV="${CONDA_ENV:-bachelor-thesis}"

# Slurm stdout/stderr.
LOG_DIR="${LOG_DIR:-/itet-stor/${USER}/net_scratch/vaa-question-similarity/jobs/out}"

# HuggingFace cache on data-scratch to avoid itet-stor quota issues.
export HF_HOME="${HF_HOME:-/usr/itetnas04/data-scratch-01/${USER}/data/.cache/huggingface}"

# Extra sbatch arguments added to every submission.
SBATCH_EXTRA_ARGS=(--exclude=tikgpu10,tikgpu[06-09],arton[10-11])

# Runs at the start of every job, before the env is activated.
cluster_setup() {
    :
}
