# Cluster profile: generic SLURM cluster.
# Sourced by jobs/_lib/common.sh. Every profile must define the same variables and cluster_setup().
# All values are defaults only; override them in jobs/cluster.local.sh (see cluster.local.example.sh)
# or in the environment of the submitting shell.

# Python environment (conda/mamba install root + env name).
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
CONDA_ENV="${CONDA_ENV:-vaa}"

# Slurm stdout/stderr.
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/jobs/out}"

# HuggingFace model cache.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# Extra sbatch arguments added to every submission (e.g. account, partition, node exclusions).
if ! declare -p SBATCH_EXTRA_ARGS &>/dev/null; then
    SBATCH_EXTRA_ARGS=()
fi

# Runs at the start of every job, before the env is activated (e.g. CLUSTER_SETUP_CMD="module load proxy").
cluster_setup() {
    if [[ -n "${CLUSTER_SETUP_CMD:-}" ]]; then
        eval "${CLUSTER_SETUP_CMD}"
    fi
}
