# Cluster profile: ETH Euler (Slurm 25.05).
# Sourced by jobs/_lib/common.sh. Every profile must define the same variables and cluster_setup().
# Euler notes: memory must be requested per CPU (--mem-per-cpu; --mem is rejected), don't pin
# partitions, GPUs are shareholders-only, compute nodes reach the internet only via eth_proxy.

# Python environment (conda/mamba install root + env name).
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniforge3}"
CONDA_ENV="${CONDA_ENV:-bachelor-thesis}"

# Slurm stdout/stderr. $SCRATCH is purged after ~2 weeks — fine for logs, not for results.
LOG_DIR="${LOG_DIR:-${SCRATCH}/vaa-question-similarity/logs}"

# HuggingFace model cache (re-downloadable, so scratch is fine).
export HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"

# Extra sbatch arguments added to every submission (e.g. -A <share>; see `my_share_info`).
SBATCH_EXTRA_ARGS=()

# Runs at the start of every job, before the env is activated.
cluster_setup() {
    module load eth_proxy
}
