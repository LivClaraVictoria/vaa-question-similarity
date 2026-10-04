# Shared helpers for launchers and job scripts. Source it, don't execute it.
#
#   Launcher:   source "$(dirname "${BASH_SOURCE[0]}")/../_lib/common.sh"
#               activate_env               # only if the launcher runs python itself
#               JOB=$(submit [sbatch args...] job_script.sh)
#   Job script: source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}/jobs/_lib/common.sh"
#               job_preamble
#
# All cluster-specific settings live in jobs/clusters/<name>.sh; the active one is chosen in
# jobs/cluster.conf (or with CLUSTER=<name> in the environment).

# Project root = two levels above this file, wherever the checkout lives.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_DIR

source "${PROJECT_DIR}/jobs/cluster.conf"
if [[ ! -f "${PROJECT_DIR}/jobs/clusters/${CLUSTER}.sh" ]]; then
    echo "Unknown cluster profile '${CLUSTER}' (expected jobs/clusters/${CLUSTER}.sh)" >&2
    return 1 2>/dev/null || exit 1
fi
source "${PROJECT_DIR}/jobs/clusters/${CLUSTER}.sh"
export CLUSTER

# Activate the project's conda environment as configured by the cluster profile.
activate_env() {
    if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
        echo "conda not found at CONDA_ROOT=${CONDA_ROOT} (cluster profile: ${CLUSTER})" >&2
        return 1
    fi
    eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
    conda activate "${CONDA_ENV}"
    # Prefer the env's own C++ runtime over the system one; otherwise e.g. sqlite3 (pulled in by
    # IPython/ipywidgets) fails with "CXXABI not found" once torch has loaded the system libstdc++.
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

# sbatch wrapper: adds log paths, env export and the profile's extra args; prints the job id.
# Set SUBMIT_DRY_RUN=1 to print the sbatch command instead of submitting.
submit() {
    local pattern='%x_%j'
    local arg
    for arg in "$@"; do
        [[ "${arg}" == --array* || "${arg}" == -a ]] && pattern='%x_%A_%a'
    done
    local cmd=(sbatch --parsable --export=ALL
        --output="${LOG_DIR}/${pattern}.out" --error="${LOG_DIR}/${pattern}.err"
        "${SBATCH_EXTRA_ARGS[@]}" "$@")
    if [[ -n "${SUBMIT_DRY_RUN:-}" ]]; then
        echo "${cmd[*]}" >&2
        echo "DRYRUN"
        return 0
    fi
    mkdir -p "${LOG_DIR}"
    "${cmd[@]}"
}

# Common start of every job: fail fast, private temp dir, node info, env, cd to project root.
job_preamble() {
    set -o errexit

    TMPDIR=$(mktemp -d -p "${TMPDIR:-/tmp}")
    if [[ ! -d ${TMPDIR} ]]; then
        echo 'Failed to create temp directory' >&2
        exit 1
    fi
    trap "exit 1" HUP INT TERM
    trap 'rm -rf "${TMPDIR}"' EXIT
    export TMPDIR

    echo "Running on node: $(hostname) (cluster profile: ${CLUSTER})"
    echo "Canton (VQS_DISTRICT): ${VQS_DISTRICT:-unset, configs use their own district}"
    echo "Starting on: $(date)"

    cluster_setup
    activate_env
    cd "${PROJECT_DIR}"
}
