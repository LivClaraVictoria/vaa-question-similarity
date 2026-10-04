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
# Set SUBMIT_AFTER=<id>[:<id>...] to make every job also wait (afterany) for those jobs, and
# SUBMIT_LOG=<file> to append every submitted job id to that file (used by run_all_cantons.sh).
submit() {
    local pattern='%x_%j'
    local arg has_dep=""
    local args=()
    for arg in "$@"; do
        [[ "${arg}" == --array* || "${arg}" == -a ]] && pattern='%x_%A_%a'
        if [[ -n "${SUBMIT_AFTER:-}" && "${arg}" == --dependency=* ]]; then
            arg="${arg},afterany:${SUBMIT_AFTER}"   # comma = AND with the job's own dependency
            has_dep=1
        fi
        args+=("${arg}")
    done
    if [[ -n "${SUBMIT_AFTER:-}" && -z "${has_dep}" ]]; then
        args=(--dependency="afterany:${SUBMIT_AFTER}" "${args[@]}")
    fi
    local cmd=(sbatch --parsable --export=ALL
        --output="${LOG_DIR}/${pattern}.out" --error="${LOG_DIR}/${pattern}.err"
        "${SBATCH_EXTRA_ARGS[@]}" "${args[@]}")
    if [[ -n "${SUBMIT_DRY_RUN:-}" ]]; then
        echo "${cmd[*]}" >&2
        echo "DRYRUN"
        return 0
    fi
    mkdir -p "${LOG_DIR}"
    local id
    id=$("${cmd[@]}") || return 1
    id="${id%%;*}"
    [[ -n "${SUBMIT_LOG:-}" ]] && echo "${id}" >> "${SUBMIT_LOG}"
    echo "${id}"
}

# Root of the experiment_results tree for the active canton (mirrors vqs.config_utils.canton_results_path):
# the validation canton ZH (or no VQS_DISTRICT) uses experiment_results/, test cantons
# experiment_results/cantons/<code>/. Launchers put their worker dirs under it.
results_root() {
    if [[ -n "${VQS_DISTRICT:-}" && "${VQS_DISTRICT}" != "ZH" ]]; then
        echo "${PROJECT_DIR}/experiment_results/cantons/${VQS_DISTRICT}"
    else
        echo "${PROJECT_DIR}/experiment_results"
    fi
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
