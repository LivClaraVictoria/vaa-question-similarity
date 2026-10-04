#!/bin/bash
# Launcher: re-run the thesis experiments (exactly as run on the validation canton ZH, same
# configs/parameters) on every test canton, via VQS_DISTRICT.
#
# Euler caps queued jobs per user (~1000, array tasks count individually), far below the ~11k
# jobs of a full run. So this script submits itself as a small long-running orchestrator job that
# launches one (experiment, canton) unit at a time whenever it fits under MAX_QUEUED:
#   - the base canton (BE) goes first, then the comparators by size;
#   - for experiments with large worker arrays, a comparator only starts once BE's jobs of that
#     experiment have left the queue, so the canton-independent embedding caches (cloned-question
#     distances) are computed once, by BE, instead of by every canton at the same time.
# Results: experiment_results/cantons/<code>/<same layout as ZH>. Manifest (code version, cantons,
# job ids, launcher logs, orchestrator log): experiment_results/cantons/_runs/<ts>/.
#
# Run with:  bash jobs/cantons/run_all_cantons.sh
# Options:   SUBMIT_DRY_RUN=1   print the sbatch commands only (runs inline, no waiting)
#            CANTONS="BE GE"    subset of cantons (first one = base canton)
#            EXPERIMENTS="rec_distortion behavioral"   subset of experiments
#            RUN_DIR=<existing _runs dir>   resume: units with a done_<exp>_<canton> marker are skipped

set -o errexit

SCRIPT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/jobs/cantons"
source "${SCRIPT_DIR}/../_lib/common.sh"
cd "${PROJECT_DIR}"

# Multi-seat cantons except ZH (validation), base canton first, then by size. The one-seat
# cantons (AR AI GL NW OW UR) have 1–3 candidates, so ranking metrics are meaningless there.
export CANTONS="${CANTONS:-BE AG LU SG VD VS FR SO BL TG BS GR GE SZ ZG NE TI SH JU}"
export EXPERIMENTS="${EXPERIMENTS:-rec_distortion partisan partisan_sweep approx_partisan approx_rec behavioral noise_slider}"
export MAX_QUEUED="${MAX_QUEUED:-950}"

# Jobs (array tasks) one unit submits, and the experiments gated on the base canton.
declare -A UNIT_SIZE=([rec_distortion]=376 [partisan]=77 [partisan_sweep]=7 [approx_partisan]=53
                      [approx_rec]=1 [behavioral]=2 [noise_slider]=77)
GATED=" rec_distortion partisan partisan_sweep approx_partisan noise_slider "

if [[ -z "${RUN_DIR:-}" ]]; then
    export RUN_DIR="${PROJECT_DIR}/experiment_results/cantons/_runs/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${RUN_DIR}"
    git rev-parse HEAD > "${RUN_DIR}/git_head.txt"
    git diff HEAD > "${RUN_DIR}/uncommitted.patch"
    git status --short > "${RUN_DIR}/git_status.txt"
    echo "${CANTONS}" > "${RUN_DIR}/cantons.txt"
    echo "${EXPERIMENTS}" > "${RUN_DIR}/experiments.txt"
fi
export RUN_DIR

# From the login node: hand the loop to an orchestrator job and exit.
if [[ -z "${SLURM_JOB_ID:-}" && -z "${SUBMIT_DRY_RUN:-}" ]]; then
    ID=$(submit --job-name=canton_orchestrator --time=72:00:00 --cpus-per-task=1 --mem-per-cpu=2G \
        "${SCRIPT_DIR}/run_all_cantons.sh")
    echo "Orchestrator job ${ID} submitted. Progress: ${RUN_DIR}/orchestrator.log"
    exit 0
fi

LOG="${RUN_DIR}/orchestrator.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

run_experiment() {
    case "$1" in
        rec_distortion)  bash jobs/perfect_clones/launch_rec_distortion.sh ;;
        partisan)        bash jobs/perfect_clones/launch_partisan.sh ;;
        partisan_sweep)  bash jobs/perfect_clones/launch_partisan_sweep.sh ;;
        approx_partisan) bash jobs/approximate_clones/launch_partisan_full.sh ;;
        approx_rec)      submit jobs/approximate_clones/job_rec_distortion.sh ;;
        behavioral)      bash jobs/behavioral_metric/launch_behavioral.sh ;;
        noise_slider)    bash jobs/noise_slider/launch_robustness.sh ;;
        *) echo "Unknown experiment: $1" >&2; return 1 ;;
    esac
}

n_queued() { squeue -u "${USER}" -h -r 2>/dev/null | wc -l; }

# True if none of the jobs submitted by unit $1_$2 are still queued/running.
unit_finished() {
    local ids_file="${RUN_DIR}/jobs_$1_$2.txt"
    [[ -f "${ids_file}" ]] || return 1
    ! squeue -u "${USER}" -h -o "%F" 2>/dev/null | sort -u | grep -qxFf "${ids_file}"
}

read -r -a CANTON_LIST <<< "${CANTONS}"
read -r -a EXP_LIST <<< "${EXPERIMENTS}"
BASE="${CANTON_LIST[0]}"

PENDING=()
for C in "${CANTON_LIST[@]}"; do
    for EXP in "${EXP_LIST[@]}"; do
        [[ -f "${RUN_DIR}/done_${EXP}_${C}" ]] || PENDING+=("${EXP}:${C}")
    done
done
log "Orchestrator start: ${#PENDING[@]} units, base ${BASE}, max queued ${MAX_QUEUED}"

declare -A ATTEMPTS
while (( ${#PENDING[@]} > 0 )); do
    LAUNCHED=0
    REMAINING=()
    for UNIT in "${PENDING[@]}"; do
        EXP="${UNIT%%:*}"; C="${UNIT##*:}"
        if (( LAUNCHED )) && [[ -z "${SUBMIT_DRY_RUN:-}" ]]; then
            REMAINING+=("${UNIT}"); continue        # one unit per pass, then re-check the queue
        fi
        if [[ "${C}" != "${BASE}" && "${GATED}" == *" ${EXP} "* && -z "${SUBMIT_DRY_RUN:-}" ]] \
            && ! unit_finished "${EXP}" "${BASE}"; then
            REMAINING+=("${UNIT}"); continue
        fi
        if [[ -z "${SUBMIT_DRY_RUN:-}" ]] && (( $(n_queued) + ${UNIT_SIZE[${EXP}]} > MAX_QUEUED )); then
            REMAINING+=("${UNIT}"); continue
        fi

        export VQS_DISTRICT="${C}"
        export SUBMIT_LOG="${RUN_DIR}/jobs_${EXP}_${C}.txt"
        rm -f "${SUBMIT_LOG}"
        if run_experiment "${EXP}" >> "${RUN_DIR}/launch_${EXP}.log" 2>&1; then
            touch "${RUN_DIR}/done_${EXP}_${C}"
            log "launched ${EXP} ${C}: $(cat "${SUBMIT_LOG}" 2>/dev/null | wc -l) submissions"
        else
            # Partial submission (e.g. queue limit hit): cancel it and retry the whole unit later.
            [[ -f "${SUBMIT_LOG}" ]] && scancel $(cat "${SUBMIT_LOG}") 2>/dev/null || true
            ATTEMPTS[${UNIT}]=$(( ${ATTEMPTS[${UNIT}]:-0} + 1 ))
            if (( ATTEMPTS[${UNIT}] < 3 )); then
                log "FAILED ${EXP} ${C} (attempt ${ATTEMPTS[${UNIT}]}), cancelled partial jobs, will retry"
                REMAINING+=("${UNIT}")
            else
                log "GAVE UP ${EXP} ${C} after 3 attempts — see launch_${EXP}.log"
            fi
        fi
        unset VQS_DISTRICT SUBMIT_LOG
        LAUNCHED=1
    done
    PENDING=("${REMAINING[@]}")
    if (( ${#PENDING[@]} > 0 && ! LAUNCHED )); then
        sleep 120
    fi
done
log "All units launched."
