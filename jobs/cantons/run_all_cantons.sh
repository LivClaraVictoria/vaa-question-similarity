#!/bin/bash
# Launcher: run the pipeline on every test canton with the parameters chosen on the validation
# canton ZH — ONE alpha, no alpha sweeps, same configs as the ZH runs (selected via VQS_DISTRICT).
#
# Per canton: `pipeline` (distances -> CRW -> recommendations), `rec_distortion` (75 questions x 5
# clone types at the fixed alpha) and `partisan_sweep` (Phase 2 for 6 parties; the question
# selection is transferred from ZH's Phase 1). More experiments: EXPERIMENTS="... approx_partisan".
#
# Efficiency
#   - fixed alpha: 1 CRW run per task instead of 21;
#   - the un-cloned base side (baseline + CRW recommendations) is computed once per canton and read
#     from VQS_REC_CACHE_DIR by every task; cloned-question embeddings are shared across cantons;
#   - 1 CPU per job (the ranking code is single-threaded); memory per job is scaled to the canton's
#     voters x candidates from the peak (sacct MaxRSS) measured on the base canton BE.
# Priority: BE is submitted first and runs alone per experiment; every other canton waits until
# BE's jobs of that experiment are done and is submitted with --nice, so BE always wins when the
# memory quota is exhausted. Euler's per-user cap on queued jobs is respected (MAX_QUEUED).
#
# Run with:  ALPHA=0.4 bash jobs/cantons/run_all_cantons.sh      (ALPHA = the alpha chosen on ZH)
# Options:   SUBMIT_DRY_RUN=1   print the sbatch commands only (runs inline, no waiting)
#            CANTONS="BE GE"    subset of cantons (first one = base canton)
#            EXPERIMENTS="pipeline rec_distortion"   subset of experiments
#            RUN_DIR=<existing _runs dir>   resume: units with a done_<exp>_<canton> marker are skipped
# Manifest, orchestrator log and job ids: experiment_results/cantons/_runs/<ts>/ ; status: status.sh

set -o errexit

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRIPT_DIR="${PROJECT_DIR}/jobs/cantons"
source "${PROJECT_DIR}/jobs/_lib/common.sh"
cd "${PROJECT_DIR}"

if [[ -z "${ALPHA:-}" ]]; then
    echo "ERROR: set ALPHA to the alpha chosen on the validation canton ZH, e.g. ALPHA=0.4 bash $0" >&2
    exit 1
fi

# Multi-seat cantons except ZH (validation), base canton first, then by size. The one-seat
# cantons (AR AI GL NW OW UR) have 1-3 candidates, so ranking metrics are meaningless there.
export CANTONS="${CANTONS:-BE AG LU SG VD VS FR SO BL TG BS GR GE SZ ZG NE TI SH JU}"
export EXPERIMENTS="${EXPERIMENTS:-pipeline rec_distortion partisan_sweep}"
export MAX_QUEUED="${MAX_QUEUED:-950}"
export ALPHA
export VQS_ALPHA="${ALPHA}"
export ALPHAS="${ALPHA}"                                   # read by the rec_distortion workers
export VQS_REC_CACHE_DIR="${VQS_REC_CACHE_DIR:-${SCRATCH:-${PROJECT_DIR}/cache}/vaa-question-similarity/cache/recommendations}"
mkdir -p "${VQS_REC_CACHE_DIR}"

# Voters / candidates per canton (2023 data; for sizing jobs). Memory ~ voters x candidates.
declare -A VOTERS=([BE]=34052 [AG]=19063 [LU]=11897 [SG]=10344 [VD]=9456 [VS]=7810 [FR]=7022 [SO]=6151
                   [BL]=6032 [TG]=5491 [BS]=5431 [GR]=4126 [GE]=3791 [SZ]=3165 [ZG]=3094 [NE]=2406
                   [TI]=2020 [SH]=1196 [JU]=1099)
declare -A CANDS=([BE]=685 [AG]=568 [LU]=329 [SG]=288 [VD]=337 [VS]=199 [FR]=137 [SO]=163
                  [BL]=163 [TG]=187 [BS]=107 [GR]=109 [GE]=217 [SZ]=98 [ZG]=84 [NE]=56
                  [TI]=144 [SH]=36 [JU]=34)
# Queue slots per unit, and the memory (GB, 1 CPU) given to the base canton before any measurement.
declare -A UNIT_SIZE=([pipeline]=1 [rec_distortion]=376 [partisan_sweep]=7)
declare -A BASE_MEM_GB=([pipeline]=32 [rec_distortion]=20 [partisan_sweep]=32)
MIN_MEM_GB=6        # embedding model + data
MARGIN=1.4

if [[ -z "${RUN_DIR:-}" ]]; then
    export RUN_DIR="${PROJECT_DIR}/experiment_results/cantons/_runs/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${RUN_DIR}"
    git rev-parse HEAD > "${RUN_DIR}/git_head.txt"
    git diff HEAD > "${RUN_DIR}/uncommitted.patch"
    echo "alpha=${ALPHA} cantons=${CANTONS} experiments=${EXPERIMENTS}" > "${RUN_DIR}/params.txt"
fi
export RUN_DIR

# From the login node: hand the loop to a tiny orchestrator job and exit.
if [[ -z "${SLURM_JOB_ID:-}" && -z "${SUBMIT_DRY_RUN:-}" ]]; then
    ID=$(submit --job-name=canton_orchestrator --time=72:00:00 --cpus-per-task=1 --mem-per-cpu=1G \
        "${SCRIPT_DIR}/run_all_cantons.sh")
    echo "Orchestrator job ${ID} submitted. Progress: ${RUN_DIR}/orchestrator.log"
    exit 0
fi

LOG="${RUN_DIR}/orchestrator.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

run_experiment() {
    case "$1" in
        pipeline)       PIPELINE_CONFIG="configs/base_pipeline/pipeline_e5_instruct_ZH.py" \
                            submit jobs/_generic/job_pipeline_single.sh ;;
        rec_distortion) bash jobs/perfect_clones/launch_rec_distortion.sh ;;
        partisan_sweep) bash jobs/perfect_clones/launch_partisan_sweep.sh ;;
        approx_partisan) PHASE1_CSV="experiment_results/party_impact/mini_maxi/phase1/pipeline_e5_instruct_ZH_a03/mini_maxi_pipeline_e5_instruct_ZH_a03_0311_1538.csv" \
                            bash jobs/approximate_clones/launch_partisan.sh ;;
        *) echo "Unknown experiment: $1" >&2; return 1 ;;
    esac
}

n_queued() { squeue -u "${USER}" -h -r 2>/dev/null | wc -l; }

# True if none of the jobs submitted by unit $1_$2 are still queued/running.
unit_finished() {
    local ids_file="${RUN_DIR}/jobs_$1_$2.txt"
    [[ -f "${ids_file}" ]] || return 1
    ! squeue -u "${USER}" -h -r -o "%F" 2>/dev/null | sort -u | grep -qxFf "${ids_file}"
}

# Peak RSS (GB) over all jobs of a unit, from sacct; empty if unavailable.
measure_peak_gb() {
    sacct -j "$(paste -sd, "${RUN_DIR}/jobs_$1_$2.txt")" -n -P -o MaxRSS 2>/dev/null | awk '
        { v=$1; u=substr(v,length(v)); n=substr(v,1,length(v)-1)+0
          if (u=="K") n/=1048576; else if (u=="M") n/=1024; else if (u!="G") next
          if (n>m) m=n }
        END { if (m>0) printf "%.1f", m }'
}

# Memory (GB) for canton $2 in experiment $1: BASE_MEM_GB for the base canton, otherwise the base
# canton's measured peak scaled by voters x candidates (never above the base canton's request).
mem_for() {
    local exp="$1" c="$2"
    if [[ "${c}" == "${BASE}" ]]; then echo "${BASE_MEM_GB[${exp}]}"; return; fi
    local peak; peak=$(cat "${RUN_DIR}/peak_${exp}" 2>/dev/null || true)
    [[ -z "${peak}" ]] && { echo "${BASE_MEM_GB[${exp}]}"; return; }
    awk -v p="${peak}" -v m="${MARGIN}" -v vc="$((VOTERS[${c}] * CANDS[${c}]))" \
        -v vb="$((VOTERS[${BASE}] * CANDS[${BASE}]))" -v lo="${MIN_MEM_GB}" -v hi="${BASE_MEM_GB[${exp}]}" \
        'BEGIN { g=p*m*vc/vb; if (g<lo) g=lo; if (g>hi) g=hi; printf "%d", (g==int(g))?g:int(g)+1 }'
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
log "Orchestrator start: alpha=${ALPHA}, ${#PENDING[@]} units, base ${BASE}, max queued ${MAX_QUEUED}"

declare -A ATTEMPTS
while (( ${#PENDING[@]} > 0 )); do
    LAUNCHED=0
    REMAINING=()
    for UNIT in "${PENDING[@]}"; do
        EXP="${UNIT%%:*}"; C="${UNIT##*:}"
        if (( LAUNCHED )) && [[ -z "${SUBMIT_DRY_RUN:-}" ]]; then
            REMAINING+=("${UNIT}"); continue        # one unit per pass, then re-check the queue
        fi
        if [[ "${C}" != "${BASE}" && -z "${SUBMIT_DRY_RUN:-}" ]]; then
            if ! unit_finished "${EXP}" "${BASE}"; then REMAINING+=("${UNIT}"); continue; fi
            if [[ ! -s "${RUN_DIR}/peak_${EXP}" ]]; then
                measure_peak_gb "${EXP}" "${BASE}" > "${RUN_DIR}/peak_${EXP}"
                log "measured ${EXP} peak on ${BASE}: $(cat "${RUN_DIR}/peak_${EXP}") GB"
            fi
        fi
        if [[ -z "${SUBMIT_DRY_RUN:-}" ]] && (( $(n_queued) + ${UNIT_SIZE[${EXP}]} > MAX_QUEUED )); then
            REMAINING+=("${UNIT}"); continue
        fi

        MEM=$(mem_for "${EXP}" "${C}")
        export VQS_DISTRICT="${C}"
        export SUBMIT_RES="--cpus-per-task=1 --mem-per-cpu=${MEM}G"
        [[ "${C}" != "${BASE}" ]] && export SUBMIT_NICE=1000 || unset SUBMIT_NICE
        export SUBMIT_LOG="${RUN_DIR}/jobs_${EXP}_${C}.txt"
        rm -f "${SUBMIT_LOG}"
        if run_experiment "${EXP}" >> "${RUN_DIR}/launch_${EXP}.log" 2>&1; then
            touch "${RUN_DIR}/done_${EXP}_${C}"
            log "launched ${EXP} ${C} (${MEM} GB, 1 CPU): $(cat "${SUBMIT_LOG}" 2>/dev/null | wc -l) submissions"
        else
            # Partial submission (e.g. queue limit hit): cancel exactly the jobs of this unit, retry later.
            [[ -s "${SUBMIT_LOG}" ]] && scancel $(cat "${SUBMIT_LOG}") 2>/dev/null || true
            ATTEMPTS[${UNIT}]=$(( ${ATTEMPTS[${UNIT}]:-0} + 1 ))
            if (( ATTEMPTS[${UNIT}] < 3 )); then
                log "FAILED ${EXP} ${C} (attempt ${ATTEMPTS[${UNIT}]}), cancelled its partial jobs, will retry"
                REMAINING+=("${UNIT}")
            else
                log "GAVE UP ${EXP} ${C} after 3 attempts — see launch_${EXP}.log"
            fi
        fi
        unset VQS_DISTRICT SUBMIT_LOG SUBMIT_RES SUBMIT_NICE
        LAUNCHED=1
    done
    PENDING=("${REMAINING[@]}")
    if (( ${#PENDING[@]} > 0 && ! LAUNCHED )); then
        sleep 120
    fi
done
log "All units launched."
