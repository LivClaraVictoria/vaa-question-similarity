#!/bin/bash
# Status of an all-canton run: job states per (experiment, canton) unit, plus failed/OOM jobs.
# Usage: bash jobs/cantons/status.sh [RUN_DIR]     (default: the latest run)
RUN_DIR="${1:-$(ls -td "$(dirname "${BASH_SOURCE[0]}")"/../../experiment_results/cantons/_runs/*/ | head -1)}"
echo "Run: ${RUN_DIR}"; cat "${RUN_DIR}/params.txt" 2>/dev/null; tail -5 "${RUN_DIR}/orchestrator.log" 2>/dev/null
echo
printf "%-24s %s\n" "unit" "states"
for f in "${RUN_DIR}"/jobs_*.txt; do
    [[ -f "$f" ]] || continue
    unit=$(basename "$f" .txt); unit=${unit#jobs_}
    st=$(sacct -j "$(paste -sd, "$f")" -X -n -P -o State 2>/dev/null | sed 's/ .*//' | sort | uniq -c | awk '{printf "%s=%s ", $2, $1}')
    printf "%-24s %s\n" "${unit}" "${st}"
done
echo; echo "Failed / out-of-memory / timed-out jobs:"
for f in "${RUN_DIR}"/jobs_*.txt; do
    [[ -f "$f" ]] || continue
    sacct -j "$(paste -sd, "$f")" -X -n -P -o JobID,JobName,State,MaxRSS 2>/dev/null | grep -E "FAILED|OUT_OF_MEMORY|TIMEOUT"
done
