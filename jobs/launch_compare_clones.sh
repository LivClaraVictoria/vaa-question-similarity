#!/bin/bash
# Launcher: compare 3 clone runs against baseline, all in parallel.
# Requires pipeline results to already exist.
# Run with: bash jobs/launch_compare_clones.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIRECTORY="/itet-stor/liweiss/net_scratch/vaa-question-similarity"

# Helper: find the most recent parquet matching a prefix in a directory.
find_rec() {
    local dir=$1
    local prefix=$2
    ls -t "${dir}/${prefix}"*.parquet 2>/dev/null | head -1
}

RECS_CLEANED="${DIRECTORY}/experiment_results/recommendation_results/cleaned"
RECS_CLONED="${DIRECTORY}/experiment_results/recommendation_results/cloned"

BASE_REC=$(find_rec "${RECS_CLEANED}" "recs_pipeline_e5_ZH")
if [[ -z "${BASE_REC}" ]]; then
    echo "ERROR: Could not find base recommendation parquet in ${RECS_CLEANED}" >&2
    exit 1
fi
echo "Base rec: ${BASE_REC}"

echo "=== Compare Clones (3 parallel jobs) ==="

for CLONE_PREFIX in "recs_identical_q32214_n10_e5_ZH" "recs_identical_highcandvar_n10_e5_ZH" "recs_identical_combinedvar_n10_e5_ZH"; do
    CLONE_REC=$(find_rec "${RECS_CLONED}" "${CLONE_PREFIX}")
    if [[ -z "${CLONE_REC}" ]]; then
        echo "WARNING: Could not find ${CLONE_PREFIX} — skipping" >&2
        continue
    fi
    export REC_A="${BASE_REC}"
    export REC_B="${CLONE_REC}"
    JOB=$(sbatch --parsable --export=ALL --job-name="cmp_${CLONE_PREFIX}" "${SCRIPT_DIR}/job_comparator_single.sh")
    echo "  ${CLONE_PREFIX}: job ${JOB}"
done

echo "  Monitor: squeue -u \$USER"
