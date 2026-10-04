#!/bin/bash
# Submit a single job script with the active cluster profile (log paths, extra args).
# Usage: bash jobs/submit.sh [sbatch args...] jobs/_generic/job_pipeline_single.sh
# Example: PIPELINE_CONFIG=configs/base_pipeline/pipeline_e5_ZH.py bash jobs/submit.sh jobs/_generic/job_pipeline_single.sh

set -o errexit

source "$(dirname "${BASH_SOURCE[0]}")/_lib/common.sh"

JOB=$(submit "$@")
echo "Submitted job ${JOB} (cluster profile: ${CLUSTER}, logs: ${LOG_DIR})"
