#!/bin/bash
# Launcher: submit the behavioral-metric comparison and deployment simulation as two
# independent SLURM jobs (no dependency — they share no state).
#
# Run interactively (plain bash, NOT sbatch) from the project root:
#   bash jobs/behavioral_metric/launch_behavioral.sh
#
# Optional env overrides forwarded to the workers:
#   DEPLOY_CONFIG, DEPLOY_SEEDS, DEPLOY_ALPHAS, COMPARE_CONFIG
set -o errexit

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../_lib/common.sh"
cd "${PROJECT_DIR}"

JOB_CMP=$(submit "${HERE}/job_compare.sh")
JOB_DEP=$(submit "${HERE}/job_deploy.sh")

echo "Submitted:"
echo "  metric comparison : ${JOB_CMP}"
echo "  deployment sim    : ${JOB_DEP}"
