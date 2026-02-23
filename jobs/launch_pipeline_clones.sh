#!/bin/bash
# Launcher: run base + 3 clone pipelines in parallel.
# Run with: bash jobs/launch_pipeline_clones.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Pipeline Clones (4 parallel jobs) ==="

export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_e5_ZH.py"
JOB_BASE=$(sbatch --parsable --export=ALL --job-name=pipe_base "${SCRIPT_DIR}/job_pipeline_single.sh")
echo "  Base pipeline: ${JOB_BASE}"

export PIPELINE_CONFIG="configs/full_pipeline/cloned/identical_q32214_n10_e5_ZH.py"
JOB_Q32214=$(sbatch --parsable --export=ALL --job-name=pipe_q32214 "${SCRIPT_DIR}/job_pipeline_single.sh")
echo "  Clone q32214:  ${JOB_Q32214}"

export PIPELINE_CONFIG="configs/full_pipeline/cloned/identical_highcandvar_n10_e5_ZH.py"
JOB_HIGHCAND=$(sbatch --parsable --export=ALL --job-name=pipe_highcand "${SCRIPT_DIR}/job_pipeline_single.sh")
echo "  Clone highcandvar: ${JOB_HIGHCAND}"

export PIPELINE_CONFIG="configs/full_pipeline/cloned/identical_combinedvar_n10_e5_ZH.py"
JOB_COMBINED=$(sbatch --parsable --export=ALL --job-name=pipe_combined "${SCRIPT_DIR}/job_pipeline_single.sh")
echo "  Clone combinedvar: ${JOB_COMBINED}"

echo ""
echo "  All 4 pipeline jobs submitted in parallel."
echo "  Job IDs: base=${JOB_BASE}, q32214=${JOB_Q32214}, highcand=${JOB_HIGHCAND}, combined=${JOB_COMBINED}"
echo "  Monitor: squeue -u \$USER"
