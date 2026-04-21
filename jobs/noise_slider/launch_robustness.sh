#!/bin/bash
# Launcher: noise-slider robustness sweep.
# Submits one job per question (75-task array) + a dependent collect job.
# Run with: bash jobs/noise_slider/launch_robustness.sh
#
# Env vars (optional):
#   PIPELINE_CONFIG   — base pipeline config (default: E5-INSTRUCT ZH α=0.4)
#   ALPHA             — CRW alpha (default: 0.4)
#   N_SEEDS           — seeds per λ (default: 20)
#   LAMBDAS           — comma-separated λ values (default: plan grid)

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"
SWEEP_DIR="${PROJECT_DIR}/experiment_results/noise_slider/robustness/workers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export PIPELINE_CONFIG="${PIPELINE_CONFIG:-configs/base_pipeline/pipeline_e5_instruct_ZH_a04.py}"
export ALPHA="${ALPHA:-0.4}"
export N_SEEDS="${N_SEEDS:-20}"
export LAMBDAS="${LAMBDAS:-}"
export SWEEP_DIR

# Determine question count directly from the questions parquet.
N_QUESTIONS=$(python -c "
import pandas as pd
df = pd.read_parquet('${PROJECT_DIR}/data/cleaned/df_questions.parquet')
print(len(df[df['ID_question'] < 9_000_000]))
")
MAX_IDX=$((N_QUESTIONS - 1))

echo "=== Noise Slider Robustness Sweep ==="
echo "  Config    : ${PIPELINE_CONFIG}"
echo "  Alpha     : ${ALPHA}"
echo "  Seeds/λ   : ${N_SEEDS}"
echo "  Lambdas   : ${LAMBDAS:-<plan default>}"
echo "  Sweep dir : ${SWEEP_DIR}"
echo "  Questions : ${N_QUESTIONS} (array 0-${MAX_IDX})"

# Step 1: pre-generate paraphrases serially (paraphrase cache has a write race).
PREP_JOB=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/job_robustness_prepare.sh")
echo "  Prepare submitted : job ${PREP_JOB}"

# Step 2: worker array depends on prepare.
SWEEP_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${PREP_JOB} \
    --array=0-${MAX_IDX} "${SCRIPT_DIR}/job_robustness_worker.sh")
echo "  Workers submitted : job array ${SWEEP_JOB} (${N_QUESTIONS} tasks)"

# Step 3: collect depends on all workers.
COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} \
    "${SCRIPT_DIR}/job_robustness_collect.sh")
echo "  Collect submitted : job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo "  Monitor: squeue -u \$USER"
