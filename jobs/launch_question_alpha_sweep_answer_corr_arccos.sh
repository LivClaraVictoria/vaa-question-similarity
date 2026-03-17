#!/bin/bash
# Launcher: per-question alpha sweep with ANSWER-CORRELATION-ARCCOS metric.
#
# Since the correlation metric operates on voter answers (not text), all clone
# types produce identical results (clones always have r=1 or r=-1 with source,
# both → distance=0 under arccos(|r|)). So we only run "identical" clones,
# which avoids needing paraphrase generation.
#
# Run with: bash jobs/launch_question_alpha_sweep_answer_corr_arccos.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity"
SWEEP_DIR="${PROJECT_DIR}/experiment_results/exp1/question_alpha_sweep/answer_corr_arccos_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SWEEP_DIR}"

export PIPELINE_CONFIG="configs/full_pipeline/base_data/pipeline_answer_corr_arccos_ZH.py"
export SWEEP_DIR
export CLONE_TYPE="identical"
export N_CLONES=4

echo "=== Per-Question Alpha Sweep (ANSWER-CORRELATION-ARCCOS) ==="
echo "  Config: ${PIPELINE_CONFIG}"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Clone type: ${CLONE_TYPE} (all types equivalent for correlation metric)"
echo "  N clones: ${N_CLONES}"

# Step 1: Determine question count
cd "${PROJECT_DIR}"
[[ -f /itet-stor/liweiss/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/liweiss/net_scratch/conda/bin/conda shell.bash hook)"
conda activate bachelor-thesis

N_QUESTIONS=$(python -c "
import pandas as pd
df = pd.read_parquet('${PROJECT_DIR}/data/cleaned/df_questions.parquet')
print(len(df[df['ID_question'] < 9_000_000]))
")
MAX_IDX=$((N_QUESTIONS - 1))

echo ""
echo "  Questions: ${N_QUESTIONS} (array 0-${MAX_IDX})"

# Step 2: Submit worker array (single clone type — all types equivalent)
echo ""
echo "--- Submitting worker array ---"
SWEEP_JOB=$(sbatch --parsable --export=ALL --array=0-${MAX_IDX} \
    --job-name="qa_sweep_corr_arccos" \
    "${SCRIPT_DIR}/job_question_alpha_sweep_worker_ct.sh")
echo "  Workers: job array ${SWEEP_JOB} (${N_QUESTIONS} tasks)"

# Step 3: Submit collect job (depends on workers)
echo ""
echo "--- Submitting collect job ---"
COLLECT_JOB=$(sbatch --parsable --export=ALL \
    --dependency=afterok:${SWEEP_JOB} \
    "${SCRIPT_DIR}/job_question_alpha_sweep_collect.sh")
echo "  Collect: job ${COLLECT_JOB} (depends on ${SWEEP_JOB})"

echo ""
echo "  Total workers: ${N_QUESTIONS}"
echo "  Estimated IO ops: ~${N_QUESTIONS} worker writes + ~${N_QUESTIONS} collect reads + 6 outputs = ~$((N_QUESTIONS * 2 + 6))"
echo "  Monitor: squeue -u \$USER"
