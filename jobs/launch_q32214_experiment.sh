#!/bin/bash
# Master launcher: Full Q32214 multi-model alpha sweep experiment.
# 1) Creates 6 clone datasets (serial, needs OPENAI_API_KEY)
# 2) Runs 28 alpha sweeps (4 models x 7 clone types, all parallel after clones)
#
# Run with: bash jobs/launch_q32214_experiment.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/exp1/model_alpha_sweep/top5impact"

echo "=========================================="
echo "  Q32214 Multi-Model Alpha Sweep Experiment"
echo "=========================================="
echo ""

# --- Phase 1: Clone creation (serial) ---
echo "--- Phase 1: Clone Dataset Creation (serial) ---"

export CLONE_CONFIG="configs/create_clones/easy_paraphrase_q32214_n10.py"
CLONE1=$(sbatch --parsable --export=ALL --job-name=clone_ep "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  easy_paraphrase:  ${CLONE1}"

export CLONE_CONFIG="configs/create_clones/hard_paraphrase_q32214_n10.py"
CLONE2=$(sbatch --parsable --export=ALL --dependency=afterok:${CLONE1} --job-name=clone_hp "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  hard_paraphrase:  ${CLONE2} (after ${CLONE1})"

export CLONE_CONFIG="configs/create_clones/negation_q32214_n10.py"
CLONE3=$(sbatch --parsable --export=ALL --dependency=afterok:${CLONE2} --job-name=clone_neg "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation:         ${CLONE3} (after ${CLONE2})"

export CLONE_CONFIG="configs/create_clones/negation_easy_q32214_n10.py"
CLONE4=$(sbatch --parsable --export=ALL --dependency=afterok:${CLONE3} --job-name=clone_ne "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation_easy:    ${CLONE4} (after ${CLONE3})"

export CLONE_CONFIG="configs/create_clones/negation_hard_q32214_n10.py"
CLONE5=$(sbatch --parsable --export=ALL --dependency=afterok:${CLONE4} --job-name=clone_nh "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  negation_hard:    ${CLONE5} (after ${CLONE4})"

export CLONE_CONFIG="configs/create_clones/natural_mixed_q32214_n5.py"
CLONE6=$(sbatch --parsable --export=ALL --dependency=afterok:${CLONE5} --job-name=clone_nat "${SCRIPT_DIR}/job_create_clone_single.sh")
echo "  natural_mixed:    ${CLONE6} (after ${CLONE5})"

LAST_CLONE=${CLONE6}
echo ""

# --- Phase 2: Alpha sweeps (all parallel, after clones done) ---
echo "--- Phase 2: Alpha Sweeps (4 models x 7 clone types = 28 sweeps) ---"

MODELS=("e5" "jina_v3" "e5_instruct" "qwen3")
BASE_CONFIGS=(
    "configs/full_pipeline/base_data/pipeline_e5_ZH.py"
    "configs/full_pipeline/base_data/pipeline_jina_v3_ZH.py"
    "configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py"
    "configs/full_pipeline/base_data/pipeline_qwen3_ZH.py"
)
CLONE_TYPES=("identical_q32214_n10" "easy_paraphrase_q32214_n10" "hard_paraphrase_q32214_n10" "negation_q32214_n10" "negation_easy_q32214_n10" "negation_hard_q32214_n10" "natural_mixed_q32214_n5")

SWEEP_COUNT=0
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    export CONFIG_A="${BASE_CONFIGS[$i]}"

    for CLONE in "${CLONE_TYPES[@]}"; do
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        export CONFIG_B="configs/full_pipeline/cloned/${CLONE}_${MODEL}_ZH.py"
        SUBFOLDER="alpha_sweep_pipeline_${MODEL}_ZH_vs_${CLONE}_${MODEL}_ZH"
        export SWEEP_DIR="${RESULTS_BASE}/${SUBFOLDER}/workers_${TIMESTAMP}"
        mkdir -p "${SWEEP_DIR}"

        SWEEP_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${LAST_CLONE} \
            --array=0-20 --job-name="asw_${CLONE:0:8}_${MODEL:0:4}" \
            "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
        COLLECT_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:${SWEEP_JOB} \
            --job-name="asc_${CLONE:0:8}_${MODEL:0:4}" \
            "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
        echo "  ${MODEL} x ${CLONE}: sweep=${SWEEP_JOB}, collect=${COLLECT_JOB}"
        SWEEP_COUNT=$((SWEEP_COUNT + 1))
        sleep 1  # ensure unique timestamps
    done
done

echo ""
echo "=== Experiment submitted ==="
echo "  Clone jobs: ${CLONE1} -> ${CLONE2} -> ${CLONE3} -> ${CLONE4} -> ${CLONE5} -> ${CLONE6}"
echo "  Alpha sweeps: ${SWEEP_COUNT} batches (each: 21 workers + 1 collect)"
echo "  Total SLURM jobs: 6 clones + $((SWEEP_COUNT * 22)) sweep = $((6 + SWEEP_COUNT * 22))"
echo "  Monitor: squeue -u \$USER"
