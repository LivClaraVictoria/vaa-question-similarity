#!/bin/bash
# Repair launcher: re-run failed alpha sweep workers and collect jobs.
# Failures were caused by "Illegal instruction" on arton10/arton11 (old NVIDIA drivers).
# Adds arton[10-11] to the exclude list for these re-runs.
#
# Failed sweeps:
#   1. BGE-M3 x easy_paraphrase:       missing workers 14 (a=1.4), 15 (a=1.5)
#   2. E5-INSTRUCT x negation_easy:    missing worker 3 (a=0.3)
#   3. GTE x negation_easy:            missing workers 1 (a=0.1), 6 (a=0.6)
#
# Run with: bash jobs/launch_exp1_repair.sh

set -o errexit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXCLUDE_NODES="tikgpu10,tikgpu[06-09],arton[10-11]"

echo "=== Experiment 1: Repair failed alpha sweep workers ==="
echo "  Excluding nodes: ${EXCLUDE_NODES}"
echo ""

# --- 1. BGE-M3 x easy_paraphrase: workers 14, 15 ---
export CONFIG_A="configs/full_pipeline/base_data/pipeline_bge_m3_ZH.py"
export CONFIG_B="configs/full_pipeline/cloned/easy_paraphrase_top5impact_n4_bge_m3_ZH.py"
export SWEEP_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/exp1/model_alpha_sweep/top5impact/alpha_sweep_pipeline_bge_m3_ZH_vs_easy_paraphrase_top5impact_n4_bge_m3_ZH/workers_20260228_022310"

W1=$(sbatch --parsable --export=ALL --exclude="${EXCLUDE_NODES}" \
    --array=14,15 --job-name="repair_bge_ep" \
    "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
echo "  BGE-M3 x easy_paraphrase:    workers 14,15 → job ${W1}"

C1=$(sbatch --parsable --export=ALL --dependency=afterok:${W1} \
    --job-name="collect_bge_ep" \
    "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
echo "  BGE-M3 x easy_paraphrase:    collect → job ${C1}"

# --- 2. E5-INSTRUCT x negation_easy: worker 3 ---
export CONFIG_A="configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py"
export CONFIG_B="configs/full_pipeline/cloned/negation_easy_top5impact_n4_e5_instruct_ZH.py"
export SWEEP_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/exp1/model_alpha_sweep/top5impact/alpha_sweep_pipeline_e5_instruct_ZH_vs_negation_easy_top5impact_n4_e5_instruct_ZH/workers_20260228_022310"

W2=$(sbatch --parsable --export=ALL --exclude="${EXCLUDE_NODES}" \
    --array=3 --job-name="repair_e5i_ne" \
    "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
echo "  E5-INSTRUCT x negation_easy: worker 3 → job ${W2}"

C2=$(sbatch --parsable --export=ALL --dependency=afterok:${W2} \
    --job-name="collect_e5i_ne" \
    "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
echo "  E5-INSTRUCT x negation_easy: collect → job ${C2}"

# --- 3. GTE x negation_easy: workers 1, 6 ---
export CONFIG_A="configs/full_pipeline/base_data/pipeline_gte_ZH.py"
export CONFIG_B="configs/full_pipeline/cloned/negation_easy_top5impact_n4_gte_ZH.py"
export SWEEP_DIR="/itet-stor/liweiss/net_scratch/vaa-question-similarity/experiment_results/exp1/model_alpha_sweep/top5impact/alpha_sweep_pipeline_gte_ZH_vs_negation_easy_top5impact_n4_gte_ZH/workers_20260228_022310"

W3=$(sbatch --parsable --export=ALL --exclude="${EXCLUDE_NODES}" \
    --array=1,6 --job-name="repair_gte_ne" \
    "${SCRIPT_DIR}/job_alpha_sweep_worker.sh")
echo "  GTE x negation_easy:          workers 1,6 → job ${W3}"

C3=$(sbatch --parsable --export=ALL --dependency=afterok:${W3} \
    --job-name="collect_gte_ne" \
    "${SCRIPT_DIR}/job_alpha_sweep_collect.sh")
echo "  GTE x negation_easy:          collect → job ${C3}"

echo ""
echo "=== Repair submitted ==="
echo "  5 worker tasks + 3 collect jobs"
echo "  Monitor: squeue -u \$USER"
