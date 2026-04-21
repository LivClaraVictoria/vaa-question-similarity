#!/bin/bash
# Launcher: run all embedding models on the fake benchmark, then evaluate.
# Run with: bash jobs/launch_model_benchmark.sh

set -o errexit
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Embedding Model Benchmark (11 models + evaluation) ==="
echo ""

# All fake benchmark configs
CONFIGS=(
    "configs/distance_only/fake/fake_sbert_cosine_sim.py"
    "configs/distance_only/fake/fake_sbert_euclidean.py"
    "configs/distance_only/fake/fake_e5.py"
    "configs/distance_only/fake/fake_e5_asymmetric.py"
    "configs/distance_only/fake/fake_e5_instruct.py"
    "configs/distance_only/fake/fake_e5_asymmetric_instruct.py"
    "configs/distance_only/fake/fake_jina_v3.py"
    "configs/distance_only/fake/fake_bge_m3.py"
    "configs/distance_only/fake/fake_gte.py"
    "configs/distance_only/fake/fake_nomic_v2.py"
    "configs/distance_only/fake/fake_qwen3.py"
)

JOB_IDS=()

for config in "${CONFIGS[@]}"; do
    # Extract short name from config filename (e.g., fake_jina_v3.py -> jina_v3)
    name=$(basename "$config" .py | sed 's/^fake_//')

    export PIPELINE_CONFIG="$config"
    JOB=$(sbatch --parsable --export=ALL --job-name="bench_${name}" "${SCRIPT_DIR}/job_pipeline_single.sh")
    JOB_IDS+=("$JOB")
    printf "  %-30s -> job %s\n" "$name" "$JOB"
done

echo ""
echo "  ${#JOB_IDS[@]} model jobs submitted in parallel."

# Build dependency string: afterok:id1:id2:...
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")

# Submit evaluation job after all models finish
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${DEP_STR} --export=ALL --job-name="bench_eval" "${SCRIPT_DIR}/job_evaluate_benchmark.sh")
echo "  Evaluation job: ${EVAL_JOB} (depends on all model jobs)"

echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Results: experiment_results/model_benchmark/"
