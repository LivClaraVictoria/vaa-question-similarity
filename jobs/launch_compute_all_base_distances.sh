#!/bin/bash
# Launch distance computation for all 10 embedding models on base ZH dataset.
# All jobs are independent — run in parallel.

CONFIGS=(
    "configs/full_pipeline/base_data/pipeline_sbert_ZH.py"
    "configs/full_pipeline/base_data/pipeline_e5_ZH.py"
    "configs/full_pipeline/base_data/pipeline_e5_asym_ZH.py"
    "configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py"
    "configs/full_pipeline/base_data/pipeline_e5_asym_instruct_ZH.py"
    "configs/full_pipeline/base_data/pipeline_jina_v3_ZH.py"
    "configs/full_pipeline/base_data/pipeline_bge_m3_ZH.py"
    "configs/full_pipeline/base_data/pipeline_gte_ZH.py"
    "configs/full_pipeline/base_data/pipeline_nomic_v2_ZH.py"
    "configs/full_pipeline/base_data/pipeline_qwen3_ZH.py"
)

JOB_IDS=()
for cfg in "${CONFIGS[@]}"; do
    model=$(echo "$cfg" | sed 's/.*pipeline_//;s/_ZH.py//')
    JID=$(sbatch --parsable --export=ALL,PIPELINE_CONFIG="$cfg" \
        --job-name="dist_${model}" \
        jobs/job_compute_distances.sh)
    echo "Submitted $model -> Job $JID"
    JOB_IDS+=("$JID")
done

echo ""
echo "All ${#JOB_IDS[@]} distance jobs submitted."
echo "Monitor with: squeue -u \$USER -n dist_"
