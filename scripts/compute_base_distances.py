"""
Compute and cache pairwise question distances for a given config.

Only computes distances — no CRW, no recommendations. Used to populate
the cache for all embedding models on the base dataset.

Usage:
    python -m scripts.compute_base_distances --config configs/full_pipeline/base_data/pipeline_sbert_ZH.py
"""

import argparse
from pathlib import Path

from main import load_config
from vqs.data_loader import load_dataset
from vqs.similarity_metrics import get_calculator


def main():
    parser = argparse.ArgumentParser(description="Compute and cache base distances")
    parser.add_argument("--config", required=True, help="Pipeline config path")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    print(f"Model: {config.dist}, District: {config.district}")

    dataset = load_dataset(config)
    calculator = get_calculator(config)
    dist_df = calculator.calculate_distance(dataset, config)

    print(f"Computed {len(dist_df)} pairwise distances")
    print("Done — distances are now cached.")


if __name__ == "__main__":
    main()
