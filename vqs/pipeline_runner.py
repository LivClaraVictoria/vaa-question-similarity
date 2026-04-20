"""
Full pipeline runner: loads data, computes distances, applies CRW, generates recommendations.
Called by the `pipeline` subcommand of main.py. With --distances-only, stops after saving distances
(steps 1–3) without running CRW or recommendations — useful for pre-caching embedding distances.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from vqs.config_utils import load_config, apply_overrides
from vqs.similarity_metrics import get_calculator, BaseDistanceCalculator
from vqs.data_loader import load_dataset
from vqs.distance_results import save_results
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.clone_robust_analysis import save_reweighting_results
from vqs.recommendation_engine import RecommendationEngine
from vqs.recommendation_saver import save_recommendation_results


def run_pipeline(config, distances_only: bool = False):
    """Execute the pipeline steps for the given config object."""

    print(f"DEBUG config.clone_id: {config.clone_id}")
    print(f"DEBUG config.CLEANED_DIR: {config.CLEANED_DIR}")

    # 1. Load data
    print("\nLoading data...")
    dataset = load_dataset(config)

    # 2. Calculate distances
    print(f"\nInitializing distance metric: {config.dist}...")
    calculator: BaseDistanceCalculator = get_calculator(config)
    print("Calculating distances...")
    results: pd.DataFrame = calculator.calculate_distance(dataset, config)

    # 3. Save distance results
    print("\nHandling the results...")
    save_results(
        df=results,
        config=config,
        important_params_list=calculator.important_params_list,
    )

    if distances_only:
        print("\n--distances-only flag set — stopping after distance computation.")
        return

    # 4. Apply Clone-Robust Weighting
    if config.data_choice != "fake":
        print("\nApplying Clone-Robust Weighting...")
        reweighter = CloneRobustReweighter(config)
        reweighted_results = reweighter.reweight(results)

        # 5. Save reweighted results
        print("\nSaving reweighted results...")
        save_reweighting_results(
            df=reweighted_results,
            config=config,
            important_params_list=reweighter.important_params_list,
        )

        print(
            f"DEBUG: config.load_voters={config.load_voters}, "
            f"config.load_candidates={config.load_candidates}"
        )
        if config.load_voters and config.load_candidates:
            # 6. Calculate recommendations
            print("\nCalculating recommendations and changes...")
            rec_engine = RecommendationEngine(config=config, data_map=dataset)
            recommendation_df = rec_engine.evaluate_pipeline(df_weights=reweighted_results)

            sys.stdout.flush()
            time.sleep(2)

            # 7. Save recommendation results
            print("\nSaving recommendation changes...")
            save_recommendation_results(
                df=recommendation_df,
                config=config,
                important_params_list=rec_engine.important_params_list,
            )


def main(argv=None):
    """CLI entry point for the pipeline subcommand."""
    parser = argparse.ArgumentParser(description="Run question similarity pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (e.g., configs/base_pipeline/pipeline_e5_ZH.py)",
    )
    parser.add_argument(
        "--distances-only",
        action="store_true",
        help="Compute and cache distances only; skip CRW and recommendations",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Key=value overrides for config values, e.g. data_year=2019 dist=SBERT",
    )

    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    config = apply_overrides(config, args.overrides)
    run_pipeline(config, distances_only=args.distances_only)
