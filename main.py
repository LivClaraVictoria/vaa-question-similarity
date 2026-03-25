from time import time
import pandas as pd
from pathlib import Path
import sys
import time
import argparse

from vqs.config_utils import load_config, apply_overrides
from vqs.similarity_metrics import get_calculator, BaseDistanceCalculator
from vqs.data_loader import load_dataset
from vqs.distance_results import save_results
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.clone_robust_analysis import save_reweighting_results
from vqs.recommendation_engine import RecommendationEngine
from vqs.recommendation_saver import save_recommendation_results


def main(config):

    print(f"DEBUG config.clone_id: {config.clone_id}")
    print(f"DEBUG config.CLEANED_DIR: {config.CLEANED_DIR}")

    # 1. Load data: get questions as list
    print("\nLoading data...")
    dataset = load_dataset(
        config
    )  # returns OG question dataframe, and possibly voters and candidates if no canton filtering is applied

    # 2. Calculate Distance
    print(f"\nInitializing distance metric: {config.dist}...")
    calculator: BaseDistanceCalculator = get_calculator(config)
    print("Calculating distances...")
    results: pd.DataFrame = calculator.calculate_distance(dataset, config)

    # 3. Save Results
    print("\nHandling the results...")
    sorted_results = save_results(
        df=results,
        config=config,
        important_params_list=calculator.important_params_list,
    )  # without ID column

    # 4. Applying Damien's Method
    if config.data_choice != "fake":
        print("\nApplying Clone-Robust Weighting...")
        reweighter = CloneRobustReweighter(config)
        reweighted_results = reweighter.reweight(results)

        # 5. Save and Display Reweighted Results
        print("\nSaving reweighted results...")
        save_reweighting_results(
            df=reweighted_results,
            config=config,
            important_params_list=reweighter.important_params_list,
        )

        print(
            f"DEBUG: config.load_voters={config.load_voters}, config.load_candidates={config.load_candidates}"
        )
        if config.load_voters and config.load_candidates:
            # 6. Calculate old and new recommendations
            print("\nCalculating recommendations and changes...")
            rec_engine = RecommendationEngine(
                config=config, data_map=dataset
            )  # only uses candidates and voters df
            recommendation_df = rec_engine.evaluate_pipeline(
                df_weights=reweighted_results
            )

            # Fixes zombie thread issue with terminal output
            sys.stdout.flush()
            time.sleep(2)

            # 7. Save Recommendation Changes and print summary stats
            print("\nSaving recommendation changes...")
            save_recommendation_results(
                df=recommendation_df,
                config=config,
                important_params_list=rec_engine.important_params_list,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run question similarity analysis")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (e.g., configs/base_constants.py or configs/fake/sbert_config.py)",
    )

    # Overrides (this format: key=value)
    # Ex: python main.py --config configs/base_constants.py dist=SBERT_EUCLIDEAN data_year=2019
    # If you want to test something without creating a whole new config file
    parser.add_argument(
        "overrides",
        nargs="*",  # accepts zero or more arguments after --config
        help="Arguments to override config values, e.g. data_year=2019 dist=SBERT",
    )

    args = parser.parse_args()
    config = load_config(Path(args.config))
    config = apply_overrides(config, args.overrides)

    # Run main with the loaded config
    main(config)
