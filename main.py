import pandas as pd
from pathlib import Path
import sys
import argparse
import importlib.util
from types import SimpleNamespace

from vqs.similarity_metrics import get_calculator, BaseDistanceCalculator
from vqs.data_loader import load_dataset
from vqs.distance_results import save_results
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.clone_robust_analysis import save_reweighting_results
from vqs.recommendation_engine import RecommendationEngine
from vqs.rec_change_analysis import RecChangeAnalyzer


def load_config(config_path: Path):
    """
    Loads a config.py file by executing it and capturing its variables
    into a namespace object.
    """

    # 1. Read the raw text of the config file
    try:
        config_script = config_path.read_text()
    except FileNotFoundError:
        print(f"Error: Config file not found at: {config_path}")
        sys.exit(1)

    # 2. Create an empty "container" (a dictionary)
    #    This is where the variables from the config file will live.
    config_vars = {"__file__": str(config_path)}

    # 3. Execute the config file's script.
    #    All variables (from the import and the overrides)
    #    are loaded into the 'config_vars' dictionary.
    try:
        exec(config_script, config_vars)
    except Exception as e:
        print(f"Error while loading config file {config_path}:\n{e}")
        sys.exit(1)

    # 4. Convert the dictionary into an object (SimpleNamespace).
    #    This lets you use dot notation (like config.dist)
    #    instead of dictionary notation (like config['dist']).
    config = SimpleNamespace(**config_vars)

    # We remove these two internal Python variables, just to keep it clean
    config_vars.pop("__builtins__", None)
    config_vars.pop("__name__", None)

    return config


def apply_overrides(config, overrides):
    """
    Parses a list of "key=value" strings and updates the config object.
    """
    if not overrides:
        return config

    print(f"\n--- Applying CLI Overrides ---")
    for item in overrides:
        if "=" not in item:
            print(
                f"Warning: Ignoring malformed override '{item}'. Use 'key=value' format."
            )
            continue

        key, value_str = item.split("=", 1)

        # Check if the key exists in the config to avoid typos
        if not hasattr(config, key):
            print(
                f"Warning: New config key '{key}' is being added (was not in config file)."
            )

        # --- Type Inference ---
        # 1. Boolean
        if value_str.lower() == "true":
            val = True
        elif value_str.lower() == "false":
            val = False
        # 2. int, float, or string
        else:
            # Try to convert to int, then float, finally keep as string
            try:
                val = int(value_str)
            except ValueError:
                try:
                    val = float(value_str)
                except ValueError:
                    val = value_str  # it's a string

        # Update the SimpleNamespace config object
        setattr(config, key, val)
        print(f" -> Set '{key}' to: {val} ({type(val).__name__})")

    print("------------------------------\n")
    return config


def main(config):
    # 1. Load data: get questions as list
    print("Loading data...")
    dataset = load_dataset(
        config
    )  # returns OG question dataframe, and possibly voters and candidates if no canton filtering is applied

    # 2. Calculate Distance
    print(f"Initializing distance metric: {config.dist}...")
    calculator: BaseDistanceCalculator = get_calculator(config)
    print("Calculating distances...")
    results: pd.DataFrame = calculator.calculate_distance(dataset, config)

    # 3. Save Results
    print("Handling the results...")
    sorted_results = save_results(
        df=results,
        config=config,
        important_params_list=calculator.important_params_list,
    )  # without ID column

    # 4. Applying Damien's Method
    if config.data_choice != "fake":
        print("Applying Clone-Robust Weighting...")
        reweighter = CloneRobustReweighter(config)
        reweighted_results = reweighter.reweight(results)

        # 5. Save and Display Reweighted Results
        print("Saving reweighted results...")
        save_reweighting_results(
            df=reweighted_results, config=config, method_key=config.crw_paper_choice
        )

        if config.load_voters and config.load_candidates:
            # 6. Calculate old and new recommendations
            print("Calculating recommendations and changes...")
            c
            rec_engine = RecommendationEngine(
                config=config, data_map=dataset
            )  # only uses candidates and voters df
            recommendation_df = rec_engine.evaluate_pipeline(
                df_weights=reweighted_results
            )

            # 7. Analyze and save recommendation changes
            print("Analyzing recommendation changes...")
            analyzer = RecChangeAnalyzer(config)
            # This will check for cache (use crw weights to calculate the hash), calculate if needed, save CSV, and save the plot
            recommendation_df = analyzer.analyze(
                df_recommendations=recommendation_df, df_weights=reweighted_results
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
