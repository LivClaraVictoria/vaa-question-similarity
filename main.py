import pandas as pd
from pathlib import Path
import argparse
import importlib.util
import sys
from types import SimpleNamespace
import vqs.similarity_metrics as similarity_metrics
from vqs.data_loader import load_dataset

# from configs.base_constants import *


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


# TODO: data visualization? evaluation methods?
def handle_data(similarities: list[dict], config):
    # TODO: handle paths better (put somewhere else idk)
    "Paths for experiment results:"
    experiment_path = config.RESULTS_DIR / "similarities.csv"

    similarities.sort(key=lambda item: item["Similarity"], reverse=True)

    if config.data_choice.lower() == "fake":
        print("Sorted similarities: ")
        for item in similarities:
            print(item["Qu1"], "\n", item["Qu2"], "\n", item["Similarity"])
            print("\n")
    else:
        df_results = pd.DataFrame(similarities)
        # df_results.sort_values(by="Similarity", ascending=False, inplace=True)
        df_results.to_csv(experiment_path, index=False)


def get_calculator(dist: str) -> similarity_metrics.DistanceCalculator:
    if dist.upper() == "SBERT":
        # could pass specific model
        return similarity_metrics.SBERTCalculator()
    else:
        raise NotImplementedError(f"Distance metric '{dist}' is not implemented.")


def main(config):
    # get questions as list
    print("Loading data...")
    dataset = load_dataset(config)

    print("Initializing distance metric...")
    calculator = get_calculator(config.dist)

    print("Calculating distances...")
    results = calculator.calculate_distance(dataset)

    print("Handling the results...")
    handle_data(similarities=results, config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run question similarity analysis")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (e.g., configs/base_constants.py or configs/fake/sbert_config.py)",
    )
    args = parser.parse_args()

    # Load the configuration
    config = load_config(Path(args.config))

    # Run main with the loaded config
    main(config)
