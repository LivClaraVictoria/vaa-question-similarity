import pandas as pd
from pathlib import Path
import vqs.similarity_metrics as similarity_metrics

# --- FILE PATHS --- (possibly in separate config file later)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiment_results"

# --- Pick distance metric --- (possibly change to input from terminal?)
dist: str = "SBERT"

# --- Pick what data to use ---
"""
"fake" | ...
"""

data_choice: str = "fake"


# TODO: clean data (use Dustin's methods), define paths to 2 diff Smartvote datasets
def get_data(data_type):
    if data_type == "fake":
        data_df_path = DATA_DIR / "fake" / "questions.csv"
        df = pd.read_csv(data_df_path)
        questions = df["question"].tolist()
    else:
        data_df_path = DATA_DIR / "raw" / "smart vote data" / "df_Questions_2019.pk1"
        df = pd.read_pickle(data_df_path)
        questions = df["question_en"].tolist()
    return questions


# TODO: data visualization? evaluation methods?
def handle_data(similarities: list[dict]):
    # TODO: handle paths better (put somewhere else idk)
    "Paths for experiment results:"
    experiment_path = RESULTS_DIR / "similarities.csv"

    similarities.sort(key=lambda item: item["Similarity"], reverse=True)

    if data_choice.lower() == "fake":
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


def main():
    # get questions as list
    print("Loading data...")
    questions = get_data(data_choice)

    print("Initializing distance metric...")
    calculator = get_calculator(dist)

    print("Calculating distances...")
    results = calculator.calculate_distance(questions=questions)

    print("Handling the results...")
    handle_data(similarities=results)


if __name__ == "__main__":
    main()
