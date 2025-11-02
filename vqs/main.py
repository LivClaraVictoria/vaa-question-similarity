import pandas as pd
from pathlib import Path
from itertools import combinations
from sentence_transformers import SentenceTransformer
import distance

# PATHS (possibly in separate config file later)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiment_results"

# Choose distance metric
metric: str = "SBERT"
dist: distance.Distance = distance.Distance(metric)

# What data to use
data: str = "fake"


def main():
    if data == "fake":
        data_df_path = DATA_DIR / "fake" / "questions.csv"
        df = pd.read_csv(data_df_path)
        questions = df["question"].tolist()

    else:
        data_df_path = DATA_DIR / "raw" / "smart vote data" / "df_Questions_2019.pk1"
        df = pd.read_pickle(data_df_path)
        questions = df["question_en"].tolist()

    experiment_path = RESULTS_DIR / "similarities.csv"

    # make list of questions

    results = dist.calculate_distance(questions=questions)

    if data == "fake":
        for d in results:
            print(d["Similarity"], d["Qu1"], d["Qu2"])
    else:
        df_results = pd.DataFrame(results)
        df_results.sort_values(by="Similarity", ascending=False, inplace=True)
        df_results.to_csv(experiment_path, index=False)


if __name__ == "__main__":
    main()
