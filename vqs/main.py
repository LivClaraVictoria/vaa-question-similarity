import pandas as pd
from pathlib import Path
from itertools import combinations
from sentence_transformers import SentenceTransformer

# PATHS (possibly in separate config file later)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "experiment_results"

# 1. Load a pretrained Sentence Transformer model
SBERT = SentenceTransformer("all-MiniLM-L6-v2")  # just some basic SBERT encoder

data_df_path = DATA_DIR / "smart vote data" / "df_Questions_2019.pk1"
experiment_path = RESULTS_DIR / "similarities.csv"

# make list of questions
df = pd.read_pickle(data_df_path)
questions = df["question_en"].tolist()

embeddings = SBERT.encode(questions)

similarities = SBERT.similarity(embeddings, embeddings)
results = []
for i, j in combinations(range(len(questions)), 2):
    results.append(
        {"Qu1": questions[i], "Qu2": questions[j], "Similarity": similarities[i][j]}
    )

df_results = pd.DataFrame(results)
df_results.sort_values(by="Similarity", ascending=False, inplace=True)
df_results.to_csv(experiment_path, index=False)

# # 2. Calculate embeddings by calling model.encode()
# embeddings = SBERT.encode(sentences)
# print(embeddings.shape)
# # [3, 384]

# # 3. Calculate the embedding similarities
# similarities = SBERT.similarity(embeddings, embeddings)
# print(similarities)
