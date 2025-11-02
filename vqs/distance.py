from itertools import combinations
from sentence_transformers import SentenceTransformer


class Distance:
    def __init__(self, str: str) -> None:
        self.type = str

    def calculate_distance(self, questions: list[str]) -> list:
        if self.type == "SBERT":
            return vanilla_SBERT(questions)
        return [0]


def vanilla_SBERT(qu_list: list[str]) -> list:
    SBERT = SentenceTransformer("all-MiniLM-L6-v2")  # just some basic SBERT encoder

    embeddings = SBERT.encode(qu_list)
    similarities = SBERT.similarity(embeddings, embeddings)
    results = []
    for i, j in combinations(range(len(qu_list)), 2):
        results.append(
            {"Qu1": qu_list[i], "Qu2": qu_list[j], "Similarity": similarities[i][j]}
        )
    return results
