from itertools import combinations
from sentence_transformers import SentenceTransformer  # type: ignore
from abc import ABC, abstractmethod


class DistanceCalculator(ABC):
    @abstractmethod
    def calculate_distance(self, dataset: dict) -> list[dict]:
        pass


class SBERTCalculator(DistanceCalculator):

    # load model once instead of every time we calcualte distance
    # actually idk if this makes a difference?
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        print(f"SBERT model '{model_name}' loaded.")

    def calculate_distance(self, dataset: dict) -> list[dict]:
        questions_df = dataset["questions"]
        questions_en = questions_df["question_EN"].tolist()
        embeddings = self.model.encode(
            questions_en
        )  # emb = model.encode(texts, normalize_embeddings=True) to normalize
        similarities = self.model.similarity(embeddings, embeddings)
        # see similarity method: model.similarity_fn_name

        results = []
        for i, j in combinations(range(len(questions_en)), 2):
            results.append(
                {
                    "Qu1": questions_en[i],
                    "Qu2": questions_en[j],
                    "Similarity": float(similarities[i][j]),  # Good to cast from tensor
                }
            )
        return results
