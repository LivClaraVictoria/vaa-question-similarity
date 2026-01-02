from itertools import combinations
from sentence_transformers import SentenceTransformer  # type: ignore
from abc import ABC, abstractmethod
import pandas as pd


class DistanceCalculator(ABC):
    @abstractmethod
    def calculate_distance(self, dataset: dict) -> pd.DataFrame:
        pass


class SBERTCalculator(DistanceCalculator):

    # load model once instead of every time we calculate distance
    # actually idk if this makes a difference?
    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", use_Euclidean: bool = False
    ):
        self.model = SentenceTransformer(model_name)
        self.use_Euclidean = use_Euclidean
        print(f"SBERT model '{model_name}' loaded.")

    def calculate_distance(self, dataset: dict) -> pd.DataFrame:
        questions_df = dataset["questions"]
        questions_en = questions_df["question_EN"].tolist()
        categories = (
            questions_df["category"].tolist()
            if "category" in questions_df.columns
            else None
        )

        embeddings = self.model.encode(
            questions_en, normalize_embeddings=True
        )  # emb = model.encode(texts, normalize_embeddings=True) to normalize

        # if self.use_Euclidean:
        #     self.model.similarity_fn_name = "euclidean"
        # uses negative euclidean, so not ideal for us

        similarities = self.model.similarity(embeddings, embeddings)
        # see similarity method: model.similarity_fn_name

        results = []
        for i, j in combinations(range(len(questions_en)), 2):
            score = float(similarities[i][j])  # cast from tensor

            if self.use_Euclidean:
                # Euclidean Dist = sqrt(2 * (1 - CosineSimilarity))
                # Clamping max(0, ...) prevents crash if float precision makes (1 - 1.0000001) negative
                dist_squared = 2 * (1 - score)
                metric_value = (max(0, dist_squared)) ** 0.5
                metric_name = "Distance"
            else:
                metric_value = score
                metric_name = "Similarity"
            results.append(
                {
                    "Qu1": questions_en[i],
                    "Qu2": questions_en[j],
                    "Cat1": categories[i] if categories else None,
                    "Cat2": categories[j] if categories else None,
                    metric_name: metric_value,
                }
            )
        return pd.DataFrame(results)
