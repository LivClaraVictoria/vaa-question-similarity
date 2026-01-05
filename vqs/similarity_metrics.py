from itertools import combinations
from sentence_transformers import SentenceTransformer  # type: ignore
from abc import ABC, abstractmethod
import pandas as pd


class DistanceCalculator(ABC):
    @abstractmethod
    def calculate_distance(self, dataset: dict) -> pd.DataFrame:
        pass


# SBERT
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
        )  # normalizing is necessary for euclidean distance and doesn't make a difference for cosine

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


# E5
class E5Calculator(DistanceCalculator):
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
    ):
        self.model = SentenceTransformer(model_name)
        print(f"E5 model '{model_name}' loaded.")

    def calculate_distance(self, dataset: dict) -> pd.DataFrame:
        questions_df = dataset["questions"]
        questions_en = questions_df["question_EN"].tolist()
        categories = (
            questions_df["category"].tolist()
            if "category" in questions_df.columns
            else None
        )

        # E5 models expect a "query: " or "passage: " prefix. (Use query passage for text retrieval. Query query for symmetric tasks)
        inputs = [f"query: {q}" for q in questions_en]

        # We normalize embeddings so that Dot Product == Cosine Similarity
        embeddings = self.model.encode(inputs, normalize_embeddings=True)
        similarities = self.model.similarity(embeddings, embeddings)

        results = []
        for i, j in combinations(range(len(questions_en)), 2):
            cosine_score = float(similarities[i][j])

            # Formula: EuclideanDist = sqrt(2 * (1 - CosineSim))
            # Use max(0, ...) to prevent floating point errors (e.g., -0.0000001)
            dist_squared = 2 * (1 - cosine_score)
            euclidean_dist = (max(0, dist_squared)) ** 0.5

            results.append(
                {
                    "Qu1": questions_en[i],
                    "Qu2": questions_en[j],
                    "Cat1": categories[i] if categories else None,
                    "Cat2": categories[j] if categories else None,
                    "Distance": euclidean_dist,
                }
            )

        return pd.DataFrame(results)


# asymmetric E5 (retrieval-style)
# only implemented for fake data so far!
class AsymmetricE5Calculator(DistanceCalculator):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.model = SentenceTransformer(model_name)
        print(f"Asymmetric E5 model '{model_name}' loaded.")

    def calculate_distance(self, dataset: dict) -> pd.DataFrame:
        questions_df = dataset["questions"]

        # 1. Separate the Anchor (Query) from the others (Passages)
        # We assume there is exactly one anchor based on your description
        anchor_row = questions_df[questions_df["category"] == "ANCHOR"]
        passage_rows = questions_df[questions_df["category"] != "ANCHOR"]

        if anchor_row.empty:
            raise ValueError("No question with category 'ANCHOR' found.")

        # Extract text
        anchor_text = anchor_row.iloc[0]["question_EN"]
        passage_texts = passage_rows["question_EN"].tolist()
        passage_cats = (
            passage_rows["category"].tolist()
            if "category" in passage_rows.columns
            else [None] * len(passage_texts)
        )

        # 2. Prepare Inputs with Correct Prefixes
        # The Anchor gets "query: ", the rest get "passage: "
        query_input = [f"query: {anchor_text}"]
        passage_inputs = [f"passage: {p}" for p in passage_texts]

        # 3. Encode separately
        # Normalize to allow the Cosine -> Euclidean shortcut
        query_embedding = self.model.encode(query_input, normalize_embeddings=True)
        passage_embeddings = self.model.encode(
            passage_inputs, normalize_embeddings=True
        )

        # 4. Calculate Similarity (1 vs N)
        # This returns a tensor of shape (1, num_passages)
        similarities = self.model.similarity(query_embedding, passage_embeddings)

        results = []
        # We iterate through the passages to pair them with the single anchor
        for idx, passage_text in enumerate(passage_texts):
            cosine_score = float(similarities[0][idx])

            # 5. Convert Cosine Similarity to Euclidean Distance
            # EuclideanDist = sqrt(2 * (1 - CosineSim))
            dist_squared = 2 * (1 - cosine_score)
            euclidean_dist = (max(0, dist_squared)) ** 0.5

            results.append(
                {
                    "Qu1": anchor_text,
                    "Qu2": passage_text,
                    "Cat1": "ANCHOR",
                    "Cat2": passage_cats[idx],
                    "Distance": euclidean_dist,
                }
            )

        return pd.DataFrame(results)
