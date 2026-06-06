"""
Embedding distance calculators for all registered models (SBERT, E5, E5-INSTRUCT, Jina v3,
BGE-M3, GTE, Nomic v2, Qwen3, answer-correlation variants). Each subclass of
BaseDistanceCalculator handles model loading, input formatting, and distance computation.
METRIC_REGISTRY maps config `dist` keys to calculator instances.
"""

from abc import ABC, abstractmethod
from typing import Any
from itertools import combinations
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer  # type: ignore
from vqs.result_management import ResultManager


# --- 1. Base Class ---
class BaseDistanceCalculator(ABC):
    def __init__(
        self,
        config,
        model_name: str,
        instruction: str | None = None,
        is_asymmetric: bool = False,
        use_euclidean: bool = True,
        trust_remote_code: bool = False,
    ):
        self.config = config
        self.model = SentenceTransformer(
            model_name, trust_remote_code=trust_remote_code
        )
        self.instruction = instruction
        self.is_asymmetric = is_asymmetric
        self.use_euclidean = use_euclidean
        self.value_name: str = "Distance" if self.use_euclidean else "Similarity"

        # Parameters that affect the distance calculation and should be included in the cache hash
        # Assumption: all districts have same questionnaire
        self.important_params_list = list(config.DISTANCE_HASH_PARAMS)

        print(
            f"Initialized {model_name} Calculator. Important parameters for caching: {self.important_params_list}"
        )

        mode_str = "Asymmetric" if is_asymmetric else "Symmetric"
        metric_str = "Euclidean" if use_euclidean else "Cosine"

        print(f"[{model_name}] loaded. Mode: {mode_str} | Metric: {metric_str}")

    @abstractmethod
    def format_input(self, text: str, role: str) -> str:
        """role: 'query' or 'passage'"""
        pass

    def _get_encode_kwargs(self, role: str) -> dict:
        """Override in subclasses to pass extra kwargs to model.encode()."""
        return {}

    def _cosine_to_euclidean(self, cosine_score: float) -> float:
        # max(0, ...) to avoid small negative values due to floating point errors
        return (max(0, 2 * (1 - cosine_score))) ** 0.5

    def calculate_distance(self, dataset: dict, config: Any) -> pd.DataFrame:
        questions_df = dataset["questions"]

        if config.data_choice == "fake":
            return self._calculate_anchor_topology(questions_df)
        else:
            return self._calculate_real_topology(questions_df)

    def _calculate_real_topology(self, df: pd.DataFrame) -> pd.DataFrame:

        # 1. Check for cached files
        prefix = f"dist_{self.config.data_year}_{self.config.dist}"
        rm = ResultManager(
            config=self.config,
            dir=self.config.DISTANCE_CACHE_DIR,
            prefix=prefix,
            params_list=self.important_params_list,
        )

        cached_df = rm.load()
        if cached_df is not None:
            return cached_df  # type: ignore

        # 2. If no cache, proceed with calculation
        print("No cache found. Starting distance computation...")

        questions = df.rename(columns=str.lower)[
            "question_en"
        ].tolist()  # 2023: question_EN, 2019: question_en
        question_ids = df["ID_question"].tolist()

        # Encode
        fmt_queries = [self.format_input(q, role="query") for q in questions]
        emb_queries = self.model.encode(
            fmt_queries,
            normalize_embeddings=True,
            **self._get_encode_kwargs("query"),
        )

        if self.is_asymmetric:
            fmt_targets = [self.format_input(q, role="passage") for q in questions]
            emb_targets = self.model.encode(
                fmt_targets,
                normalize_embeddings=True,
                **self._get_encode_kwargs("passage"),
            )
        else:
            emb_targets = emb_queries

        similarities = self.model.similarity(
            emb_queries, emb_targets
        )  # TODO: double check asymmetric case
        results = []

        # Extract with IDs
        if self.is_asymmetric:
            for i in range(len(questions)):
                for j in range(len(questions)):
                    score = float(similarities[i][j])
                    final_value = (
                        self._cosine_to_euclidean(score)
                        if self.use_euclidean
                        else score
                    )
                    results.append(
                        {
                            "Qu1": questions[i],
                            "Qu2": questions[j],
                            "ID1": question_ids[i],
                            "ID2": question_ids[j],
                            self.value_name: final_value,
                            "Type": "Real-Asymmetric",
                        }
                    )
        else:
            # combinations(..., 2) only does unique pairs (triangle of the matrix)
            for i, j in combinations(range(len(questions)), 2):
                # Identical text always gets distance 0, regardless of floating-point
                # rounding in the embedding model (identical strings can produce
                # embeddings that differ by ~1e-7, yielding distances of ~5e-4).
                if questions[i] == questions[j]:
                    final_value = 0.0
                else:
                    score = float(similarities[i][j])
                    final_value = (
                        self._cosine_to_euclidean(score) if self.use_euclidean else score
                    )
                results.append(
                    {
                        "Qu1": questions[i],
                        "Qu2": questions[j],
                        "ID1": question_ids[i],
                        "ID2": question_ids[j],
                        self.value_name: final_value,
                        "Type": "Real-Symmetric",
                    }
                )
        results_df = pd.DataFrame(results)
        rm.save(results_df)
        return results_df

    def _calculate_anchor_topology(self, df: pd.DataFrame) -> pd.DataFrame:
        # Support multi-anchor datasets via anchor_id column
        if "anchor_id" in df.columns:
            return self._calculate_multi_anchor_topology(df)

        # Backward compat: single-anchor dataset (no anchor_id column)
        anchor_row = df[df["category"] == "ANCHOR"]
        passage_rows = df[df["category"] != "ANCHOR"]

        if anchor_row.empty:
            raise ValueError("Fake dataset selected, but no 'ANCHOR' category found.")

        return self._compute_anchor_distances(
            anchor_text=anchor_row.iloc[0]["question_EN"],
            passage_rows=passage_rows,
        )

    def _calculate_multi_anchor_topology(self, df: pd.DataFrame) -> pd.DataFrame:
        all_results = []

        for anchor_id in sorted(df["anchor_id"].unique()):
            group = df[df["anchor_id"] == anchor_id]
            anchor_rows = group[group["category"] == "ANCHOR"]
            passage_rows = group[group["category"] != "ANCHOR"]

            if anchor_rows.empty:
                print(f"Warning: anchor_id={anchor_id} has no ANCHOR row, skipping.")
                continue

            result = self._compute_anchor_distances(
                anchor_text=anchor_rows.iloc[0]["question_EN"],
                passage_rows=passage_rows,
                anchor_id=anchor_id,
            )
            all_results.append(result)

        if not all_results:
            raise ValueError("No valid anchor groups found in fake dataset.")

        return pd.concat(all_results, ignore_index=True)

    def _compute_anchor_distances(
        self,
        anchor_text: str,
        passage_rows: pd.DataFrame,
        anchor_id: int | None = None,
    ) -> pd.DataFrame:
        passage_texts = passage_rows["question_EN"].tolist()
        passage_cats = passage_rows["category"].tolist()

        fmt_anchor = [self.format_input(anchor_text, role="query")]
        target_role = "passage" if self.is_asymmetric else "query"
        fmt_targets = [self.format_input(p, role=target_role) for p in passage_texts]

        emb_anchor = self.model.encode(
            fmt_anchor,
            normalize_embeddings=True,
            **self._get_encode_kwargs("query"),
        )
        emb_targets = self.model.encode(
            fmt_targets,
            normalize_embeddings=True,
            **self._get_encode_kwargs(target_role),
        )

        similarities = self.model.similarity(emb_anchor, emb_targets)

        results = []
        for i, text in enumerate(passage_texts):
            score = float(similarities[0][i])
            final_value = (
                self._cosine_to_euclidean(score) if self.use_euclidean else score
            )
            row = {
                "Qu1": anchor_text,
                "Qu2": text,
                "Cat1": "ANCHOR",
                "Cat2": passage_cats[i],
                self.value_name: final_value,
                "Type": f"Fake-{'Asymmetric' if self.is_asymmetric else 'Symmetric'}",
            }
            if anchor_id is not None:
                row["anchor_id"] = anchor_id
            results.append(row)

        return pd.DataFrame(results)


# --- 2. Subclasses ---


class SBERTCalculator(BaseDistanceCalculator):
    def __init__(self, config: Any, **kwargs):
        model_name = getattr(config, "sbert_model_name", "all-MiniLM-L6-v2")
        # Pass kwargs to override defaults (e.g. use_euclidean=False)
        super().__init__(config, model_name=model_name, **kwargs)

    def format_input(self, text: str, role: str) -> str:
        return text


class E5Calculator(BaseDistanceCalculator):
    def __init__(self, config: Any, **kwargs):
        model_name = getattr(config, "e5_model_name", "intfloat/multilingual-e5-large")
        super().__init__(config, model_name=model_name, **kwargs)

    def format_input(self, text: str, role: str) -> str:
        if role == "query":
            return f"query: {text}"
        return f"passage: {text}"


class E5InstructCalculator(BaseDistanceCalculator):
    def __init__(self, config: Any, **kwargs):
        model_name = getattr(
            config, "e5_instruct_model_name", "intfloat/multilingual-e5-large-instruct"
        )
        instruction = getattr(config, "embedding_instruction", "")
        super().__init__(
            config,
            model_name=model_name,
            instruction=instruction,
            **kwargs,
        )

    def format_input(self, text: str, role: str) -> str:
        if role == "query":
            return f"Instruction: {self.instruction}\nQuery: {text}"
        return text


class JinaV3Calculator(BaseDistanceCalculator):
    def __init__(self, config: Any, **kwargs):
        model_name = getattr(config, "jina_model_name", "jinaai/jina-embeddings-v3")
        self.task = getattr(config, "embedding_task", "separation")
        super().__init__(
            config, model_name=model_name, trust_remote_code=True, **kwargs
        )

    def format_input(self, text: str, role: str) -> str:
        return text

    def _get_encode_kwargs(self, role: str) -> dict:
        return {"task": self.task}


class BGEM3Calculator(BaseDistanceCalculator):
    def __init__(self, config: Any, **kwargs):
        model_name = getattr(config, "bge_model_name", "BAAI/bge-m3")
        super().__init__(config, model_name=model_name, **kwargs)

    def format_input(self, text: str, role: str) -> str:
        return text


class GTECalculator(BaseDistanceCalculator):
    def __init__(self, config: Any, **kwargs):
        model_name = getattr(
            config, "gte_model_name", "Alibaba-NLP/gte-multilingual-base"
        )
        super().__init__(
            config, model_name=model_name, trust_remote_code=True, **kwargs
        )

    def format_input(self, text: str, role: str) -> str:
        return text


class NomicCalculator(BaseDistanceCalculator):
    def __init__(self, config: Any, **kwargs):
        model_name = getattr(
            config, "nomic_model_name", "nomic-ai/nomic-embed-text-v2-moe"
        )
        self.prefix = getattr(config, "embedding_task", "clustering")
        super().__init__(
            config, model_name=model_name, trust_remote_code=True, **kwargs
        )

    def format_input(self, text: str, role: str) -> str:
        return f"{self.prefix}: {text}"


class Qwen3Calculator(BaseDistanceCalculator):
    def __init__(self, config: Any, **kwargs):
        model_name = getattr(
            config, "qwen3_model_name", "Qwen/Qwen3-Embedding-0.6B"
        )
        self.prompt = getattr(config, "embedding_instruction", None)
        super().__init__(config, model_name=model_name, **kwargs)

    def format_input(self, text: str, role: str) -> str:
        return text

    def _get_encode_kwargs(self, role: str) -> dict:
        if self.prompt:
            return {"prompt": self.prompt}
        return {}


# --- 2b. Answer-based distance metrics (no embedding model) ---


class AnswerBasedDistanceCalculator(ABC):
    """Base class for distance metrics computed directly from respondent answer data
    (voters and/or candidates) rather than from question-text embeddings.

    Subclasses implement only `_distance_matrix`, the metric-specific step. All shared
    scaffolding — respondent selection, caching, and result assembly — lives here.

    Requires load_voters=True and/or load_candidates=True. The respondent set is selected
    via `correlation_answer_source` ("voters" | "candidates" | "both").
    """

    def __init__(self, config: Any, **kwargs):
        self.config = config
        self.value_name = "Distance"

        # Answer-based metrics additionally depend on WHICH respondents are used: the
        # source, any voter subset (subset_n), and the out-of-sample voter split. These are
        # deliberately NOT in the global DISTANCE_HASH_PARAMS (that would invalidate every
        # embedding cache, which is text-only), so we append them here. Params missing from
        # the config resolve to None in the hash.
        extra = ["correlation_answer_source", "subset_n", "train_voter_fraction", "split_seed"]
        base = list(config.DISTANCE_HASH_PARAMS)
        self.important_params_list = base + [p for p in extra if p not in base]

        source = getattr(config, "correlation_answer_source", None) or "voters"
        print(
            f"Initialized {type(self).__name__}. Source: {source}. "
            f"Important parameters for caching: {self.important_params_list}"
        )

    @abstractmethod
    def _distance_matrix(
        self, respondents: pd.DataFrame, answer_cols: list[str]
    ) -> np.ndarray:
        """Return the symmetric NxN distance matrix (N = len(answer_cols)), indexed in
        the same order as answer_cols."""
        ...

    def _select_respondents(self, dataset: dict, answer_cols: list[str]) -> pd.DataFrame:
        source = getattr(self.config, "correlation_answer_source", None) or "voters"

        if source in ("voters", "both") and "voters" not in dataset:
            raise ValueError(
                f"correlation_answer_source='{source}' requires load_voters=True in config."
            )
        if source in ("candidates", "both") and "candidates" not in dataset:
            raise ValueError(
                f"correlation_answer_source='{source}' requires load_candidates=True in config."
            )

        if source == "voters":
            return dataset["voters"][answer_cols]
        if source == "candidates":
            return dataset["candidates"][answer_cols]
        return pd.concat(
            [dataset["voters"][answer_cols], dataset["candidates"][answer_cols]],
            ignore_index=True,
        )

    def calculate_distance(self, dataset: dict, config: Any) -> pd.DataFrame:
        if config.data_choice == "fake":
            raise ValueError(
                f"{type(self).__name__} requires real answer data. "
                "Cannot be used with data_choice='fake'."
            )

        # Check cache
        prefix = f"dist_{config.data_year}_{config.dist}"
        rm = ResultManager(
            config=config,
            dir=config.DISTANCE_CACHE_DIR,
            prefix=prefix,
            params_list=self.important_params_list,
        )
        cached_df = rm.load()
        if cached_df is not None:
            return cached_df  # type: ignore

        print(f"No cache found. Computing {config.dist} distances...")

        # Question metadata + answer columns
        questions_df = dataset["questions"]
        question_ids = questions_df["ID_question"].tolist()
        questions_text = questions_df.rename(columns=str.lower)["question_en"].tolist()
        answer_cols = [f"answer_{qid}" for qid in question_ids]

        respondents = self._select_respondents(dataset, answer_cols)
        mat = self._distance_matrix(respondents, answer_cols)

        results = [
            {
                "Qu1": questions_text[i],
                "Qu2": questions_text[j],
                "ID1": question_ids[i],
                "ID2": question_ids[j],
                "Distance": float(mat[i, j]),
                "Type": "Real-Symmetric",
            }
            for i, j in combinations(range(len(question_ids)), 2)
        ]

        results_df = pd.DataFrame(results)
        rm.save(results_df)
        return results_df


class CorrelationDistanceCalculator(AnswerBasedDistanceCalculator):
    """distance(i,j) = 1 - |Pearson_r(answers_i, answers_j)|.

    NOTE: NOT a proper metric (violates triangle inequality). Use the ArcCos variant
    for a proper metric.
    """

    def _correlation_to_distance(self, r: float) -> float:
        """Convert Pearson r to distance. Override in subclasses for different transforms."""
        return 1.0 - abs(r) if not np.isnan(r) else 1.0

    def _distance_matrix(
        self, respondents: pd.DataFrame, answer_cols: list[str]
    ) -> np.ndarray:
        # pandas .corr() uses pairwise deletion for NaNs; column order == answer_cols order
        corr = respondents.corr().to_numpy()
        n = len(answer_cols)
        mat = np.zeros((n, n))
        for i, j in combinations(range(n), 2):
            d = self._correlation_to_distance(corr[i, j])
            mat[i, j] = mat[j, i] = d
        return mat


class ArcCosCorrelationDistanceCalculator(CorrelationDistanceCalculator):
    """distance(i,j) = arccos(|Pearson_r|). Proper metric (angular distance in projective
    space), range [0, pi/2], numerically stable via np.clip."""

    def _correlation_to_distance(self, r: float) -> float:
        if np.isnan(r):
            return float(np.pi / 2)
        return float(np.arccos(np.clip(abs(r), 0.0, 1.0)))


class BehavioralL1DistanceCalculator(AnswerBasedDistanceCalculator):
    """Behavioral L1 (taxicab) distance between answer vectors, up to negation:

        d(q, q') = min( mean|x_q - x_q'|, mean|x_q - (100 - x_q')| ) / 100

    averaged over respondents who answered BOTH questions (pairwise deletion). The min
    over negation (sigma(x) = 100 - x, the SmartVote answer flip) makes a question and its
    negation close — the L1 analogue of using |r| instead of r. Normalized to [0, 1] so
    alpha reads as a population-agreement threshold. It is the squared-error-optimal
    predictor of answer gaps, and a proper pseudometric (l1 distance; negation is an
    isometry, so the min is a clean quotient metric).
    """

    MIN_OVERLAP = 30  # pairs with fewer co-respondents are treated as non-redundant (d=1)

    def _distance_matrix(
        self, respondents: pd.DataFrame, answer_cols: list[str]
    ) -> np.ndarray:
        A = respondents[answer_cols].to_numpy(dtype=float)
        n = len(answer_cols)
        mat = np.zeros((n, n))
        for i, j in combinations(range(n), 2):
            a, b = A[:, i], A[:, j]
            mask = ~np.isnan(a) & ~np.isnan(b)
            if int(mask.sum()) < self.MIN_OVERLAP:
                d = 1.0
            else:
                ai, bj = a[mask], b[mask]
                d_id = np.abs(ai - bj).mean()
                d_neg = np.abs(ai - (100.0 - bj)).mean()
                d = float(min(d_id, d_neg) / 100.0)
            mat[i, j] = mat[j, i] = d
        return mat


# --- 3. The Registry (Configuration) ---

METRIC_REGISTRY = {
    # Deviates from default (Euclidean=False)
    "SBERT": {"class": SBERTCalculator, "kwargs": {"use_euclidean": False}},
    "SBERT_EUCLIDEAN": {"class": SBERTCalculator, "kwargs": {}},
    "E5": {"class": E5Calculator, "kwargs": {}},
    "E5-ASYMMETRIC": {"class": E5Calculator, "kwargs": {"is_asymmetric": True}},
    "E5-INSTRUCT": {"class": E5InstructCalculator, "kwargs": {}},
    "E5-ASYMMETRIC-INSTRUCT": {
        "class": E5InstructCalculator,
        "kwargs": {"is_asymmetric": True},
    },
    "JINA-V3": {"class": JinaV3Calculator, "kwargs": {}},
    "BGE-M3": {"class": BGEM3Calculator, "kwargs": {}},
    "GTE": {"class": GTECalculator, "kwargs": {}},
    "NOMIC-V2": {"class": NomicCalculator, "kwargs": {}},
    "QWEN3": {"class": Qwen3Calculator, "kwargs": {}},
    "ANSWER-CORRELATION": {"class": CorrelationDistanceCalculator, "kwargs": {}},
    "ANSWER-CORRELATION-ARCCOS": {"class": ArcCosCorrelationDistanceCalculator, "kwargs": {}},
    "BEHAVIORAL-L1": {"class": BehavioralL1DistanceCalculator, "kwargs": {}},
}


# --- 4. The Factory ---


def get_calculator(config: Any) -> BaseDistanceCalculator:
    dist_method = config.dist.upper()

    if dist_method not in METRIC_REGISTRY:
        available = ", ".join(METRIC_REGISTRY.keys())
        raise NotImplementedError(
            f"Metric '{config.dist}' is not found in registry. Options: {available}"
        )

    entry = METRIC_REGISTRY[dist_method]
    CalculatorClass = entry["class"]
    specific_args = entry["kwargs"]

    return CalculatorClass(config, **specific_args)
