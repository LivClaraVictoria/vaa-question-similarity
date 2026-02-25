from abc import ABC, abstractmethod
import pandas as pd


class BaseSelector(ABC):
    @abstractmethod
    def select(
        self,
        df_questions: pd.DataFrame,
        df_candidates: pd.DataFrame,
        df_voters: pd.DataFrame,
    ) -> list[int]:
        """Returns a list of question IDs to clone."""


def build_selector(selector_type: str, params: dict) -> BaseSelector:
    if selector_type == "manual":
        return ManualSelector(**params)
    elif selector_type == "random":
        return RandomSelector(**params)
    elif selector_type == "combined_variance":
        return CombinedVarianceSelector(**params)
    elif selector_type == "high_candidate_variance":
        return HighCandidateVarianceSelector(**params)
    elif selector_type == "high_voter_variance":
        return HighVoterVarianceSelector(**params)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")


class ManualSelector(BaseSelector):
    """Tier 1: just hardcode which question to clone."""

    def __init__(self, q_ids: list[int]):
        self.q_ids = q_ids

    def select(self, df_questions, df_candidates, df_voters) -> list[int]:
        # Validate that all IDs exist
        valid_ids = set(df_questions["ID_question"].tolist())
        for q_id in self.q_ids:
            if q_id not in valid_ids:
                raise ValueError(
                    f"Question ID {q_id} not found in questions dataframe."
                )
        return self.q_ids


class RandomSelector(BaseSelector):
    """Selects n random questions. Useful as a baseline."""

    def __init__(self, n: int, seed: int = 42):
        self.n = n
        self.seed = seed

    def select(self, df_questions, df_candidates, df_voters) -> list[int]:
        return (
            df_questions["ID_question"].sample(self.n, random_state=self.seed).tolist()
        )


class HighCandidateVarianceSelector(BaseSelector):
    """
    Selects the n questions where candidates are most spread out.
    These are the questions that most differentiate candidates,
    and thus have a high potential impact on recommendations.
    """

    def __init__(self, n: int):
        self.n = n

    def select(self, df_questions, df_candidates, df_voters) -> list[int]:
        answer_cols = [c for c in df_candidates.columns if c.startswith("answer_")]
        variances = df_candidates[answer_cols].var()
        top_cols = variances.nlargest(self.n).index.tolist()
        return [int(col.replace("answer_", "")) for col in top_cols]


class CombinedVarianceSelector(BaseSelector):
    """
    Selects the n questions with the highest combined variance across candidate and voter answers.
    """

    def __init__(self, n: int):
        self.n = n

    def select(self, df_questions, df_candidates, df_voters) -> list[int]:
        answer_cols_c = [c for c in df_candidates.columns if c.startswith("answer_")]
        answer_cols_v = [c for c in df_voters.columns if c.startswith("answer_")]

        cand_var = df_candidates[answer_cols_c].var()
        voter_var = df_voters[answer_cols_v].var()

        combined = (cand_var * voter_var).dropna()
        top_cols = combined.nlargest(self.n).index.tolist()
        return [int(col.replace("answer_", "")) for col in top_cols]


class HighVoterVarianceSelector(BaseSelector):
    """
    Selects the n questions with the highest voter answer variance.
    These are the questions voters disagree on most, giving them high
    potential impact on recommendations.
    """

    def __init__(self, n: int):
        self.n = n

    def select(self, df_questions, df_candidates, df_voters) -> list[int]:
        answer_cols = [c for c in df_voters.columns if c.startswith("answer_")]
        variances = df_voters[answer_cols].var()
        top_cols = variances.nlargest(self.n).index.tolist()
        return [int(col.replace("answer_", "")) for col in top_cols]


# Future selectors ideas:
# class PartyBenefitSelector(BaseSelector): ...
# class NeutralAnswerFloodSelector(BaseSelector): ...
