"""
Remove questions from a dataset in-memory.

Inverse of clone_pipeline/applicator.py — drops questions instead of adding them.
"""

import pandas as pd


def get_category_questions(df_questions: pd.DataFrame, category_name: str) -> list[int]:
    """Return sorted question IDs for a category using the _category column."""
    if "_category" not in df_questions.columns:
        raise ValueError("_category column not found in questions DataFrame")

    mask = df_questions["_category"] == category_name
    if mask.sum() == 0:
        available = sorted(df_questions["_category"].unique())
        raise ValueError(
            f"Category '{category_name}' not found. Available: {available}"
        )

    return sorted(df_questions.loc[mask, "ID_question"].tolist())


def remove_questions(
    dataframes: dict[str, pd.DataFrame],
    question_ids_to_remove: list[int],
) -> dict[str, pd.DataFrame]:
    """
    Remove questions from the dataset (questions, voters, candidates).
    Returns new modified copies — originals are untouched.

    Handles SVDataFrame metadata updates (answer_cols, weight_cols)
    following the same pattern as clone_pipeline/applicator.py:70-77.
    """
    df_questions = dataframes["questions"].copy()
    df_voters = dataframes["voters"].copy()
    df_candidates = dataframes["candidates"].copy()

    remove_set = set(question_ids_to_remove)

    # 1. Filter question rows
    df_questions = df_questions[~df_questions["ID_question"].isin(remove_set)].reset_index(drop=True)

    # 2. Build column lists to drop
    answer_cols_to_drop = [f"answer_{qid}" for qid in question_ids_to_remove]
    weight_cols_to_drop = [f"weight_{qid}" for qid in question_ids_to_remove]

    # 3. Drop from candidates (answer columns only, no weights)
    existing_cand_drops = [c for c in answer_cols_to_drop if c in df_candidates.columns]
    df_candidates = df_candidates.drop(columns=existing_cand_drops)

    # 4. Drop from voters (answer + weight columns)
    existing_voter_answer_drops = [c for c in answer_cols_to_drop if c in df_voters.columns]
    existing_voter_weight_drops = [c for c in weight_cols_to_drop if c in df_voters.columns]
    df_voters = df_voters.drop(columns=existing_voter_answer_drops + existing_voter_weight_drops)

    # 5. Update SVDataFrame metadata (critical — prevents stale metadata crashes)
    #    Same pattern as clone_pipeline/applicator.py:74-77, but filtering OUT instead of adding
    drop_answer_set = set(answer_cols_to_drop)
    drop_weight_set = set(weight_cols_to_drop)

    if hasattr(df_candidates, "answer_cols") and df_candidates.answer_cols is not None:
        df_candidates.answer_cols = [c for c in df_candidates.answer_cols if c not in drop_answer_set]

    if hasattr(df_voters, "answer_cols") and df_voters.answer_cols is not None:
        df_voters.answer_cols = [c for c in df_voters.answer_cols if c not in drop_answer_set]
    if hasattr(df_voters, "weight_cols") and df_voters.weight_cols is not None:
        df_voters.weight_cols = [c for c in df_voters.weight_cols if c not in drop_weight_set]

    return {
        "questions": df_questions,
        "voters": df_voters,
        "candidates": df_candidates,
    }
