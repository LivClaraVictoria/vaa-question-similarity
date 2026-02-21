import pandas as pd
from clone_pipeline.spec import CloneSpec


def _apply_to_questions(df_questions: pd.DataFrame, spec: CloneSpec) -> pd.DataFrame:
    source_row = df_questions[df_questions["ID_question"] == spec.source_q_id]
    if source_row.empty:
        raise ValueError(f"Question ID {spec.source_q_id} not found in questions df.")

    clone_rows = []
    for clone_id in spec.clone_ids:
        row = source_row.copy()
        row["ID_question"] = clone_id
        clone_rows.append(row)

    return pd.concat([df_questions] + clone_rows, ignore_index=True)


def _apply_to_answers(
    df: pd.DataFrame, spec: CloneSpec, include_weights: bool
) -> pd.DataFrame:
    df = df.copy()
    src_col = f"answer_{spec.source_q_id}"

    if src_col not in df.columns:
        raise ValueError(f"Column '{src_col}' not found in dataframe.")

    src_values = df[src_col]
    if spec.flip_answers:
        src_values = 100 - src_values

    for clone_id in spec.clone_ids:
        df[f"answer_{clone_id}"] = src_values

    if include_weights:
        src_weight_col = f"weight_{spec.source_q_id}"
        if src_weight_col in df.columns:
            for clone_id in spec.clone_ids:
                df[f"weight_{clone_id}"] = df[src_weight_col]
        else:
            print(f"⚠️  No weight col for question {spec.source_q_id}, skipping.")

    return df


def apply_specs(
    specs: list[CloneSpec], dataframes: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """
    Applies a list of CloneSpecs to the three dataframes.
    Returns new modified copies — originals are untouched.
    """
    df_candidates = dataframes["candidates"].copy()
    df_voters = dataframes["voters"].copy()
    df_questions = dataframes["questions"].copy()

    for spec in specs:
        print(
            f"Applying spec: {spec.clone_id_str} "
            f"({'flipping answers' if spec.flip_answers else 'identical answers'})"
        )
        df_questions = _apply_to_questions(df_questions, spec)
        df_candidates = _apply_to_answers(df_candidates, spec, include_weights=False)
        df_voters = _apply_to_answers(df_voters, spec, include_weights=True)

    return {
        "candidates": df_candidates,
        "voters": df_voters,
        "questions": df_questions,
    }
