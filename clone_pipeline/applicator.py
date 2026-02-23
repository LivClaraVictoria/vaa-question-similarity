import pandas as pd
from clone_pipeline.spec import CloneSpec


def _apply_to_questions(
    df_questions: pd.DataFrame,
    spec: CloneSpec,
    paraphrases: dict | None = None,
) -> pd.DataFrame:
    source_row = df_questions[df_questions["ID_question"] == spec.source_q_id]
    if source_row.empty:
        raise ValueError(f"Question ID {spec.source_q_id} not found in questions df.")

    clone_rows = []
    for idx, clone_id in enumerate(spec.clone_ids):
        row = source_row.copy()
        row["ID_question"] = clone_id

        # Replace question text for non-identical clones
        if spec.clone_type != "identical":
            if paraphrases is None:
                raise ValueError(
                    f"Paraphrases dict required for clone_type '{spec.clone_type}'"
                )
            q_id_str = str(spec.source_q_id)
            texts = paraphrases.get(q_id_str, {}).get(spec.clone_type, [])
            if idx >= len(texts):
                raise ValueError(
                    f"Not enough '{spec.clone_type}' paraphrases for question "
                    f"{spec.source_q_id}: need {len(spec.clone_ids)}, have {len(texts)}"
                )
            new_text = texts[idx]
            for col in ["question_EN", "question_en"]:
                if col in row.columns:
                    row[col] = new_text

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
    specs: list[CloneSpec],
    dataframes: dict[str, pd.DataFrame],
    paraphrases: dict | None = None,
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
        df_questions = _apply_to_questions(df_questions, spec, paraphrases)
        df_candidates = _apply_to_answers(df_candidates, spec, include_weights=False)
        df_voters = _apply_to_answers(df_voters, spec, include_weights=True)

    return {
        "candidates": df_candidates,
        "voters": df_voters,
        "questions": df_questions,
    }
