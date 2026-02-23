"""
LLM-based paraphrase generation for approximate clones.

Generates unique paraphrases for each clone via OpenAI GPT-4o,
with a persistent JSON cache that accumulates across runs.

Setup (one-time):
    pip install openai
    export OPENAI_API_KEY="sk-..."
"""

import json
from pathlib import Path

import pandas as pd

from clone_pipeline.spec import CloneSpec

# Pinned model for reproducibility
MODEL = "gpt-4o-2024-08-06"
TEMPERATURE = 0.8


def _build_prompt(
    question_text: str,
    clone_type: str,
    existing_paraphrases: list[str],
) -> str:
    """Build the prompt for generating a single paraphrase of the given type."""

    type_instructions = {
        "easy_paraphrase": (
            "Rephrase the question using similar but not identical wording "
            "and a slightly different sentence structure. The meaning must be "
            "exactly the same. Some words may overlap with the original."
        ),
        "hard_paraphrase": (
            "Rephrase the question using completely different words and sentence "
            "structure. The meaning must be identical, but no significant content "
            "words should be shared with the original. You may change the grammatical "
            "form (e.g. question to statement or vice versa)."
        ),
        "negation": (
            "Negate the original question using the simplest possible change. "
            "Preferably just insert 'not' or swap a key word for its antonym "
            "(e.g. 'increase' becomes 'decrease', 'strengthen' becomes 'weaken'). "
            "Keep everything else identical to the original. "
            "Someone who agrees with the original should disagree with your version."
        ),
        "negation_easy": (
            "First, create an easy paraphrase of the original (similar wording, "
            "slightly different structure). Then negate that paraphrase so that "
            "someone who agrees with the original would disagree with your version. "
            "Return only the final negated paraphrase."
        ),
        "negation_hard": (
            "First, create a hard paraphrase of the original (completely different "
            "words and structure). Then negate that paraphrase so that someone who "
            "agrees with the original would disagree with your version. "
            "Return only the final negated paraphrase."
        ),
    }

    instruction = type_instructions[clone_type]

    existing_block = ""
    if existing_paraphrases:
        items = "\n".join(f"  - \"{p}\"" for p in existing_paraphrases)
        existing_block = (
            f"\n\nThe following paraphrases of this type have already been generated. "
            f"You MUST produce a DIFFERENT one:\n{items}"
        )

    return (
        f"You are helping generate test data for a Swiss political questionnaire.\n\n"
        f"Original question:\n\"{question_text}\"\n\n"
        f"Task: {instruction}"
        f"{existing_block}\n\n"
        f"Return a JSON object with a single key \"paraphrase\" containing your result.\n"
        f"Example: {{\"paraphrase\": \"Your paraphrase here.\"}}"
    )


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _get_question_text(questions_df: pd.DataFrame, q_id: int) -> str:
    """Extract the English question text for a given question ID."""
    row = questions_df[questions_df["ID_question"] == q_id]
    if row.empty:
        raise ValueError(f"Question ID {q_id} not found in questions dataframe.")
    # Handle both 2023 (question_EN) and 2019 (question_en) column names
    for col in ["question_EN", "question_en"]:
        if col in row.columns:
            return row.iloc[0][col]
    raise ValueError(f"No question text column found for question {q_id}.")


def ensure_paraphrases(
    specs: list[CloneSpec],
    questions_df: pd.DataFrame,
    data_year: int,
    paraphrase_dir: Path,
) -> dict:
    """
    Ensure that enough paraphrases exist for all non-identical specs.

    Checks the JSON cache, generates missing paraphrases via LLM,
    and returns the full paraphrases dict.

    Args:
        specs: List of CloneSpecs (only non-identical types trigger generation).
        questions_df: DataFrame with question IDs and text.
        data_year: 2019 or 2023, used for cache filename.
        paraphrase_dir: Directory to store the JSON cache file.

    Returns:
        Dict mapping question ID (str) -> {clone_type -> [paraphrase_1, ...]}.
    """
    cache_path = paraphrase_dir / f"paraphrases_{data_year}.json"
    cache = _load_cache(cache_path)

    # Collect what's needed: (q_id, clone_type, n_needed)
    needs: dict[tuple[int, str], int] = {}
    for spec in specs:
        if spec.clone_type == "identical":
            continue
        key = (spec.source_q_id, spec.clone_type)
        needs[key] = max(needs.get(key, 0), spec.n_clones)

    if not needs:
        return cache

    # Check what's missing
    missing: list[tuple[int, str, int]] = []  # (q_id, clone_type, n_to_generate)
    for (q_id, clone_type), n_needed in needs.items():
        q_id_str = str(q_id)
        existing = cache.get(q_id_str, {}).get(clone_type, [])
        n_existing = len(existing)
        if n_existing < n_needed:
            missing.append((q_id, clone_type, n_needed - n_existing))

    if not missing:
        print(f"All paraphrases already cached in {cache_path}")
        return cache

    # Generate missing paraphrases
    total = sum(n for _, _, n in missing)
    # Lazy import — only needed when actually generating paraphrases
    from openai import OpenAI

    print(f"Generating {total} paraphrases via {MODEL}...")
    client = OpenAI()
    generated_count = 0

    for q_id, clone_type, n_to_generate in missing:
        q_id_str = str(q_id)
        q_text = _get_question_text(questions_df, q_id)

        # Initialize cache structure
        if q_id_str not in cache:
            cache[q_id_str] = {"original": q_text}
        if clone_type not in cache[q_id_str]:
            cache[q_id_str][clone_type] = []

        existing = cache[q_id_str][clone_type]

        for i in range(n_to_generate):
            prompt = _build_prompt(q_text, clone_type, existing)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
            )
            result = json.loads(response.choices[0].message.content)
            paraphrase = result["paraphrase"]
            existing.append(paraphrase)
            generated_count += 1

            # Save after each generation (crash-safe)
            _save_cache(cache, cache_path)
            print(
                f"  [{generated_count}/{total}] q{q_id} {clone_type}: "
                f"\"{paraphrase[:60]}{'...' if len(paraphrase) > 60 else ''}\""
            )

    print(f"Done. Cache saved to {cache_path}")
    return cache
