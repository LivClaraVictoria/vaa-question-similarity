"""
Generate the fake benchmark dataset for embedding model evaluation.

Two-step workflow:
  1. Select anchor questions from the 2019 SmartVote dataset (saves to JSON for review)
  2. Generate variants via GPT-4o and assemble the CSV

Usage:
    # Step 1: Select 15 random anchors (review data/fake/benchmark_anchors.json afterward)
    python scripts/generate_fake_benchmark.py --select --n-anchors 15 --seed 42

    # Step 2: Generate variants + write CSV (requires OPENAI_API_KEY)
    python scripts/generate_fake_benchmark.py --generate
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# Pinned model for reproducibility
MODEL = "gpt-4o-2024-08-06"
TEMPERATURE = 0.8

# Variant specification: (category_name, count_per_anchor)
# Same-topic (6): variants that a good model should embed close to the anchor
# Traps (4): variants that share keywords/syntax but differ in topic
VARIANT_SPEC = [
    ("Negation", 1),
    ("Easy Paraphrase", 1),
    ("Easy Paraphrase Negation", 1),
    ("Hard Paraphrase", 2),
    ("Hard Paraphrase Negation", 1),
    ("Syntax Trap", 2),
    ("Keyword Trap", 2),
]


def _build_all_variants_prompt(question_text: str) -> str:
    """Build a single GPT-4o prompt that generates ALL variants for one anchor."""

    return f"""You are helping generate test data for evaluating embedding models on a Swiss political questionnaire. We need variants of a question that test whether models understand TOPIC similarity (not just word overlap).

Original question:
"{question_text}"

Generate ALL of the following variants. Return a JSON object with these exact keys:

1. "negation": Negate the original using the simplest change. Insert "not" or swap a key word for its antonym (e.g. "increase" → "decrease"). Keep everything else identical. Someone who agrees with the original should disagree with your version.

2. "easy_paraphrase": Rephrase using similar but not identical wording and slightly different structure. The meaning must be exactly the same. Some word overlap with the original is fine.

3. "easy_paraphrase_negation": Create a paraphrase (similar wording, slightly different structure) and then negate it. Someone who agrees with the original should disagree. Return only the final negated paraphrase.

4. "hard_paraphrase_1": Rephrase using COMPLETELY DIFFERENT words and sentence structure. The meaning must be identical, but you MUST NOT share significant content words with the original OR with the easy_paraphrase above. Use synonyms, different grammatical forms, or circumlocutions. You may change question to statement or vice versa.

5. "hard_paraphrase_2": Another hard paraphrase, different from hard_paraphrase_1. Again, NO shared content words with the original. Must also differ from hard_paraphrase_1.

6. "hard_paraphrase_negation": Create a hard paraphrase (completely different words) and negate it. Must clearly express the OPPOSITE stance. Not just a rhetorical question — an unambiguous disagreement with the original.

7. "syntax_trap_1": A question about a COMPLETELY DIFFERENT political topic that mimics the syntactic structure of the original. Must be about a genuinely unrelated policy area.

8. "syntax_trap_2": Another syntax trap, different topic from syntax_trap_1. Again, unrelated policy area.

9. "keyword_trap_1": A sentence using keywords from the original but about a COMPLETELY DIFFERENT topic — either a non-political context or an unrelated policy area. Should fool a naive keyword-matcher but not a good topic model.

10. "keyword_trap_2": Another keyword trap, different from keyword_trap_1.

IMPORTANT RULES:
- Every variant must be a single sentence or question.
- Hard paraphrases MUST use different vocabulary from the original AND from the easy paraphrase. If the original says "taxes", use "fiscal levies" or "government revenue". If it says "legalized", use "permitted by law" or "decriminalized".
- All negations must UNAMBIGUOUSLY express the opposite stance. Not a double negative or rhetorical question.
- Traps must be about genuinely different topics — not the same topic with a slight twist.
- Each variant must be distinct from all others.

Return valid JSON with exactly these 10 keys."""


def select_anchors(
    questions_path: Path,
    output_path: Path,
    n_anchors: int,
    seed: int,
) -> None:
    """Randomly select anchor questions from the 2019 dataset and save to JSON."""
    df = pd.read_parquet(questions_path)
    print(f"Loaded {len(df)} questions from {questions_path}")

    sample = df.sample(n=n_anchors, random_state=seed)

    anchors = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        anchors.append({
            "anchor_id": i,
            "source_id": int(row["ID_question"]),
            "question_en": row["question_en"],
            "question_de": row.get("question_de", ""),
            "variants": {},
        })

    data = {
        "metadata": {
            "seed": seed,
            "n_anchors": n_anchors,
            "source": str(questions_path),
            "selected_at": datetime.now().isoformat(),
        },
        "anchors": anchors,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved {n_anchors} anchor questions to {output_path}")
    print("\nSelected questions:")
    for a in anchors:
        print(f"  [{a['anchor_id']}] q{a['source_id']}: {a['question_en'][:80]}...")
    print(f"\nReview the JSON file, then run with --generate to create variants.")


# Maps JSON response keys → category name in CSV
RESPONSE_KEY_MAP = [
    ("negation", "Negation"),
    ("easy_paraphrase", "Easy Paraphrase"),
    ("easy_paraphrase_negation", "Easy Paraphrase Negation"),
    ("hard_paraphrase_1", "Hard Paraphrase"),
    ("hard_paraphrase_2", "Hard Paraphrase"),
    ("hard_paraphrase_negation", "Hard Paraphrase Negation"),
    ("syntax_trap_1", "Syntax Trap"),
    ("syntax_trap_2", "Syntax Trap"),
    ("keyword_trap_1", "Keyword Trap"),
    ("keyword_trap_2", "Keyword Trap"),
]


def _anchor_needs_generation(anchor: dict) -> bool:
    """Check if an anchor is missing any variants."""
    for category, count in VARIANT_SPEC:
        if len(anchor["variants"].get(category, [])) < count:
            return True
    return False


def generate_variants(anchor_path: Path, csv_path: Path, regenerate: bool = False) -> None:
    """Generate variants via GPT-4o for each anchor and write the benchmark CSV."""
    from openai import OpenAI

    data = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchors = data["anchors"]
    client = OpenAI()

    # Find anchors that need generation
    to_generate = []
    for anchor in anchors:
        if regenerate or _anchor_needs_generation(anchor):
            to_generate.append(anchor)

    if not to_generate:
        print("All variants already generated. Assembling CSV...")
    else:
        print(f"Generating variants for {len(to_generate)}/{len(anchors)} anchors via {MODEL}...")

        for i, anchor in enumerate(to_generate, 1):
            q_text = anchor["question_en"]
            prompt = _build_all_variants_prompt(q_text)

            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
            )
            result = json.loads(response.choices[0].message.content)

            # Parse response into variant categories
            anchor["variants"] = {}
            for key, category in RESPONSE_KEY_MAP:
                if key not in result:
                    print(f"  WARNING: missing key '{key}' for anchor {anchor['anchor_id']}")
                    continue
                if category not in anchor["variants"]:
                    anchor["variants"][category] = []
                anchor["variants"][category].append(result[key])

            # Save after each anchor (crash-safe)
            anchor_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  [{i}/{len(to_generate)}] anchor {anchor['anchor_id']}: "
                  f"\"{q_text[:50]}{'...' if len(q_text) > 50 else ''}\"")

        print(f"Done generating. Cache saved to {anchor_path}")

    # Assemble CSV
    rows = []
    row_id = 1
    for anchor in anchors:
        aid = anchor["anchor_id"]
        # Anchor row
        rows.append({
            "id": row_id,
            "anchor_id": aid,
            "category": "ANCHOR",
            "question_EN": anchor["question_en"],
        })
        row_id += 1

        # Variant rows
        for category, count in VARIANT_SPEC:
            variants = anchor["variants"].get(category, [])
            for v in variants[:count]:
                rows.append({
                    "id": row_id,
                    "anchor_id": aid,
                    "category": category,
                    "question_EN": v,
                })
                row_id += 1

    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmark CSV written to {csv_path}")
    print(f"  {len(anchors)} anchors × {sum(c for _, c in VARIANT_SPEC)} variants = "
          f"{len(df)} total rows")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fake benchmark dataset for embedding model evaluation"
    )
    parser.add_argument(
        "--select", action="store_true",
        help="Select anchor questions from the 2019 dataset",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate variants via GPT-4o and assemble CSV",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Force regeneration of all variants (ignores cached ones)",
    )
    parser.add_argument(
        "--n-anchors", type=int, default=15,
        help="Number of anchor questions to select (default: 15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for anchor selection (default: 42)",
    )
    parser.add_argument(
        "--questions-path", type=Path,
        default=Path("data/cleaned/df_questions19.parquet"),
        help="Path to the 2019 questions parquet",
    )
    parser.add_argument(
        "--anchor-path", type=Path,
        default=Path("data/fake/benchmark_anchors.json"),
        help="Path to the anchor questions JSON",
    )
    parser.add_argument(
        "--csv-path", type=Path,
        default=Path("data/fake/fake_questions.csv"),
        help="Output path for the benchmark CSV",
    )
    args = parser.parse_args()

    if not args.select and not args.generate:
        parser.error("Specify --select or --generate (or both)")

    if args.select:
        select_anchors(args.questions_path, args.anchor_path, args.n_anchors, args.seed)

    if args.generate:
        if not args.anchor_path.exists():
            print(f"Anchor file not found: {args.anchor_path}")
            print("Run with --select first to pick anchor questions.")
            return
        generate_variants(args.anchor_path, args.csv_path, args.regenerate)


if __name__ == "__main__":
    main()
