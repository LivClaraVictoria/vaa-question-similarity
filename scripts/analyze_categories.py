"""
Analyze SmartVote question categories: size, mini/full distribution,
and cross-reference with question impact results.

Outputs:
    experiment_results/category_analysis/category_analysis.csv
    experiment_results/category_analysis/category_analysis.txt
    experiment_results/category_analysis/per_question_by_category.csv

Usage:
    python -m scripts.analyze_categories
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/cleaned")
IMPACT_DIR = Path("experiment_results/question_impact_results")
OUTPUT_DIR = Path("experiment_results/category_analysis")


def _find_impact_csv() -> Path | None:
    """Find the most recent question impact CSV."""
    csvs = sorted(IMPACT_DIR.glob("question_impact_*.csv"))
    return csvs[-1] if csvs else None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    questions = pd.read_parquet(DATA_DIR / "df_questions.parquet")
    print(f"Loaded {len(questions)} questions from df_questions.parquet")

    # Check for category columns
    if "_category" not in questions.columns:
        print("ERROR: _category column not found in questions DataFrame", file=sys.stderr)
        sys.exit(1)

    # Load impact results if available
    impact_path = _find_impact_csv()
    if impact_path:
        impact = pd.read_csv(impact_path)
        print(f"Loaded impact results from {impact_path.name} ({len(impact)} questions)")
    else:
        impact = None
        print("No question impact CSV found — impact columns will be N/A")

    # ------------------------------------------------------------------
    # Merge questions with impact
    # ------------------------------------------------------------------
    df = questions[["ID_question", "_category", "category", "rapide"]].copy()

    # Add question text column (handle 2023 vs 2019 naming)
    for col in questions.columns:
        if "question_en" in col.lower():
            df["question_text"] = questions[col]
            break

    if impact is not None:
        impact_cols = ["question_id", "jaccard_mean", "spearman_mean", "kendall_mean",
                       "impact", "composite_rank", "combined_var",
                       "voter_nan_pct", "candidate_nan_pct"]
        impact_subset = impact[[c for c in impact_cols if c in impact.columns]]
        df = df.merge(impact_subset, left_on="ID_question", right_on="question_id", how="left")
        if "question_id" in df.columns:
            df = df.drop(columns=["question_id"])

    # ------------------------------------------------------------------
    # Per-question table (sorted by category, then question ID)
    # ------------------------------------------------------------------
    per_q = df.sort_values(["_category", "ID_question"]).reset_index(drop=True)
    per_q_path = OUTPUT_DIR / "per_question_by_category.csv"
    per_q.to_csv(per_q_path, index=False)
    print(f"  -> {per_q_path}")

    # ------------------------------------------------------------------
    # Category summary
    # ------------------------------------------------------------------
    agg = df.groupby("_category").agg(
        category_id=("category", "first"),
        n_questions=("ID_question", "count"),
        n_mini=("rapide", lambda x: int((x == 1).sum())),
        question_ids=("ID_question", lambda x: sorted(x.tolist())),
    ).reset_index()

    agg["n_full_only"] = agg["n_questions"] - agg["n_mini"]

    if impact is not None and "impact" in df.columns:
        impact_agg = df.groupby("_category").agg(
            mean_impact=("impact", "mean"),
            max_impact=("impact", "max"),
            min_impact=("impact", "min"),
            std_impact=("impact", "std"),
            sum_impact=("impact", "sum"),
            mean_jaccard=("jaccard_mean", "mean"),
            mean_combined_var=("combined_var", "mean"),
        ).reset_index()
        agg = agg.merge(impact_agg, on="_category")

    agg = agg.sort_values("n_questions", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = OUTPUT_DIR / "category_analysis.csv"
    agg.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------
    lines = []
    lines.append("=" * 100)
    lines.append("SMARTVOTE 2023 — CATEGORY ANALYSIS")
    lines.append("=" * 100)
    lines.append(f"Total questions: {len(df)}")
    lines.append(f"Total categories: {agg['_category'].nunique()}")
    lines.append(f"Questions in mini questionnaire: {int((df['rapide'] == 1).sum())}")
    lines.append(f"Questions in full-only: {int((df['rapide'] != 1).sum())}")
    if impact is not None:
        lines.append(f"Impact data source: {impact_path.name}")
    lines.append("")

    # Summary table
    has_impact = impact is not None and "mean_impact" in agg.columns

    header = f"{'Category':<38} {'N':>3} {'Mini':>4} {'Full':>4}"
    if has_impact:
        header += f" | {'Mean Imp':>8} {'Max Imp':>7} {'Sum Imp':>7} {'Std Imp':>7} {'Mean Jac':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in agg.iterrows():
        line = f"{row['_category']:<38} {row['n_questions']:>3} {row['n_mini']:>4} {row['n_full_only']:>4}"
        if has_impact:
            line += (
                f" | {row['mean_impact']:>8.4f} {row['max_impact']:>7.4f} "
                f"{row['sum_impact']:>7.4f} {row['std_impact']:>7.4f} {row['mean_jaccard']:>8.4f}"
            )
        lines.append(line)

    lines.append("")
    lines.append("Legend:")
    lines.append("  N = total questions in category")
    lines.append("  Mini = questions also in the mini (rapide) questionnaire")
    lines.append("  Full = questions only in the full questionnaire")
    if has_impact:
        lines.append("  Mean/Max/Sum/Std Imp = impact (1 - Jaccard) when question is cloned 10x identical")
        lines.append("  Mean Jac = mean Jaccard similarity when question is cloned 10x")

    # Per-category detail
    lines.append("")
    lines.append("=" * 100)
    lines.append("PER-QUESTION DETAIL (categories with 5+ questions)")
    lines.append("=" * 100)

    for _, cat_row in agg[agg["n_questions"] >= 5].iterrows():
        cat = cat_row["_category"]
        cat_qs = df[df["_category"] == cat].sort_values("ID_question")

        lines.append(f"\n--- {cat} ({cat_row['n_questions']} questions, "
                      f"{cat_row['n_mini']} mini, {cat_row['n_full_only']} full-only) ---")

        if has_impact:
            lines.append(f"  Category totals: mean_impact={cat_row['mean_impact']:.4f}, "
                          f"sum_impact={cat_row['sum_impact']:.4f}, std={cat_row['std_impact']:.4f}")
            lines.append("")

        for _, q in cat_qs.iterrows():
            rapide_str = "MINI" if q["rapide"] == 1 else "full"
            text = str(q.get("question_text", ""))[:70]

            detail = f"  Q{int(q['ID_question']):>5}  [{rapide_str:>4}]"
            if has_impact and pd.notna(q.get("impact")):
                detail += f"  impact={q['impact']:.4f}  jaccard={q['jaccard_mean']:.4f}"
            detail += f"  {text}"
            lines.append(detail)

    # Categories with <5 questions (brief)
    small_cats = agg[agg["n_questions"] < 5]
    if len(small_cats) > 0:
        lines.append("")
        lines.append("=" * 100)
        lines.append("SMALLER CATEGORIES (< 5 questions)")
        lines.append("=" * 100)
        for _, cat_row in small_cats.iterrows():
            cat = cat_row["_category"]
            cat_qs = df[df["_category"] == cat].sort_values("ID_question")
            qids = ", ".join(f"Q{int(q)}" for q in cat_qs["ID_question"])
            mini_count = cat_row["n_mini"]
            lines.append(f"  {cat}: {cat_row['n_questions']} questions ({mini_count} mini) — {qids}")

    # Observations for Experiment 2 topic selection
    lines.append("")
    lines.append("=" * 100)
    lines.append("NOTES FOR EXPERIMENT 2 TOPIC SELECTION")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Categories suitable for progressive removal (5+ questions):")
    for _, row in agg[agg["n_questions"] >= 5].iterrows():
        note = f"  {row['_category']:<38} {row['n_questions']} questions ({row['n_questions']-1} removal steps)"
        if row["n_mini"] == 0:
            note += "  [all full-only, none in mini]"
        else:
            note += f"  [{row['n_mini']} mini, {row['n_full_only']} full-only]"
        lines.append(note)

    lines.append("")
    lines.append("=" * 100)

    report = "\n".join(lines)

    txt_path = OUTPUT_DIR / "category_analysis.txt"
    txt_path.write_text(report)
    print(f"  -> {txt_path}")

    # Also print to stdout
    print()
    print(report)


if __name__ == "__main__":
    main()
