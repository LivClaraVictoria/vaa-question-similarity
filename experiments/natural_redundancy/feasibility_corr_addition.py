"""
Feasibility check: Does adding highly correlated questions to the mini
questionnaire cause MORE recommendation distortion than adding low-correlation
questions?

If not, a correlation-based CRW experiment won't have a meaningful signal.

Usage:
    python scripts/feasibility_corr_addition.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "dependencies" / "rsfp")
)

from vqs.config_utils import load_config
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine
from experiments.natural_redundancy.mini_maxi_party_impact import (
    filter_to_mini,
    add_question_to_mini,
    add_questions_to_mini,
    compute_redundancy_scores,
)


# ---------------------------------------------------------------------------
# Inline Jaccard / rank metrics (no CrossRunAnalyzer needed)
# ---------------------------------------------------------------------------


def compare_recs(recs_a: pd.DataFrame, recs_b: pd.DataFrame, n: int) -> dict:
    """Compare two baseline recommendation DataFrames. Returns aggregate metrics."""
    # Extract ranked lists from matchID columns
    std_cols_a = sorted(
        [c for c in recs_a.columns if c.startswith("_matchID_")],
        key=lambda c: int(c.split("_")[2]),
    )
    std_cols_b = sorted(
        [c for c in recs_b.columns if c.startswith("_matchID_")],
        key=lambda c: int(c.split("_")[2]),
    )

    if recs_a.index.name != "voterID":
        recs_a = recs_a.set_index("voterID")
    if recs_b.index.name != "voterID":
        recs_b = recs_b.set_index("voterID")

    common_voters = recs_a.index.intersection(recs_b.index)
    jaccards = []
    spearmans = []
    kendalls = []

    for vid in common_voters:
        list_a = recs_a.loc[vid, std_cols_a].dropna().tolist()
        list_b = recs_b.loc[vid, std_cols_b].dropna().tolist()

        # Jaccard on top-n
        s1 = set(list_a[:n])
        s2 = set(list_b[:n])
        if s1 or s2:
            jaccards.append(len(s1 & s2) / len(s1 | s2))
        else:
            jaccards.append(1.0)

        # Spearman / Kendall on full list
        common_cands = set(list_a) & set(list_b)
        if len(common_cands) >= 2:
            rank_b = {c: i for i, c in enumerate(list_b)}
            ranks_b = np.array([rank_b[c] for c in list_a if c in rank_b])
            ranks_a = np.arange(len(ranks_b))
            s, _ = spearmanr(ranks_a, ranks_b)
            k, _ = kendalltau(ranks_a, ranks_b)
            spearmans.append(s)
            kendalls.append(k)

    return {
        "jaccard_mean": np.mean(jaccards),
        "jaccard_median": np.median(jaccards),
        "jaccard_p10": np.percentile(jaccards, 10),
        "spearman_mean": np.mean(spearmans),
        "kendall_mean": np.mean(kendalls),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    config_path = Path("configs/full_pipeline/base_data/pipeline_e5_ZH.py")
    config = load_config(config_path)
    N_TOP = 36  # ZH canton seats

    # 1. Load data
    print("=" * 70)
    print("FEASIBILITY CHECK: Correlation-driven question addition")
    print("=" * 70)

    print("\n--- Loading full dataset ---")
    full_dataset = load_dataset(config)

    print("\n--- Filtering to mini questionnaire ---")
    mini_dataset, mini_ids, full_only_ids = filter_to_mini(full_dataset)
    print(f"  Mini: {len(mini_ids)} questions, Full-only: {len(full_only_ids)} questions")

    # 2. Compute redundancy scores
    print("\n--- Computing redundancy scores ---")
    redundancy = compute_redundancy_scores(
        full_dataset["voters"], mini_ids, full_only_ids
    )

    # Build sorted list
    q_info = []
    questions_df = full_dataset["questions"]
    for q_id in full_only_ids:
        row = questions_df[questions_df["ID_question"] == q_id]
        text = ""
        cat = ""
        if len(row) > 0:
            for col in ["question_EN", "question_en", "question_DE", "question_de"]:
                if col in row.columns and pd.notna(row.iloc[0].get(col)):
                    text = str(row.iloc[0][col])[:60]
                    break
            if "_category" in row.columns:
                cat = str(row.iloc[0]["_category"])
        scores = redundancy.get(q_id, {"mean_abs_r": 0, "max_abs_r": 0, "n_high_corr": 0})
        q_info.append({
            "q_id": q_id,
            "text": text,
            "category": cat,
            **scores,
        })

    q_info.sort(key=lambda x: x["max_abs_r"], reverse=True)

    # Print all questions ranked by correlation
    print("\n" + "=" * 70)
    print("REDUNDANCY SCORES (all full-only questions, sorted by max |r|)")
    print("=" * 70)
    for i, q in enumerate(q_info):
        marker = ""
        if i < 3:
            marker = " ← HIGH"
        elif i >= len(q_info) - 3:
            marker = " ← LOW"
        print(
            f"  Q{q['q_id']} | max_r={q['max_abs_r']:.3f} | mean_r={q['mean_abs_r']:.3f} | "
            f"n_high={q['n_high_corr']:2d} | {q['category'][:25]:25s} | {q['text'][:50]}{marker}"
        )

    # 3. Compute mini baseline recommendations
    print("\n--- Computing mini baseline recommendations ---")
    rec_engine = RecommendationEngine(config=config, data_map=mini_dataset)
    mini_recs = rec_engine.run_baseline()

    # 4. Pick top 3 and bottom 3
    high_corr_qs = q_info[:3]
    low_corr_qs = q_info[-3:]

    print("\n" + "=" * 70)
    print("SINGLE-QUESTION ADDITION DISTORTION")
    print("=" * 70)
    print(f"{'':54s} | {'max_r':>6s} | {'mean_r':>6s} | {'Jaccard':>7s} | {'Spearman':>8s} | {'Kendall':>7s}")
    print("-" * 105)

    high_results = []
    low_results = []

    for label, group, results_list in [
        ("HIGH CORRELATION", high_corr_qs, high_results),
        ("LOW CORRELATION", low_corr_qs, low_results),
    ]:
        print(f"{label}:")
        for q in group:
            aug_dataset = add_question_to_mini(mini_dataset, full_dataset, q["q_id"])
            aug_engine = RecommendationEngine(config=config, data_map=aug_dataset)
            aug_recs = aug_engine.run_baseline()
            metrics = compare_recs(mini_recs, aug_recs, N_TOP)
            results_list.append(metrics)

            text_short = q["text"][:40]
            print(
                f"  Q{q['q_id']} \"{text_short}\"{'':>{max(0, 38-len(text_short))}} | "
                f"{q['max_abs_r']:6.3f} | {q['mean_abs_r']:6.3f} | "
                f"{metrics['jaccard_mean']:7.4f} | {metrics['spearman_mean']:8.4f} | "
                f"{metrics['kendall_mean']:7.4f}"
            )
        print()

    # Averages
    high_jac = np.mean([r["jaccard_mean"] for r in high_results])
    low_jac = np.mean([r["jaccard_mean"] for r in low_results])
    high_sp = np.mean([r["spearman_mean"] for r in high_results])
    low_sp = np.mean([r["spearman_mean"] for r in low_results])

    print(f"HIGH avg Jaccard: {high_jac:.4f} | LOW avg Jaccard: {low_jac:.4f} | Diff: {high_jac - low_jac:+.4f}")
    print(f"HIGH avg Spearman: {high_sp:.4f} | LOW avg Spearman: {low_sp:.4f} | Diff: {high_sp - low_sp:+.4f}")

    if abs(high_jac - low_jac) < 0.01:
        print("\n→ Difference < 0.01 — correlation does NOT drive differential distortion.")
        print("  Experiment unlikely to show meaningful signal.")
    else:
        print(f"\n→ Difference = {abs(high_jac - low_jac):.4f} — there may be a signal worth investigating.")

    # 5. Cumulative addition
    print("\n" + "=" * 70)
    print("CUMULATIVE ADDITION (all 3 at once)")
    print("=" * 70)

    for label, group in [("HIGH CORR", high_corr_qs), ("LOW CORR", low_corr_qs)]:
        q_ids = [q["q_id"] for q in group]
        aug_dataset = add_questions_to_mini(mini_dataset, full_dataset, q_ids)
        aug_engine = RecommendationEngine(config=config, data_map=aug_dataset)
        aug_recs = aug_engine.run_baseline()
        metrics = compare_recs(mini_recs, aug_recs, N_TOP)

        qs_str = ", ".join(f"Q{qid}" for qid in q_ids)
        print(
            f"  {label:9s} ({qs_str}): "
            f"Jaccard={metrics['jaccard_mean']:.4f}  "
            f"Spearman={metrics['spearman_mean']:.4f}  "
            f"Kendall={metrics['kendall_mean']:.4f}"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
