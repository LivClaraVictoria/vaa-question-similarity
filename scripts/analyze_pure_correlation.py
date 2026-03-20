"""
Analyze pure Pearson correlations between all question pairs.

Not a distance metric — shows raw signed correlations from voter answer data.
Produces a density diagram, sorted CSV, and top-N most correlated pairs.

Usage:
    python -m scripts.analyze_pure_correlation --config configs/full_pipeline/base_data/pipeline_answer_corr_ZH.py
    python -m scripts.analyze_pure_correlation --config configs/full_pipeline/base_data/pipeline_answer_corr_ZH.py --top 30
"""

import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("experiment_results/distance_analysis/PURE_CORRELATION_2023")


def main():
    parser = argparse.ArgumentParser(description="Analyze pure Pearson correlations between questions")
    parser.add_argument("--config", required=True, help="Pipeline config path (needs load_voters=True)")
    parser.add_argument("--top", type=int, default=20, help="Number of top correlated pairs to show")
    args = parser.parse_args()

    # Load config and dataset
    from main import load_config
    from vqs.data_loader import load_dataset

    config = load_config(Path(args.config))
    dataset = load_dataset(config)

    questions_df = dataset["questions"]
    question_ids = questions_df["ID_question"].tolist()
    questions_text = questions_df.rename(columns=str.lower)["question_en"].tolist()
    id_to_text = dict(zip(question_ids, questions_text))

    # Compute pairwise Pearson correlations from voter answers
    answer_cols = [f"answer_{qid}" for qid in question_ids]
    voters_df = dataset["voters"][answer_cols]
    corr_matrix = voters_df.corr()

    # Extract upper triangle into list
    rows = []
    for i, j in combinations(range(len(question_ids)), 2):
        id1, id2 = question_ids[i], question_ids[j]
        r = corr_matrix.iloc[i, j]
        rows.append({
            "ID1": id1,
            "ID2": id2,
            "Qu1": id_to_text[id1],
            "Qu2": id_to_text[id2],
            "correlation": r,
            "abs_correlation": abs(r),
        })

    df = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    correlations = df["correlation"].values

    # Summary stats
    stats = {
        "n_pairs": len(df),
        "n_questions": len(question_ids),
        "mean_r": correlations.mean(),
        "median_r": np.median(correlations),
        "std_r": correlations.std(),
        "min_r": correlations.min(),
        "max_r": correlations.max(),
        "mean_abs_r": df["abs_correlation"].mean(),
        "median_abs_r": df["abs_correlation"].median(),
        "max_abs_r": df["abs_correlation"].max(),
        "n_above_03": (df["abs_correlation"] > 0.3).sum(),
        "n_above_05": (df["abs_correlation"] > 0.5).sum(),
        "n_negative": (correlations < 0).sum(),
        "n_positive": (correlations > 0).sum(),
    }

    # Build report
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"{'=' * 70}")
    p(f"PURE CORRELATION ANALYSIS")
    p(f"{'=' * 70}")
    p(f"  Questions:  {stats['n_questions']}")
    p(f"  Pairs:      {stats['n_pairs']}")
    p()
    p(f"  {'Statistic':<25} {'Value':>10}")
    p(f"  {'-' * 37}")
    for key in ["mean_r", "median_r", "std_r", "min_r", "max_r",
                "mean_abs_r", "median_abs_r", "max_abs_r"]:
        p(f"  {key:<25} {stats[key]:>10.4f}")
    p()
    p(f"  Positive correlations:  {stats['n_positive']}")
    p(f"  Negative correlations:  {stats['n_negative']}")
    p(f"  |r| > 0.3:             {stats['n_above_03']}")
    p(f"  |r| > 0.5:             {stats['n_above_05']}")

    # Top pairs
    p()
    p(f"{'=' * 70}")
    p(f"TOP {args.top} MOST CORRELATED PAIRS (by |r|)")
    p(f"{'=' * 70}")
    for _, row in df.head(args.top).iterrows():
        id1, id2 = int(row["ID1"]), int(row["ID2"])
        r = row["correlation"]
        q1 = str(row["Qu1"])[:50]
        q2 = str(row["Qu2"])[:50]
        p(f"  {id1:>8} — {id2:>8}  r={r:+.4f}  |r|={abs(r):.4f}")
        p(f"    Q1: {q1}")
        p(f"    Q2: {q2}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw correlation distribution
    ax = axes[0]
    ax.hist(correlations, bins=50, alpha=0.7, color="steelblue", edgecolor="white", density=True)
    from scipy.stats import gaussian_kde
    xs = np.linspace(correlations.min() - 0.05, correlations.max() + 0.05, 200)
    kde = gaussian_kde(correlations)
    ax.plot(xs, kde(xs), color="darkblue", linewidth=1.5)
    ax.axvline(0, color="black", linestyle=":", alpha=0.3)
    ax.axvline(np.median(correlations), color="red", linestyle="--",
               label=f"median={np.median(correlations):.3f}")
    ax.axvline(correlations.mean(), color="orange", linestyle="--",
               label=f"mean={correlations.mean():.3f}")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Density")
    ax.set_title("Raw correlation distribution")
    ax.legend(fontsize=8)

    # Right: |r| distribution
    ax = axes[1]
    abs_corr = df["abs_correlation"].values
    ax.hist(abs_corr, bins=50, alpha=0.7, color="coral", edgecolor="white", density=True)
    xs2 = np.linspace(0, abs_corr.max() + 0.05, 200)
    kde2 = gaussian_kde(abs_corr)
    ax.plot(xs2, kde2(xs2), color="darkred", linewidth=1.5)
    ax.axvline(np.median(abs_corr), color="red", linestyle="--",
               label=f"median={np.median(abs_corr):.3f}")
    ax.axvline(abs_corr.mean(), color="orange", linestyle="--",
               label=f"mean={abs_corr.mean():.3f}")
    ax.set_xlabel("|Pearson r|")
    ax.set_ylabel("Density")
    ax.set_title("|Correlation| distribution")
    ax.legend(fontsize=8)

    fig.suptitle("Pure Correlation Analysis — All Question Pairs", fontsize=12, y=1.02)
    fig.tight_layout()

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base_name = f"pure_correlation_{timestamp}"

    csv_path = OUTPUT_DIR / f"{base_name}_all_correlations.csv"
    df.to_csv(csv_path, index=False)

    plot_path = OUTPUT_DIR / f"{base_name}_distribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report_path = OUTPUT_DIR / f"{base_name}_report.txt"
    report_path.write_text("\n".join(lines))

    p()
    p(f"--- Saved to {OUTPUT_DIR}/ ---")
    p(f"  {csv_path.name}")
    p(f"  {plot_path.name}")
    p(f"  {report_path.name}")


if __name__ == "__main__":
    main()
