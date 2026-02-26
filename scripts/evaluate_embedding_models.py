"""
Evaluate embedding models on the fake benchmark dataset.

Reads distance CSVs from experiment_results/distance_metric/fake_results/,
computes per-model scoring metrics, and saves a comparison table + plot
to experiment_results/model_benchmark/.

Usage:
    python scripts/evaluate_embedding_models.py
    python scripts/evaluate_embedding_models.py --results-dir experiment_results/distance_metric/fake_results/
    python scripts/evaluate_embedding_models.py --output-dir experiment_results/model_benchmark/
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Categories classified by group
SAME_TOPIC_CATS = {
    "Negation",
    "Easy Paraphrase",
    "Easy Paraphrase Negation",
    "Hard Paraphrase",
    "Hard Paraphrase Negation",
}
TRAP_CATS = {"Syntax Trap", "Keyword Trap"}
NEGATION_CATS = {"Negation"}


def parse_model_name(filename: str) -> str | None:
    """Extract model name from filename like 'E5_fake_20260202_0004.csv'.

    Normalizes to uppercase to deduplicate e.g. 'E5' vs 'e5'.
    """
    # Format: {dist_method}_{data_year}_{timestamp}_{hash}.{ext}
    # or legacy: {dist_method}_fake.csv
    parts = filename.split("_")
    if len(parts) < 2:
        return None

    # Reconstruct model name: everything before 'fake' or before first timestamp
    model_parts = []
    for part in parts:
        # Stop at 'fake', a year like '2023', or a timestamp-like number (8+ digits)
        if part == "fake" or (part.isdigit() and len(part) >= 4):
            break
        model_parts.append(part)

    if not model_parts:
        return None
    return "-".join(model_parts).upper()


def load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all fake result files (parquet + CSV), grouped by model name.

    Skips files without Cat1/Cat2 columns (legacy format) and
    deduplicates by model name (keeps most recent file).
    Parquet files take priority over CSVs for the same model.
    """
    models = {}
    # Collect all result files: CSVs first, then parquets (so parquets override)
    files = sorted(results_dir.glob("*.csv")) + sorted(results_dir.glob("*.parquet"))
    for f in files:
        # Skip the comparison output file itself
        if "model_comparison" in f.name:
            continue
        model = parse_model_name(f.name)
        if model is None:
            continue
        if f.suffix == ".parquet":
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f)
        # Skip legacy files without category columns
        if "Cat1" not in df.columns or "Cat2" not in df.columns:
            print(f"  Skipping {f.name} (no Cat1/Cat2 columns — legacy format)")
            continue
        # If multiple runs exist for same model, keep the most recent (last alphabetically)
        # Parquet files override CSVs since they're processed second
        models[model] = df
    return models


def score_anchor(df: pd.DataFrame, metric_col: str, is_distance: bool) -> dict | None:
    """Compute metrics for a single anchor group.

    Returns None if essential categories are missing.
    """
    same_topic = df[df["Cat2"].isin(SAME_TOPIC_CATS)]
    traps = df[df["Cat2"].isin(TRAP_CATS)]
    negations = df[df["Cat2"].isin(NEGATION_CATS)]

    if traps.empty or negations.empty or same_topic.empty:
        return None

    mean_trap = traps[metric_col].mean()
    if mean_trap == 0:
        return None  # avoid division by zero

    # 1. Negation invariance: dist(anchor, negation) / mean(trap distances)
    neg_invariance = negations[metric_col].mean() / mean_trap

    # 2. Topic coherence: mean(same-topic distances) / mean(trap distances)
    topic_coherence = same_topic[metric_col].mean() / mean_trap

    # 3. Ordering accuracy: % of (same-topic, trap) pairs where same_topic < trap
    correct = 0
    total = 0
    for _, st_row in same_topic.iterrows():
        for _, tr_row in traps.iterrows():
            total += 1
            if is_distance:
                if st_row[metric_col] < tr_row[metric_col]:
                    correct += 1
            else:
                # Similarity: higher = more similar, so same-topic should be higher
                if st_row[metric_col] > tr_row[metric_col]:
                    correct += 1

    ordering_accuracy = (correct / total * 100) if total > 0 else 0.0

    return {
        "negation_invariance": neg_invariance,
        "topic_coherence": topic_coherence,
        "ordering_accuracy": ordering_accuracy,
    }


def evaluate_model(df: pd.DataFrame) -> dict:
    """Compute metrics for a model across all anchors."""
    # Determine metric column
    if "Distance" in df.columns:
        metric_col = "Distance"
        is_distance = True
    elif "Similarity" in df.columns:
        metric_col = "Similarity"
        is_distance = False
    else:
        raise ValueError("No 'Distance' or 'Similarity' column found.")

    # Filter to anchor comparisons only
    df = df[df["Cat1"] == "ANCHOR"].copy()

    # Group by anchor_id if present, otherwise treat as single anchor
    if "anchor_id" in df.columns:
        groups = df.groupby("anchor_id")
    else:
        groups = [("single", df)]

    anchor_scores = []
    for _, group_df in groups:
        score = score_anchor(group_df, metric_col, is_distance)
        if score is not None:
            anchor_scores.append(score)

    if not anchor_scores:
        return {
            "negation_invariance_mean": np.nan,
            "negation_invariance_std": np.nan,
            "topic_coherence_mean": np.nan,
            "topic_coherence_std": np.nan,
            "ordering_accuracy_mean": np.nan,
            "ordering_accuracy_std": np.nan,
            "n_anchors": 0,
        }

    scores_df = pd.DataFrame(anchor_scores)
    return {
        "negation_invariance_mean": scores_df["negation_invariance"].mean(),
        "negation_invariance_std": scores_df["negation_invariance"].std(),
        "topic_coherence_mean": scores_df["topic_coherence"].mean(),
        "topic_coherence_std": scores_df["topic_coherence"].std(),
        "ordering_accuracy_mean": scores_df["ordering_accuracy"].mean(),
        "ordering_accuracy_std": scores_df["ordering_accuracy"].std(),
        "n_anchors": len(anchor_scores),
    }


def plot_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Create grouped bar chart comparing models across metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.set_theme(style="whitegrid")

    models = comparison_df["model"].tolist()
    x = np.arange(len(models))
    bar_width = 0.6

    # Plot 1: Negation invariance (lower is better)
    ax = axes[0]
    vals = comparison_df["negation_invariance_mean"].values
    errs = comparison_df["negation_invariance_std"].fillna(0).values
    bars = ax.bar(x, vals, bar_width, yerr=errs, capsize=4, color="steelblue")
    ax.set_title("Negation Invariance\n(lower = better)")
    ax.set_ylabel("dist(anchor, negation) / mean(trap dist)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")

    # Plot 2: Topic coherence (lower is better)
    ax = axes[1]
    vals = comparison_df["topic_coherence_mean"].values
    errs = comparison_df["topic_coherence_std"].fillna(0).values
    bars = ax.bar(x, vals, bar_width, yerr=errs, capsize=4, color="seagreen")
    ax.set_title("Topic Coherence\n(lower = better)")
    ax.set_ylabel("mean(same-topic dist) / mean(trap dist)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")

    # Plot 3: Ordering accuracy (higher is better)
    ax = axes[2]
    vals = comparison_df["ordering_accuracy_mean"].values
    errs = comparison_df["ordering_accuracy_std"].fillna(0).values
    bars = ax.bar(x, vals, bar_width, yerr=errs, capsize=4, color="coral")
    ax.set_title("Ordering Accuracy\n(higher = better)")
    ax.set_ylabel("% same-topic < trap")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")

    plt.suptitle("Embedding Model Comparison on Fake Benchmark", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding models on fake benchmark")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiment_results/distance_metric/fake_results"),
        help="Directory containing fake result CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_results/model_benchmark"),
        help="Directory to save evaluation outputs (CSV + plot)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all model results
    models = load_results(results_dir)
    if not models:
        print("No result CSVs found.")
        return

    print(f"Found {len(models)} model(s): {', '.join(models.keys())}\n")

    # Evaluate each model
    rows = []
    for model_name, df in sorted(models.items()):
        scores = evaluate_model(df)
        scores["model"] = model_name
        rows.append(scores)

    comparison_df = pd.DataFrame(rows)
    # Reorder columns
    col_order = [
        "model",
        "negation_invariance_mean",
        "negation_invariance_std",
        "topic_coherence_mean",
        "topic_coherence_std",
        "ordering_accuracy_mean",
        "ordering_accuracy_std",
        "n_anchors",
    ]
    comparison_df = comparison_df[col_order]

    # Sort by ordering accuracy (primary, desc), then negation invariance (tiebreaker, asc)
    comparison_df = comparison_df.sort_values(
        ["ordering_accuracy_mean", "negation_invariance_mean"],
        ascending=[False, True],
    )

    # Print to console
    print("=" * 90)
    print("MODEL COMPARISON — Fake Benchmark")
    print("=" * 90)
    for _, row in comparison_df.iterrows():
        print(f"\n  {row['model']} ({int(row['n_anchors'])} anchor(s))")
        print(f"    Negation invariance : {row['negation_invariance_mean']:.4f} (std: {row['negation_invariance_std']:.4f})")
        print(f"    Topic coherence     : {row['topic_coherence_mean']:.4f} (std: {row['topic_coherence_std']:.4f})")
        print(f"    Ordering accuracy   : {row['ordering_accuracy_mean']:.1f}% (std: {row['ordering_accuracy_std']:.1f}%)")
    print("\n" + "=" * 90)

    # Save CSV
    csv_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nComparison CSV saved to: {csv_path}")

    # Save plot
    plot_path = output_dir / "model_comparison.png"
    plot_comparison(comparison_df, plot_path)


if __name__ == "__main__":
    main()
