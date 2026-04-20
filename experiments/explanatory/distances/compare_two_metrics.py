"""
Compare two distance metrics (e.g., embedding-based vs answer-correlation).

Produces a scatter plot, rank-comparison bar chart, and summary statistics
showing how the two metrics agree or diverge on question-pair distances.

Usage:
    python -m scripts.compare_distance_metrics \
        --config_a configs/base_pipeline/pipeline_answer_corr_ZH.py \
        --config_b configs/base_pipeline/pipeline_e5_instruct_ZH.py

    # Or with direct distance files (skip pipeline computation):
    python -m scripts.compare_distance_metrics \
        --file_a cache/distance_calculations/dist_answer_corr.parquet \
        --file_b cache/distance_calculations/dist_e5_instruct.parquet \
        --label_a "Answer Correlation" --label_b "E5-Instruct"
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
from pathlib import Path
from scipy import stats


OUTPUT_DIR = Path("experiment_results/embedding_correlation_comparison")


def load_distances_from_config(config_path: str) -> tuple[pd.DataFrame, str]:
    """Load a config, compute/load distances via the pipeline, return (df, label)."""
    from vqs.config_utils import load_config
    from vqs.similarity_metrics import get_calculator
    from vqs.data_loader import load_dataset

    config = load_config(Path(config_path))
    label = config.dist
    dataset = load_dataset(config)
    calculator = get_calculator(config)
    df = calculator.calculate_distance(dataset, config)
    return df, label


def load_distances_from_file(file_path: str) -> pd.DataFrame:
    """Load a distance DataFrame from a parquet or CSV file."""
    p = Path(file_path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def merge_distances(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    """Merge two distance DataFrames on question ID pairs."""
    # Normalize column names: both should have "Distance"
    val_a = "Distance" if "Distance" in df_a.columns else "Similarity"
    val_b = "Distance" if "Distance" in df_b.columns else "Similarity"

    a = df_a[["ID1", "ID2", "Qu1", "Qu2", val_a]].rename(columns={val_a: label_a})
    b = df_b[["ID1", "ID2", val_b]].rename(columns={val_b: label_b})

    # Ensure consistent ordering (smaller ID first) for symmetric metrics
    for df in [a, b]:
        mask = df["ID1"] > df["ID2"]
        df.loc[mask, ["ID1", "ID2"]] = df.loc[mask, ["ID2", "ID1"]].values

    merged = a.merge(b, on=["ID1", "ID2"], how="inner")
    return merged


def plot_scatter(merged: pd.DataFrame, label_a: str, label_b: str, output_dir: Path):
    """Scatter plot of metric A vs metric B distances."""
    fig, ax = plt.subplots(figsize=(8, 8))

    x = merged[label_b].values
    y = merged[label_a].values

    ax.scatter(x, y, alpha=0.3, s=15, edgecolors="none")

    # Correlation stats
    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)

    ax.set_xlabel(f"{label_b} distance", fontsize=12)
    ax.set_ylabel(f"{label_a} distance", fontsize=12)
    ax.set_title(
        f"Distance Comparison: {label_a} vs {label_b}\n"
        f"Pearson r = {r_pearson:.3f}, Spearman r = {r_spearman:.3f}",
        fontsize=13,
    )

    # Reference line (only if same scale)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="y = x")

    ax.legend()
    plt.tight_layout()
    path = output_dir / "distance_scatter.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved scatter plot to {path}")


def plot_rank_comparison(merged: pd.DataFrame, label_a: str, label_b: str, output_dir: Path, top_n: int = 20):
    """Side-by-side bar chart: top-N closest pairs per metric."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax, label in zip(axes, [label_a, label_b]):
        top = merged.nsmallest(top_n, label).copy()
        top["pair"] = top["Qu1"].str[:40] + "\n× " + top["Qu2"].str[:40]
        top = top.sort_values(label, ascending=True)

        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top)))
        ax.barh(range(len(top)), top[label].values, color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["pair"].values, fontsize=7)
        ax.set_xlabel("Distance", fontsize=11)
        ax.set_title(f"Top {top_n} Closest Pairs ({label})", fontsize=12)
        ax.invert_yaxis()

    plt.tight_layout()
    path = output_dir / "top_pairs_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved rank comparison to {path}")


def plot_divergence(merged: pd.DataFrame, label_a: str, label_b: str, output_dir: Path, top_n: int = 15):
    """Highlight pairs where the two metrics disagree most (by rank difference)."""
    merged = merged.copy()
    merged[f"rank_{label_a}"] = merged[label_a].rank()
    merged[f"rank_{label_b}"] = merged[label_b].rank()
    merged["rank_diff"] = (merged[f"rank_{label_a}"] - merged[f"rank_{label_b}"]).abs()

    top = merged.nlargest(top_n, "rank_diff").copy()
    top["pair"] = top["Qu1"].str[:40] + "\n× " + top["Qu2"].str[:40]
    top = top.sort_values("rank_diff", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(top))
    ax.barh(y_pos, top["rank_diff"].values, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["pair"].values, fontsize=7)
    ax.set_xlabel("Absolute Rank Difference", fontsize=11)
    ax.set_title(
        f"Most Divergent Pairs: {label_a} vs {label_b}\n"
        f"(pairs where the two metrics disagree most on relative closeness)",
        fontsize=12,
    )
    ax.invert_yaxis()

    plt.tight_layout()
    path = output_dir / "divergent_pairs.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved divergence plot to {path}")


def print_summary(merged: pd.DataFrame, label_a: str, label_b: str):
    """Print summary statistics."""
    x = merged[label_b].values
    y = merged[label_a].values

    r_pearson, _ = stats.pearsonr(x, y)
    r_spearman, _ = stats.spearmanr(x, y)
    r_kendall, _ = stats.kendalltau(x, y)

    print(f"\n{'='*60}")
    print(f"Distance Comparison: {label_a} vs {label_b}")
    print(f"{'='*60}")
    print(f"Question pairs:     {len(merged)}")
    print(f"\n{label_a:30s}  range: [{y.min():.4f}, {y.max():.4f}]  mean: {y.mean():.4f}")
    print(f"{label_b:30s}  range: [{x.min():.4f}, {x.max():.4f}]  mean: {x.mean():.4f}")
    print(f"\nCorrelation between the two distance vectors:")
    print(f"  Pearson:   {r_pearson:.4f}")
    print(f"  Spearman:  {r_spearman:.4f}")
    print(f"  Kendall:   {r_kendall:.4f}")

    # Rank agreement: how many of the top-10 closest pairs overlap?
    for k in [10, 20]:
        top_a = set(merged.nsmallest(k, label_a).index)
        top_b = set(merged.nsmallest(k, label_b).index)
        overlap = len(top_a & top_b)
        print(f"\n  Top-{k} closest pairs overlap: {overlap}/{k} ({100*overlap/k:.0f}%)")


def save_merged_csv(merged: pd.DataFrame, label_a: str, label_b: str, output_dir: Path):
    """Save the merged comparison DataFrame."""
    path = output_dir / "distance_comparison.csv"
    merged.to_csv(path, index=False)
    print(f"Saved merged data to {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare two distance metrics")
    parser.add_argument("--config_a", type=str, help="Config path for metric A")
    parser.add_argument("--config_b", type=str, help="Config path for metric B")
    parser.add_argument("--file_a", type=str, help="Direct distance file path for metric A (alternative to --config_a)")
    parser.add_argument("--file_b", type=str, help="Direct distance file path for metric B (alternative to --config_b)")
    parser.add_argument("--label_a", type=str, default=None, help="Display label for metric A")
    parser.add_argument("--label_b", type=str, default=None, help="Display label for metric B")
    args = parser.parse_args()

    # Load metric A
    if args.config_a:
        df_a, auto_label_a = load_distances_from_config(args.config_a)
    elif args.file_a:
        df_a = load_distances_from_file(args.file_a)
        auto_label_a = Path(args.file_a).stem
    else:
        parser.error("Provide either --config_a or --file_a")

    # Load metric B
    if args.config_b:
        df_b, auto_label_b = load_distances_from_config(args.config_b)
    elif args.file_b:
        df_b = load_distances_from_file(args.file_b)
        auto_label_b = Path(args.file_b).stem
    else:
        parser.error("Provide either --config_b or --file_b")

    label_a = args.label_a or auto_label_a
    label_b = args.label_b or auto_label_b

    # Merge on (ID1, ID2)
    merged = merge_distances(df_a, df_b, label_a, label_b)
    print(f"\nMerged {len(merged)} question pairs.")

    if len(merged) == 0:
        print("ERROR: No overlapping question pairs found. Check that both metrics cover the same questions.")
        return

    # Output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_summary(merged, label_a, label_b)
    save_merged_csv(merged, label_a, label_b, OUTPUT_DIR)
    plot_scatter(merged, label_a, label_b, OUTPUT_DIR)
    plot_rank_comparison(merged, label_a, label_b, OUTPUT_DIR)
    plot_divergence(merged, label_a, label_b, OUTPUT_DIR)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
