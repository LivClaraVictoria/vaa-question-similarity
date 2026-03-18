"""
Analyze pairwise voter-answer correlations segmented by mini (rapide) vs
full-only question groups.

Computes |Pearson r| and arccos distance for all question pairs, classified as:
  - within-mini (rapide=1, 30 questions)
  - within-full-only (rapide=0, 45 questions)
  - cross (mini <-> full-only)

Outputs saved to experiment_results/correlation_metric_results/correlation_analysis/.

Usage:
    python -m scripts.analyze_mini_maxi_correlations \
        --config configs/full_pipeline/base_data/pipeline_answer_corr_arccos_ZH.py
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from main import load_config
from vqs.data_loader import load_dataset

OUTPUT_DIR = Path("experiment_results/correlation_metric_results/correlation_analysis")


def compute_pairwise_correlations(
    voters_df: pd.DataFrame,
    question_ids: list[int],
) -> list[dict]:
    """Compute pairwise |Pearson r| for all question pairs using voter answers."""
    answer_cols = {}
    for qid in question_ids:
        col = f"answer_{qid}"
        if col in voters_df.columns:
            answer_cols[qid] = voters_df[col]

    pairs = []
    ids = list(answer_cols.keys())
    for i, id1 in enumerate(ids):
        vals_i = answer_cols[id1]
        for id2 in ids[i + 1:]:
            vals_j = answer_cols[id2]
            mask = vals_i.notna() & vals_j.notna()
            if mask.sum() < 10:
                continue
            r, _ = pearsonr(vals_i[mask], vals_j[mask])
            abs_r = abs(r)
            pairs.append({
                "ID1": id1,
                "ID2": id2,
                "abs_r": abs_r,
                "arccos_dist": np.arccos(np.clip(abs_r, 0.0, 1.0)),
                "signed_r": r,
            })
    return pairs


def classify_pair(id1: int, id2: int, mini_ids: set, full_only_ids: set) -> str:
    in_mini_1 = id1 in mini_ids
    in_mini_2 = id2 in mini_ids
    if in_mini_1 and in_mini_2:
        return "within-mini"
    elif not in_mini_1 and not in_mini_2:
        return "within-full-only"
    else:
        return "cross"


def compute_group_stats(df: pd.DataFrame, group_name: str) -> dict:
    stats = {"group": group_name, "n_pairs": len(df)}
    for col, label in [("abs_r", "abs_r"), ("arccos_dist", "arccos_dist")]:
        vals = df[col].values
        stats[f"{label}_mean"] = vals.mean()
        stats[f"{label}_median"] = np.median(vals)
        stats[f"{label}_std"] = vals.std()
        stats[f"{label}_min"] = vals.min()
        stats[f"{label}_max"] = vals.max()
        stats[f"{label}_p25"] = np.percentile(vals, 25)
        stats[f"{label}_p75"] = np.percentile(vals, 75)
        stats[f"{label}_p90"] = np.percentile(vals, 90)
        stats[f"{label}_p95"] = np.percentile(vals, 95)
    return stats


def plot_distributions(df_pairs: pd.DataFrame, output_path: Path):
    """2x2 figure: histograms + box plots for |r| and arccos distance by group."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    groups = ["within-mini", "within-full-only", "cross"]
    colors = {"within-mini": "#3498db", "within-full-only": "#e67e22", "cross": "#2ecc71"}
    labels = {"within-mini": "Within mini (30q)", "within-full-only": "Within full-only (45q)", "cross": "Cross (mini↔full)"}

    # Top-left: |r| histograms
    ax = axes[0, 0]
    for g in groups:
        subset = df_pairs.loc[df_pairs["group"] == g, "abs_r"]
        ax.hist(subset, bins=40, alpha=0.5, color=colors[g], label=f"{labels[g]} (n={len(subset)})", edgecolor="white")
    ax.set_xlabel("|Pearson r|")
    ax.set_ylabel("Count")
    ax.set_title("Pairwise |Pearson r| by Group")
    ax.legend(fontsize=8)

    # Top-right: arccos distance histograms
    ax = axes[0, 1]
    for g in groups:
        subset = df_pairs.loc[df_pairs["group"] == g, "arccos_dist"]
        ax.hist(subset, bins=40, alpha=0.5, color=colors[g], label=f"{labels[g]} (n={len(subset)})", edgecolor="white")
    ax.set_xlabel("Arccos distance")
    ax.set_ylabel("Count")
    ax.set_title("Arccos Distance by Group")
    ax.legend(fontsize=8)

    # Bottom-left: |r| box plots
    ax = axes[1, 0]
    data_r = [df_pairs.loc[df_pairs["group"] == g, "abs_r"].values for g in groups]
    data_r.append(df_pairs["abs_r"].values)
    bp = ax.boxplot(data_r, labels=[labels[g] for g in groups] + ["All"], vert=True, patch_artist=True)
    box_colors = [colors[g] for g in groups] + ["#95a5a6"]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("|Pearson r|")
    ax.set_title("Distribution Comparison: |Pearson r|")
    ax.tick_params(axis="x", rotation=15)

    # Bottom-right: arccos distance box plots
    ax = axes[1, 1]
    data_d = [df_pairs.loc[df_pairs["group"] == g, "arccos_dist"].values for g in groups]
    data_d.append(df_pairs["arccos_dist"].values)
    bp = ax.boxplot(data_d, labels=[labels[g] for g in groups] + ["All"], vert=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Arccos distance")
    ax.set_title("Distribution Comparison: Arccos Distance")
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Mini vs Full-Only Question Correlations (Voter Answers)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    df_pairs: pd.DataFrame,
    mini_ids: list[int],
    full_only_ids: list[int],
    cat_map: dict,
    output_path: Path,
):
    """75x75 |r| heatmap, mini questions first, with dividing line."""
    ordered_ids = list(mini_ids) + list(full_only_ids)
    n = len(ordered_ids)

    # Build matrix
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)
    pair_lookup = {}
    for _, row in df_pairs.iterrows():
        pair_lookup[(int(row["ID1"]), int(row["ID2"]))] = row["abs_r"]
        pair_lookup[(int(row["ID2"]), int(row["ID1"]))] = row["abs_r"]

    for i in range(n):
        for j in range(i + 1, n):
            val = pair_lookup.get((ordered_ids[i], ordered_ids[j]), np.nan)
            matrix[i, j] = val
            matrix[j, i] = val

    # Labels
    tick_labels = []
    for qid in ordered_ids:
        cat = cat_map.get(qid, "?")
        short_cat = cat[:15] if len(cat) > 15 else cat
        tick_labels.append(f"{qid} ({short_cat})")

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=0.8, aspect="equal")

    n_mini = len(mini_ids)
    # Draw dividing lines
    ax.axhline(n_mini - 0.5, color="black", linewidth=2)
    ax.axvline(n_mini - 0.5, color="black", linewidth=2)

    # Quadrant labels
    ax.text(n_mini / 2, -2, "Mini", ha="center", fontsize=11, fontweight="bold")
    ax.text(n_mini + (n - n_mini) / 2, -2, "Full-only", ha="center", fontsize=11, fontweight="bold")
    ax.text(-4, n_mini / 2, "Mini", va="center", fontsize=11, fontweight="bold", rotation=90)
    ax.text(-4, n_mini + (n - n_mini) / 2, "Full-only", va="center", fontsize=11, fontweight="bold", rotation=90)

    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("|Pearson r|")
    ax.set_title("Pairwise |Pearson r| — Mini (top-left) vs Full-only (bottom-right)", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pairwise voter-answer correlations by mini vs full-only groups"
    )
    parser.add_argument("--config", required=True, help="Pipeline config with load_voters=True")
    parser.add_argument("--top", type=int, default=30, help="Number of top correlated pairs to show")
    args = parser.parse_args()

    # Load data
    print("Loading dataset...")
    config = load_config(Path(args.config))
    dataset = load_dataset(config)
    questions_df = dataset["questions"]
    voters_df = dataset["voters"]

    # Classify questions
    mini_ids = set(questions_df.loc[questions_df["rapide"] == 1, "ID_question"].tolist())
    full_only_ids = set(questions_df.loc[questions_df["rapide"] == 0, "ID_question"].tolist())
    all_ids = sorted(mini_ids | full_only_ids)
    print(f"  Mini: {len(mini_ids)} questions, Full-only: {len(full_only_ids)} questions")

    cat_map = questions_df.set_index("ID_question")["_category"].to_dict()

    # Compute pairwise correlations
    print("Computing pairwise correlations...")
    pairs = compute_pairwise_correlations(voters_df, all_ids)
    df_pairs = pd.DataFrame(pairs)
    print(f"  {len(df_pairs)} pairs computed")

    # Classify pairs
    df_pairs["group"] = df_pairs.apply(
        lambda r: classify_pair(int(r["ID1"]), int(r["ID2"]), mini_ids, full_only_ids), axis=1
    )
    df_pairs["cat1"] = df_pairs["ID1"].map(cat_map)
    df_pairs["cat2"] = df_pairs["ID2"].map(cat_map)

    # Print group counts
    for g in ["within-mini", "within-full-only", "cross"]:
        n = (df_pairs["group"] == g).sum()
        print(f"  {g}: {n} pairs")

    # Compute per-group stats
    stats_rows = []
    for g in ["within-mini", "within-full-only", "cross", "all"]:
        subset = df_pairs if g == "all" else df_pairs[df_pairs["group"] == g]
        stats_rows.append(compute_group_stats(subset, g))
    df_stats = pd.DataFrame(stats_rows)

    # Report
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"{'=' * 75}")
    p("MINI vs FULL-ONLY CORRELATION ANALYSIS")
    p(f"  Mini (rapide=1): {len(mini_ids)} questions")
    p(f"  Full-only (rapide=0): {len(full_only_ids)} questions")
    p(f"  Total pairs: {len(df_pairs)}")
    p(f"{'=' * 75}")

    p(f"\n  {'Group':<22} {'N':>6} {'Mean |r|':>9} {'Med |r|':>9} {'Max |r|':>9} "
      f"{'Mean dist':>10} {'Med dist':>10}")
    p(f"  {'-' * 80}")
    for _, row in df_stats.iterrows():
        p(f"  {row['group']:<22} {int(row['n_pairs']):>6} "
          f"{row['abs_r_mean']:>9.4f} {row['abs_r_median']:>9.4f} {row['abs_r_max']:>9.4f} "
          f"{row['arccos_dist_mean']:>10.4f} {row['arccos_dist_median']:>10.4f}")

    # Top correlated pairs
    top_df = df_pairs.nlargest(args.top, "abs_r")
    p(f"\n{'=' * 75}")
    p(f"TOP {args.top} MOST CORRELATED PAIRS (by |Pearson r|)")
    p(f"{'=' * 75}")
    p(f"\n  {'Rank':>4} {'ID1':>8} {'ID2':>8} {'|r|':>7} {'dist':>7} {'Group':<20} {'Categories'}")
    p(f"  {'-' * 85}")
    for rank, (_, row) in enumerate(top_df.iterrows(), 1):
        p(f"  {rank:>4} {int(row['ID1']):>8} {int(row['ID2']):>8} "
          f"{row['abs_r']:>7.4f} {row['arccos_dist']:>7.4f} "
          f"{row['group']:<20} {row['cat1']} / {row['cat2']}")

    # Group breakdown of top pairs
    p(f"\n  Top-{args.top} group breakdown:")
    for g in ["within-mini", "within-full-only", "cross"]:
        n = (top_df["group"] == g).sum()
        p(f"    {g}: {n}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = f"mini_maxi_corr_{timestamp}"

    stats_path = OUTPUT_DIR / f"{base}_stats.csv"
    df_stats.to_csv(stats_path, index=False)

    all_pairs_path = OUTPUT_DIR / f"{base}_all_pairs.csv"
    df_pairs.sort_values("abs_r", ascending=False).to_csv(all_pairs_path, index=False)

    top_path = OUTPUT_DIR / f"{base}_top_pairs.csv"
    top_df.to_csv(top_path, index=False)

    dist_plot_path = OUTPUT_DIR / f"{base}_distributions.png"
    plot_distributions(df_pairs, dist_plot_path)

    heatmap_path = OUTPUT_DIR / f"{base}_heatmap.png"
    plot_heatmap(df_pairs, sorted(mini_ids), sorted(full_only_ids), cat_map, heatmap_path)

    report_path = OUTPUT_DIR / f"{base}_report.txt"
    report_path.write_text("\n".join(lines))

    p(f"\n--- Saved to {OUTPUT_DIR}/ ---")
    p(f"  {stats_path.name}")
    p(f"  {all_pairs_path.name}")
    p(f"  {top_path.name}")
    p(f"  {dist_plot_path.name}")
    p(f"  {heatmap_path.name}")
    p(f"  {report_path.name}")


if __name__ == "__main__":
    main()
