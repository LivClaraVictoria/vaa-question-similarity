"""
Metric agreement analysis: shows that Jaccard, Spearman, and Kendall metrics
move together, justifying the use of Jaccard alone in main figures.

Uses existing question alpha sweep data for E5-INSTRUCT and ANSWER-CORRELATION-ARCCOS.

Usage:
    python -m scripts.metric_agreement_analysis
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

OUTPUT_DIR = Path("experiment_results/metric_agreement")

DATA_SOURCES = {
    "E5-INSTRUCT": (
        "experiment_results/exp1/question_alpha_sweep/e5_instruct_5ct_n4/compiled/"
        "question_alpha_sweep_pipeline_e5_instruct_ZH_a04_5ct_n5_0311_1543.csv"
    ),
    "ANSWER-CORR-ARCCOS": (
        "experiment_results/exp1/question_alpha_sweep/"
        "question_alpha_sweep_pipeline_answer_corr_arccos_ZH_identical_n5/"
        "question_alpha_sweep_pipeline_answer_corr_arccos_ZH_identical_n5_0317_1911.csv"
    ),
}

CRW_METRICS = ["crw_jaccard_mean", "crw_spearman_mean", "crw_kendall_mean"]
BASE_METRICS = ["base_jaccard_mean", "base_spearman_mean", "base_kendall_mean"]
METRIC_LABELS = {"jaccard": "Jaccard", "spearman": "Spearman", "kendall": "Kendall"}
METRIC_PAIRS = [("jaccard", "spearman"), ("jaccard", "kendall"), ("spearman", "kendall")]


def compute_pairwise_correlations(df, metrics, condition_label, dataset_label):
    """Compute Spearman correlations between all pairs of metrics."""
    rows = []
    short_names = [m.split("_")[1] for m in metrics]  # e.g. "jaccard"
    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            rho, p = spearmanr(df[metrics[i]], df[metrics[j]])
            rows.append({
                "dataset": dataset_label,
                "condition": condition_label,
                "metric_a": short_names[i],
                "metric_b": short_names[j],
                "spearman_rho": rho,
                "p_value": p,
                "n_observations": len(df),
            })
    return rows


def plot_scatter_matrix(df, dataset_label, has_clone_types):
    """3×2 scatter: top row CRW, bottom row baseline."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row_idx, (metrics, cond_label) in enumerate([
        (CRW_METRICS, "CRW"),
        (BASE_METRICS, "Baseline"),
    ]):
        for col_idx, (ma, mb) in enumerate(METRIC_PAIRS):
            ax = axes[row_idx, col_idx]
            col_a = f"{cond_label.lower()}_{ma}_mean" if cond_label == "Baseline" else f"crw_{ma}_mean"
            col_b = f"{cond_label.lower()}_{mb}_mean" if cond_label == "Baseline" else f"crw_{mb}_mean"
            # Fix baseline column names
            col_a = f"base_{ma}_mean" if cond_label == "Baseline" else f"crw_{ma}_mean"
            col_b = f"base_{mb}_mean" if cond_label == "Baseline" else f"crw_{mb}_mean"

            if has_clone_types:
                for ct in df["clone_type"].unique():
                    mask = df["clone_type"] == ct
                    ax.scatter(df.loc[mask, col_a], df.loc[mask, col_b],
                               alpha=0.15, s=8, label=ct.replace("_", " "))
            else:
                ax.scatter(df[col_a], df[col_b], alpha=0.15, s=8, color="#2196F3")

            # Compute and annotate correlation
            rho, _ = spearmanr(df[col_a], df[col_b])
            ax.annotate(f"$\\rho_s$ = {rho:.3f}", xy=(0.05, 0.92),
                        xycoords="axes fraction", fontsize=11, fontweight="bold")

            ax.set_xlabel(f"{METRIC_LABELS[ma]} (mean)")
            ax.set_ylabel(f"{METRIC_LABELS[mb]} (mean)")
            ax.set_title(f"{cond_label}: {METRIC_LABELS[ma]} vs {METRIC_LABELS[mb]}")

    if has_clone_types:
        # Single legend for all subplots
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Metric Agreement — {dataset_label}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03 if has_clone_types else 0, 1, 0.96])
    return fig


def plot_best_alpha_scatter(dfs_dict):
    """For each question × clone_type, take the alpha maximizing CRW Jaccard.
    Plot Spearman vs Jaccard and Kendall vs Jaccard side by side, one row per dataset."""
    n_datasets = len(dfs_dict)
    fig, axes = plt.subplots(n_datasets, 2, figsize=(12, 5 * n_datasets))
    if n_datasets == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (name, df) in enumerate(dfs_dict.items()):
        # Group and pick best alpha per question (× clone_type if present)
        group_cols = ["question_id"]
        if "clone_type" in df.columns:
            group_cols.append("clone_type")

        best = df.loc[df.groupby(group_cols)["crw_jaccard_mean"].idxmax()]

        for col_idx, other in enumerate(["spearman", "kendall"]):
            ax = axes[row_idx, col_idx]
            col_other = f"crw_{other}_mean"

            if "clone_type" in best.columns:
                for ct in best["clone_type"].unique():
                    mask = best["clone_type"] == ct
                    ax.scatter(best.loc[mask, "crw_jaccard_mean"],
                               best.loc[mask, col_other],
                               alpha=0.6, s=25, label=ct.replace("_", " "))
            else:
                ax.scatter(best["crw_jaccard_mean"], best[col_other],
                           alpha=0.6, s=25, color="#2196F3")

            rho, _ = spearmanr(best["crw_jaccard_mean"], best[col_other])
            ax.annotate(f"$\\rho_s$ = {rho:.3f}", xy=(0.05, 0.92),
                        xycoords="axes fraction", fontsize=11, fontweight="bold")
            ax.ticklabel_format(useOffset=False)
            ax.set_xlabel("CRW Jaccard (mean)")
            ax.set_ylabel(f"CRW {METRIC_LABELS[other]} (mean)")
            ax.set_title(f"{name} — Best-alpha per question")

        if "clone_type" in best.columns:
            handles, labels = axes[row_idx, 0].get_legend_handles_labels()
            for ax in axes[row_idx]:
                ax.legend(handles, labels, fontsize=8, loc="lower right")

    plt.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    all_corr_rows = []
    loaded = {}

    for name, path in DATA_SOURCES.items():
        print(f"\nLoading {name}: {path}")
        df = pd.read_csv(path)
        loaded[name] = df
        print(f"  Rows: {len(df)}")

        has_clone_types = "clone_type" in df.columns and df["clone_type"].nunique() > 1

        # Pairwise correlations — all rows
        all_corr_rows.extend(compute_pairwise_correlations(df, CRW_METRICS, "crw", name))
        all_corr_rows.extend(compute_pairwise_correlations(df, BASE_METRICS, "baseline", name))

        # Per clone type if available
        if has_clone_types:
            for ct in df["clone_type"].unique():
                subset = df[df["clone_type"] == ct]
                all_corr_rows.extend(
                    compute_pairwise_correlations(subset, CRW_METRICS, f"crw_{ct}", name)
                )

        # Scatter matrix
        fig = plot_scatter_matrix(df, name, has_clone_types)
        slug = name.lower().replace("-", "_").replace(" ", "_")
        fig.savefig(OUTPUT_DIR / f"metric_agreement_scatter_{slug}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved scatter: metric_agreement_scatter_{slug}.png")

    # Summary CSV
    corr_df = pd.DataFrame(all_corr_rows)
    corr_df.to_csv(OUTPUT_DIR / "metric_agreement_summary.csv", index=False)
    print(f"\nSaved summary CSV ({len(corr_df)} rows)")

    # Print summary table
    print("\n=== Metric Agreement Summary ===")
    for _, row in corr_df[corr_df["condition"].isin(["crw", "baseline"])].iterrows():
        print(f"  {row['dataset']:25s} {row['condition']:10s} "
              f"{row['metric_a']:>8s} vs {row['metric_b']:<8s}: "
              f"rho={row['spearman_rho']:.4f}  (n={row['n_observations']})")

    # Best-alpha scatter
    fig = plot_best_alpha_scatter(loaded)
    fig.savefig(OUTPUT_DIR / "metric_agreement_per_question_best.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved best-alpha scatter")

    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
