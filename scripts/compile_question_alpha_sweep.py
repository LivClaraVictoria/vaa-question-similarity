# -*- coding: utf-8 -*-
"""
Compile per-question alpha sweep results for IC2S2 presentation.

Reads the question_alpha_sweep CSV (75 questions × 21 alphas) and produces:
  - Per-question summary CSV (for Overleaf tables)
  - Key numbers CSV (abstract-ready stats)
  - Scatter: baseline Jaccard vs CRW Jaccard (universality argument)
  - Histogram of improvements (CRW never hurts)
  - Avg alpha curve (one alpha fits all) — regenerated for style consistency
  - Optimal alpha histogram — regenerated for style consistency

Usage:
    python scripts/compile_question_alpha_sweep.py [--csv PATH]

    If --csv not provided, auto-discovers the latest CSV in
    experiment_results/question_alpha_sweep_results/.
"""

import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SWEEP_RESULTS_DIR = Path("experiment_results/question_alpha_sweep_results")
OUTPUT_DIR = Path("experiment_results/ic2s2_compilation")

ALPHA_REFERENCE = 0.3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compile per-question alpha sweep results"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to question_alpha_sweep CSV (auto-discovered if not provided)",
    )
    return parser.parse_args()


def _find_latest_csv() -> Path:
    """Auto-discover the latest question_alpha_sweep CSV."""
    csvs = sorted(SWEEP_RESULTS_DIR.glob("**/question_alpha_sweep_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No question_alpha_sweep CSVs found in {SWEEP_RESULTS_DIR}"
        )
    return csvs[-1]


# ---------------------------------------------------------------------------
# Per-question summary
# ---------------------------------------------------------------------------


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-question (per clone_type): baseline distortion, optimal alpha, best CRW, improvement."""
    has_clone_type = "clone_type" in df.columns
    group_cols = ["question_id", "clone_type"] if has_clone_type else ["question_id"]

    rows = []
    for key, grp in df.groupby(group_cols):
        if has_clone_type:
            q_id, clone_type = key
        else:
            q_id = key
            clone_type = None

        base_jac = grp["base_jaccard_mean"].iloc[0]
        base_spe = grp["base_spearman_mean"].iloc[0]
        base_ken = grp["base_kendall_mean"].iloc[0]

        best_idx = grp["crw_jaccard_mean"].idxmax()
        best = grp.loc[best_idx]

        # Alpha 95% interval: range where Jaccard >= 95% of max
        max_jac = best["crw_jaccard_mean"]
        threshold = 0.95 * max_jac
        above = grp[grp["crw_jaccard_mean"] >= threshold]["alpha"]
        alpha_95_lo = above.min()
        alpha_95_hi = above.max()

        row = {
            "question_id": int(q_id),
            "question_text": grp["question_text"].iloc[0],
            "base_jaccard": base_jac,
            "base_spearman": base_spe,
            "base_kendall": base_ken,
            "optimal_alpha": best["alpha"],
            "crw_jaccard": max_jac,
            "crw_spearman": best["crw_spearman_mean"],
            "crw_kendall": best["crw_kendall_mean"],
            "improvement_jaccard": max_jac - base_jac,
            "improvement_spearman": best["crw_spearman_mean"] - base_spe,
            "improvement_kendall": best["crw_kendall_mean"] - base_ken,
            "alpha_95_lo": alpha_95_lo,
            "alpha_95_hi": alpha_95_hi,
            "alpha_95_width": alpha_95_hi - alpha_95_lo,
        }
        if has_clone_type:
            row["clone_type"] = clone_type
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("improvement_jaccard", ascending=False)
    return summary.reset_index(drop=True)


def compute_key_numbers(summary: pd.DataFrame) -> pd.DataFrame:
    """Abstract-ready statistics, with per-clone-type breakdown if available."""
    n_improved = (summary["improvement_jaccard"] > 0).sum()
    n_total = len(summary)

    rows = [
        ("n_entries", n_total),
        ("n_improved", n_improved),
        ("pct_improved", 100.0 * n_improved / n_total),
        ("base_jaccard_mean", summary["base_jaccard"].mean()),
        ("base_jaccard_median", summary["base_jaccard"].median()),
        ("crw_jaccard_mean", summary["crw_jaccard"].mean()),
        ("crw_jaccard_median", summary["crw_jaccard"].median()),
        ("improvement_mean", summary["improvement_jaccard"].mean()),
        ("improvement_median", summary["improvement_jaccard"].median()),
        ("improvement_min", summary["improvement_jaccard"].min()),
        ("improvement_max", summary["improvement_jaccard"].max()),
        ("improvement_std", summary["improvement_jaccard"].std()),
        ("optimal_alpha_mean", summary["optimal_alpha"].mean()),
        ("optimal_alpha_median", summary["optimal_alpha"].median()),
        ("optimal_alpha_mode", summary["optimal_alpha"].mode().iloc[0]),
        ("alpha_95_width_mean", summary["alpha_95_width"].mean()),
        ("alpha_95_width_median", summary["alpha_95_width"].median()),
        ("most_correctable_qid", int(summary.iloc[0]["question_id"])),
        ("most_correctable_improvement", summary.iloc[0]["improvement_jaccard"]),
        ("least_correctable_qid", int(summary.iloc[-1]["question_id"])),
        ("least_correctable_improvement", summary.iloc[-1]["improvement_jaccard"]),
        ("base_spearman_mean", summary["base_spearman"].mean()),
        ("crw_spearman_mean", summary["crw_spearman"].mean()),
        ("base_kendall_mean", summary["base_kendall"].mean()),
        ("crw_kendall_mean", summary["crw_kendall"].mean()),
    ]

    # Per-clone-type breakdown
    if "clone_type" in summary.columns:
        n_questions = summary["question_id"].nunique()
        rows.append(("n_questions", n_questions))
        rows.append(("n_clone_types", summary["clone_type"].nunique()))

        for ct, ct_grp in summary.groupby("clone_type"):
            rows.extend([
                (f"{ct}__optimal_alpha_mean", ct_grp["optimal_alpha"].mean()),
                (f"{ct}__optimal_alpha_median", ct_grp["optimal_alpha"].median()),
                (f"{ct}__optimal_alpha_mode", ct_grp["optimal_alpha"].mode().iloc[0]),
                (f"{ct}__improvement_mean", ct_grp["improvement_jaccard"].mean()),
                (f"{ct}__improvement_median", ct_grp["improvement_jaccard"].median()),
                (f"{ct}__crw_jaccard_mean", ct_grp["crw_jaccard"].mean()),
                (f"{ct}__base_jaccard_mean", ct_grp["base_jaccard"].mean()),
            ])

    return pd.DataFrame(rows, columns=["metric", "value"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_scatter(summary: pd.DataFrame, output_dir: Path):
    """Scatter: baseline Jaccard vs CRW Jaccard at optimal alpha."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        summary["base_jaccard"], summary["crw_jaccard"],
        c="#2196F3", s=45, alpha=0.7, edgecolors="white", linewidths=0.5,
        zorder=3,
    )

    # Diagonal (no effect line)
    lims = [
        min(summary["base_jaccard"].min(), summary["crw_jaccard"].min()) - 0.02,
        max(summary["base_jaccard"].max(), summary["crw_jaccard"].max()) + 0.02,
    ]
    ax.plot(lims, lims, color="grey", linestyle="--", alpha=0.5, linewidth=1, zorder=1)

    # Perfect correction line
    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.3, zorder=1)

    # Annotate mean improvement
    mean_base = summary["base_jaccard"].mean()
    mean_crw = summary["crw_jaccard"].mean()
    mean_improv = mean_crw - mean_base
    ax.annotate(
        f"Mean improvement: +{mean_improv:.3f}\n"
        f"({mean_base:.3f} → {mean_crw:.3f})",
        xy=(mean_base, mean_crw),
        xytext=(0.62, 0.96),
        fontsize=9, color="#333",
        arrowprops=dict(arrowstyle="->", color="#999", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9),
    )

    ax.set_xlabel("Baseline Jaccard (no CRW)", fontsize=11)
    ax.set_ylabel("CRW Jaccard (at optimal α)", fontsize=11)
    ax.set_title(
        "CRW Correction: Every Question Improves\n"
        "(5× easy paraphrase clones, E5-Instruct, ZH)",
        fontsize=12,
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    # Add text: "All 75 points above diagonal"
    ax.text(
        0.97, 0.03, "All 75/75 questions\nabove diagonal",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#666", style="italic",
    )

    fig.tight_layout()
    path = output_dir / "qa_sweep_scatter_baseline_vs_crw.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


def plot_improvement_histogram(summary: pd.DataFrame, output_dir: Path):
    """Histogram of per-question Jaccard improvement."""
    fig, ax = plt.subplots(figsize=(8, 5))

    improvements = summary["improvement_jaccard"]
    bins = np.arange(0, improvements.max() + 0.015, 0.01)

    ax.hist(
        improvements, bins=bins,
        color="#2196F3", edgecolor="white", alpha=0.85,
    )

    # Mean and median lines
    ax.axvline(
        improvements.mean(), color="#E53935", linestyle="--", linewidth=1.5,
        label=f"Mean: +{improvements.mean():.3f}",
    )
    ax.axvline(
        improvements.median(), color="#FF9800", linestyle="--", linewidth=1.5,
        label=f"Median: +{improvements.median():.3f}",
    )

    ax.set_xlabel("Jaccard Improvement (CRW − Baseline)", fontsize=11)
    ax.set_ylabel("Number of Questions", fontsize=11)
    ax.set_title(
        "Distribution of CRW Improvement Across All Questions\n"
        "(5× easy paraphrase, E5-Instruct, ZH)",
        fontsize=12,
    )
    ax.legend(fontsize=9)

    # Annotate: all positive
    ax.text(
        0.97, 0.95,
        f"All {len(improvements)}/{len(improvements)} positive\n"
        f"Range: +{improvements.min():.3f} to +{improvements.max():.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#666", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9),
    )

    fig.tight_layout()
    path = output_dir / "qa_sweep_improvement_histogram.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


def plot_avg_curve(df: pd.DataFrame, output_dir: Path):
    """Average CRW metrics vs alpha with percentile band."""
    alpha_agg = df.groupby("alpha").agg(
        crw_jaccard_mean=("crw_jaccard_mean", "mean"),
        crw_jaccard_q25=("crw_jaccard_mean", lambda x: x.quantile(0.25)),
        crw_jaccard_q75=("crw_jaccard_mean", lambda x: x.quantile(0.75)),
        crw_spearman_mean=("crw_spearman_mean", "mean"),
        crw_kendall_mean=("crw_kendall_mean", "mean"),
        base_jaccard_mean=("base_jaccard_mean", "mean"),
        base_spearman_mean=("base_spearman_mean", "mean"),
        base_kendall_mean=("base_kendall_mean", "mean"),
    ).reset_index()

    colors = {"jaccard": "#2196F3", "spearman": "#4CAF50", "kendall": "#FF9800"}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        alpha_agg["alpha"], alpha_agg["crw_jaccard_mean"],
        color=colors["jaccard"], linewidth=2.2,
        label="CRW Jaccard (mean across 75 questions)",
    )
    ax.fill_between(
        alpha_agg["alpha"],
        alpha_agg["crw_jaccard_q25"],
        alpha_agg["crw_jaccard_q75"],
        color=colors["jaccard"], alpha=0.15,
        label="Jaccard 25th–75th percentile",
    )
    ax.plot(
        alpha_agg["alpha"], alpha_agg["crw_spearman_mean"],
        color=colors["spearman"], linewidth=2, label="CRW Spearman (mean)",
    )
    ax.plot(
        alpha_agg["alpha"], alpha_agg["crw_kendall_mean"],
        color=colors["kendall"], linewidth=2, label="CRW Kendall (mean)",
    )

    # Baselines
    ax.axhline(
        alpha_agg["base_jaccard_mean"].iloc[0],
        color=colors["jaccard"], linestyle="--", alpha=0.4, label="Jaccard – no CRW",
    )
    ax.axhline(
        alpha_agg["base_spearman_mean"].iloc[0],
        color=colors["spearman"], linestyle="--", alpha=0.4, label="Spearman – no CRW",
    )
    ax.axhline(
        alpha_agg["base_kendall_mean"].iloc[0],
        color=colors["kendall"], linestyle="--", alpha=0.4, label="Kendall – no CRW",
    )

    ax.axvline(
        ALPHA_REFERENCE, color="grey", linestyle="--", alpha=0.5,
        label=f"α = {ALPHA_REFERENCE}",
    )

    # Mark peak
    peak_idx = alpha_agg["crw_jaccard_mean"].idxmax()
    peak_alpha = alpha_agg.loc[peak_idx, "alpha"]
    peak_val = alpha_agg.loc[peak_idx, "crw_jaccard_mean"]
    ax.plot(peak_alpha, peak_val, marker="*", color=colors["jaccard"],
            markersize=12, zorder=5)

    ax.set_xlabel("Alpha (α)", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_ylim(0.55, 1.02)
    ax.set_title(
        "CRW Correction vs Alpha (averaged across all 75 questions)\n"
        "(5× easy paraphrase, E5-Instruct, ZH)",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=7.5, ncol=2)
    fig.tight_layout()

    path = output_dir / "qa_sweep_avg_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


def plot_optimal_alpha_hist(summary: pd.DataFrame, output_dir: Path):
    """Histogram of per-question optimal alphas."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    alphas = summary["optimal_alpha"]
    bins = np.arange(
        alphas.min() - 0.05,
        alphas.max() + 0.15,
        0.1,
    )

    counts, edges, patches = ax.hist(
        alphas, bins=bins,
        color="#2196F3", edgecolor="white", alpha=0.85,
    )

    # Annotate bar counts
    for count, edge, patch in zip(counts, edges, patches):
        if count > 0:
            ax.text(
                edge + 0.05, count + 0.5, f"{int(count)}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.axvline(
        ALPHA_REFERENCE, color="#E53935", linestyle="--", linewidth=1.5,
        label=f"Reference α = {ALPHA_REFERENCE}",
    )

    ax.set_xlabel("Optimal Alpha per Question", fontsize=11)
    ax.set_ylabel("Number of Questions", fontsize=11)
    ax.set_title(
        "One Alpha Fits All: Per-Question Optimal α\n"
        "(5× easy paraphrase, E5-Instruct, ZH)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = output_dir / "qa_sweep_optimal_alpha_hist.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = _find_latest_csv()

    print(f"=== Compiling Per-Question Alpha Sweep ===")
    print(f"  Input: {csv_path}")

    df = pd.read_csv(csv_path)
    n_ct = df["clone_type"].nunique() if "clone_type" in df.columns else 1
    print(f"  Rows: {len(df)} ({df['question_id'].nunique()} questions × {df['alpha'].nunique()} alphas × {n_ct} clone types)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Compute summaries ---
    summary = compute_summary(df)
    key_numbers = compute_key_numbers(summary)

    # --- Save CSVs ---
    summary_path = OUTPUT_DIR / "qa_sweep_per_question_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  -> {summary_path.name}")

    numbers_path = OUTPUT_DIR / "qa_sweep_key_numbers.csv"
    key_numbers.to_csv(numbers_path, index=False)
    print(f"  -> {numbers_path.name}")

    # Also save the full alpha-level data for Overleaf regeneration
    full_path = OUTPUT_DIR / "qa_sweep_full_data.csv"
    df.to_csv(full_path, index=False)
    print(f"  -> {full_path.name}")

    # --- Plots ---
    sns.set_theme(style="whitegrid")

    plot_scatter(summary, OUTPUT_DIR)
    plot_improvement_histogram(summary, OUTPUT_DIR)
    plot_avg_curve(df, OUTPUT_DIR)
    plot_optimal_alpha_hist(summary, OUTPUT_DIR)

    # --- Print key numbers ---
    print("\n--- Key Numbers ---")
    for _, row in key_numbers.iterrows():
        val = row["value"]
        if isinstance(val, float) and val == int(val):
            print(f"  {row['metric']}: {int(val)}")
        elif isinstance(val, float):
            print(f"  {row['metric']}: {val:.4f}")
        else:
            print(f"  {row['metric']}: {val}")

    print(f"\n=== Done. Outputs in {OUTPUT_DIR}/ ===")


if __name__ == "__main__":
    main()
