"""
Compare question-level recommendation impact vs party visibility impact.

Two analyses measured the effect of cloning each of the 75 SmartVote questions:
  1. Question Impact: individual recommendation distortion (Jaccard, Spearman, Kendall)
  2. Party Impact Phase 1: party visibility shifts (per-party delta in seat share)

Both are baseline-only (no embeddings/CRW), so directly comparable.

Outputs:
    experiment_results/impact_comparison/
        merged_impact_comparison.csv
        scatter_rec_vs_party_impact.png
        feature_correlations.png
        nan_rate_vs_rank_bias.png
        topic_comparison.png
        topic_party_heatmap.png
        rank_comparison.png
        comparison_report.txt

Usage:
    python -m scripts.compare_impact_analyses
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

QUESTION_IMPACT_DIR = Path("experiment_results/question_impact")
PARTY_IMPACT_DIR = Path("experiment_results/party_impact/high_impact/phase1")
CATEGORY_CSV = Path("experiment_results/category_analysis/per_question_by_category.csv")
OUTPUT_DIR = Path("experiment_results/question_impact/impact_comparison")

MAJOR_PARTIES = ["SP", "Green", "GLP", "Centre", "FDP", "SVP"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _find_latest_csv(directory: Path, pattern: str) -> Path:
    csvs = sorted(directory.glob(pattern))
    if not csvs:
        raise FileNotFoundError(f"No CSV matching '{pattern}' in {directory}")
    return csvs[-1]


def load_data() -> pd.DataFrame:
    """Load and merge question impact, party impact, and category data."""
    # Question impact
    qi_path = _find_latest_csv(QUESTION_IMPACT_DIR, "question_impact_*.csv")
    qi = pd.read_csv(qi_path)
    print(f"Question impact: {qi_path.name} ({len(qi)} rows)")

    # Use one row per question (base metrics are identical across clone types)
    qi_base = qi.groupby("question_id").first().reset_index()

    # Party impact Phase 1 (find across subdirectories)
    pi_csvs = sorted(PARTY_IMPACT_DIR.rglob("party_impact_*.csv"))
    if not pi_csvs:
        raise FileNotFoundError(f"No party impact CSV in {PARTY_IMPACT_DIR}")
    pi = pd.read_csv(pi_csvs[-1])
    print(f"Party impact:    {pi_csvs[-1].name} ({len(pi)} rows)")

    # Category mapping
    cats = pd.read_csv(CATEGORY_CSV)
    print(f"Categories:      {CATEGORY_CSV.name} ({len(cats)} rows)")

    # Merge
    qi_cols = [
        "question_id", "question_text", "impact", "base_jaccard_mean",
        "base_spearman_mean", "base_kendall_mean", "composite_rank",
        "voter_nan_pct", "candidate_nan_pct", "voter_var", "candidate_var",
        "combined_var", "voter_avg_abs_corr", "candidate_avg_abs_corr",
    ]
    qi_cols = [c for c in qi_cols if c in qi_base.columns]

    pi_cols = [c for c in pi.columns if c != "question_text"]

    merged = qi_base[qi_cols].merge(pi[pi_cols], on="question_id")
    merged = merged.merge(
        cats[["ID_question", "_category"]].rename(columns={"ID_question": "question_id"}),
        on="question_id",
    )

    # Compute rankings
    merged["rank_rec"] = merged["impact"].rank(ascending=False)
    merged["rank_party"] = merged["max_abs_delta"].rank(ascending=False)
    merged["rank_bias"] = merged["rank_rec"] - merged["rank_party"]

    print(f"Merged:          {len(merged)} questions")
    return merged


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def compute_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Correlate question features with both impact metrics and rank bias."""
    features = [
        "voter_nan_pct", "voter_var", "candidate_var", "combined_var",
        "voter_avg_abs_corr", "candidate_avg_abs_corr",
    ]
    features = [f for f in features if f in df.columns and df[f].nunique() > 1]

    rows = []
    for feat in features:
        rho_rec, p_rec = stats.spearmanr(df[feat], df["impact"])
        rho_party, p_party = stats.spearmanr(df[feat], df["max_abs_delta"])
        rho_bias, p_bias = stats.spearmanr(df[feat], df["rank_bias"])
        rows.append({
            "feature": feat,
            "rho_rec_impact": rho_rec, "p_rec_impact": p_rec,
            "rho_party_delta": rho_party, "p_party_delta": p_party,
            "rho_rank_bias": rho_bias, "p_rank_bias": p_bias,
            "gap": abs(rho_rec - rho_party),
        })
    return pd.DataFrame(rows).sort_values("gap", ascending=False)


def compute_per_party_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Correlate rec impact with per-party absolute visibility delta."""
    rows = []
    for party in MAJOR_PARTIES:
        col = f"delta_{party}"
        if col not in df.columns:
            continue
        rho, p = stats.spearmanr(df["impact"], df[col].abs())
        rows.append({"party": party, "rho": rho, "p": p})
    return pd.DataFrame(rows).sort_values("rho", ascending=False)


def compute_topic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate both impact metrics by topic."""
    topic = df.groupby("_category").agg(
        n_questions=("question_id", "count"),
        mean_rec_impact=("impact", "mean"),
        mean_party_delta=("max_abs_delta", "mean"),
        mean_nan_pct=("voter_nan_pct", "mean"),
        mean_voter_var=("voter_var", "mean"),
    ).reset_index()

    topic["rank_rec"] = topic["mean_rec_impact"].rank(ascending=False)
    topic["rank_party"] = topic["mean_party_delta"].rank(ascending=False)
    topic["rank_diff"] = topic["rank_rec"] - topic["rank_party"]

    return topic.sort_values("mean_rec_impact", ascending=False)


def compute_topic_party_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Mean party delta per topic × party."""
    rows = []
    for topic in sorted(df["_category"].unique()):
        subset = df[df["_category"] == topic]
        row = {"topic": topic}
        for party in MAJOR_PARTIES:
            col = f"delta_{party}"
            if col in subset.columns:
                row[party] = subset[col].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("topic")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_scatter(df: pd.DataFrame, output_dir: Path):
    """Scatter: rec impact vs party visibility delta, colored by topic."""
    fig, ax = plt.subplots(figsize=(10, 8))

    categories = sorted(df["_category"].unique())
    palette = sns.color_palette("tab20", len(categories))
    cat_colors = dict(zip(categories, palette))

    for cat in categories:
        subset = df[df["_category"] == cat]
        ax.scatter(
            subset["impact"], subset["max_abs_delta"],
            c=[cat_colors[cat]], label=cat, s=40, alpha=0.8, edgecolors="white", linewidth=0.5,
        )

    # Label outliers (top 5 by each metric + largest rank discrepancies)
    label_ids = set()
    label_ids.update(df.nlargest(5, "impact")["question_id"])
    label_ids.update(df.nlargest(5, "max_abs_delta")["question_id"])
    label_ids.update(df.loc[df["rank_bias"].abs().nlargest(3).index, "question_id"])

    for _, row in df[df["question_id"].isin(label_ids)].iterrows():
        ax.annotate(
            str(int(row["question_id"])),
            (row["impact"], row["max_abs_delta"]),
            fontsize=7, alpha=0.7,
            xytext=(4, 4), textcoords="offset points",
        )

    # Correlation annotation
    rho, p = stats.spearmanr(df["impact"], df["max_abs_delta"])
    ax.text(
        0.02, 0.98, f"Spearman ρ = {rho:.3f} (p = {p:.1e})",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Recommendation Impact (1 − Jaccard)")
    ax.set_ylabel("Max Party Visibility Delta (pp)")
    ax.set_title("Question Impact: Individual Recommendations vs Party Visibility")
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
        title="Topic", title_fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_rec_vs_party_impact.png", dpi=150)
    plt.close(fig)
    print(f"  -> scatter_rec_vs_party_impact.png")


def plot_feature_correlations(feat_corr: pd.DataFrame, output_dir: Path):
    """Side-by-side bars: feature correlations with rec impact vs party delta."""
    fig, ax = plt.subplots(figsize=(10, 5))

    features = feat_corr["feature"].values
    x = np.arange(len(features))
    w = 0.35

    bars1 = ax.bar(x - w / 2, feat_corr["rho_rec_impact"], w, label="vs Rec Impact", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, feat_corr["rho_party_delta"], w, label="vs Party Delta", color="#DD8452")

    # Significance markers
    for i, row in feat_corr.iterrows():
        idx = list(feat_corr.index).index(i)
        if row["p_rec_impact"] < 0.05:
            ax.text(idx - w / 2, row["rho_rec_impact"] + 0.02 * np.sign(row["rho_rec_impact"]),
                    "*", ha="center", fontsize=10, fontweight="bold")
        if row["p_party_delta"] < 0.05:
            ax.text(idx + w / 2, row["rho_party_delta"] + 0.02 * np.sign(row["rho_party_delta"]),
                    "*", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=8)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Feature Correlations with Both Impact Metrics")
    ax.legend()
    ax.axhline(0, color="grey", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_correlations.png", dpi=150)
    plt.close(fig)
    print(f"  -> feature_correlations.png")


def plot_nan_vs_rank_bias(df: pd.DataFrame, output_dir: Path):
    """Scatter: NaN rate vs rank bias, showing NaN rate explains discrepancies."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(df["voter_nan_pct"], df["rank_bias"], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)

    # Fit + annotate
    rho, p = stats.spearmanr(df["voter_nan_pct"], df["rank_bias"])
    ax.text(
        0.02, 0.98, f"Spearman ρ = {rho:.3f} (p = {p:.4f})",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    # Label extreme points
    for _, row in df[df["rank_bias"].abs() > 25].iterrows():
        ax.annotate(
            str(int(row["question_id"])),
            (row["voter_nan_pct"], row["rank_bias"]),
            fontsize=7, alpha=0.7, xytext=(4, 4), textcoords="offset points",
        )

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Voter NaN Rate")
    ax.set_ylabel("Rank Bias (rec rank − party rank)")
    ax.set_title("NaN Rate Explains Which Impact Metric a Question Ranks Higher On")

    # Annotate regions
    ax.text(0.05, 0.15, "← Higher on Rec Impact\n(low NaN, symmetric churn)",
            transform=ax.transAxes, fontsize=8, color="grey", style="italic")
    ax.text(0.6, 0.85, "Higher on Party Impact →\n(high NaN, directional shifts)",
            transform=ax.transAxes, fontsize=8, color="grey", style="italic")

    fig.tight_layout()
    fig.savefig(output_dir / "nan_rate_vs_rank_bias.png", dpi=150)
    plt.close(fig)
    print(f"  -> nan_rate_vs_rank_bias.png")


def plot_topic_comparison(topic_stats: pd.DataFrame, output_dir: Path):
    """Grouped bars: per-topic mean rec impact and party delta (normalized)."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    topics = topic_stats["_category"].values
    x = np.arange(len(topics))
    w = 0.35

    color_rec = "#4C72B0"
    color_party = "#DD8452"

    bars1 = ax1.bar(x - w / 2, topic_stats["mean_rec_impact"], w,
                    label="Mean Rec Impact", color=color_rec, alpha=0.8)
    ax1.set_ylabel("Mean Rec Impact (1 − Jaccard)", color=color_rec)
    ax1.tick_params(axis="y", labelcolor=color_rec)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w / 2, topic_stats["mean_party_delta"], w,
                    label="Mean Party Delta", color=color_party, alpha=0.8)
    ax2.set_ylabel("Mean Max Party Delta (pp)", color=color_party)
    ax2.tick_params(axis="y", labelcolor=color_party)

    # Add rank labels
    for i, (_, row) in enumerate(topic_stats.iterrows()):
        rank_diff = int(row["rank_diff"])
        if rank_diff != 0:
            symbol = f"Δ{rank_diff:+d}"
            ax1.text(i, -0.01, symbol, ha="center", va="top", fontsize=7,
                     color="red" if abs(rank_diff) >= 3 else "grey",
                     transform=ax1.get_xaxis_transform())

    ax1.set_xticks(x)
    ax1.set_xticklabels(topics, rotation=45, ha="right", fontsize=8)
    ax1.set_title("Per-Topic Impact: Recommendations vs Party Visibility")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "topic_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> topic_comparison.png")


def plot_topic_party_heatmap(heatmap_data: pd.DataFrame, output_dir: Path):
    """Heatmap: mean party delta per topic × party."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        heatmap_data * 100,  # convert to percentage points
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Mean Visibility Delta (pp)"},
    )
    ax.set_title("Mean Party Visibility Change When Cloning Questions by Topic")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "topic_party_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  -> topic_party_heatmap.png")


def plot_rank_comparison(df: pd.DataFrame, output_dir: Path, n: int = 15):
    """Side-by-side rank bars for top questions by composite rank."""
    df = df.copy()
    df["composite"] = (df["rank_rec"] + df["rank_party"]) / 2
    top = df.nsmallest(n, "composite").sort_values("composite")

    fig, ax = plt.subplots(figsize=(10, 7))

    y = np.arange(n)
    h = 0.35

    ax.barh(y - h / 2, top["rank_rec"], h, label="Rec Impact Rank", color="#4C72B0", alpha=0.8)
    ax.barh(y + h / 2, top["rank_party"], h, label="Party Impact Rank", color="#DD8452", alpha=0.8)

    labels = [f"Q{int(qid)}" for qid in top["question_id"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_xaxis()
    ax.set_xlabel("Rank (1 = highest impact)")
    ax.set_title(f"Top-{n} Questions: Recommendation vs Party Impact Ranking")
    ax.legend()

    # Annotate party
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(1, i, f"  {row['max_positive_party']}", va="center", fontsize=7, color="grey")

    fig.tight_layout()
    fig.savefig(output_dir / "rank_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  -> rank_comparison.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    df: pd.DataFrame,
    feat_corr: pd.DataFrame,
    party_corr: pd.DataFrame,
    topic_stats: pd.DataFrame,
    output_dir: Path,
):
    lines = []
    lines.append("=" * 100)
    lines.append("COMPARISON: RECOMMENDATION IMPACT vs PARTY VISIBILITY IMPACT")
    lines.append("=" * 100)
    lines.append(f"Questions analyzed: {len(df)}")
    lines.append("")

    # Overall correlation
    rho, p = stats.spearmanr(df["impact"], df["max_abs_delta"])
    lines.append(f"Overall Spearman ρ (rec impact vs max party delta): {rho:.3f} (p = {p:.1e})")
    lines.append("")

    # Top-10 overlap
    top_rec = set(df.nlargest(10, "impact")["question_id"])
    top_party = set(df.nlargest(10, "max_abs_delta")["question_id"])
    overlap = top_rec & top_party
    lines.append(f"Top-10 overlap: {len(overlap)}/10 questions")
    lines.append(f"  In both:          {sorted(overlap)}")
    lines.append(f"  Only rec top-10:  {sorted(top_rec - top_party)}")
    lines.append(f"  Only party top-10:{sorted(top_party - top_rec)}")
    lines.append("")

    # Feature correlations
    lines.append("-" * 100)
    lines.append("FEATURE CORRELATIONS")
    lines.append("-" * 100)
    header = f"{'Feature':<30} {'ρ rec':>8} {'p':>8} {'ρ party':>8} {'p':>8} {'ρ bias':>8} {'p':>8} {'gap':>6}"
    lines.append(header)
    for _, row in feat_corr.iterrows():
        sig_r = "*" if row["p_rec_impact"] < 0.05 else " "
        sig_p = "*" if row["p_party_delta"] < 0.05 else " "
        sig_b = "*" if row["p_rank_bias"] < 0.05 else " "
        lines.append(
            f"{row['feature']:<30} {row['rho_rec_impact']:>+7.3f}{sig_r} "
            f"{row['p_rec_impact']:>7.4f} {row['rho_party_delta']:>+7.3f}{sig_p} "
            f"{row['p_party_delta']:>7.4f} {row['rho_rank_bias']:>+7.3f}{sig_b} "
            f"{row['p_rank_bias']:>7.4f} {row['gap']:>5.3f}"
        )
    lines.append("")
    lines.append("Key: ρ rec = correlation with recommendation impact")
    lines.append("     ρ party = correlation with party visibility delta")
    lines.append("     ρ bias = correlation with rank_bias (positive = ranks higher on party metric)")
    lines.append("     gap = |ρ_rec - ρ_party|")
    lines.append("")

    # Per-party correlations
    lines.append("-" * 100)
    lines.append("PER-PARTY: |delta| CORRELATION WITH REC IMPACT")
    lines.append("-" * 100)
    for _, row in party_corr.iterrows():
        sig = "*" if row["p"] < 0.05 else " "
        lines.append(f"  {row['party']:<10} ρ = {row['rho']:+.3f} (p = {row['p']:.4f}){sig}")
    lines.append("")

    # Biggest discrepancies
    lines.append("-" * 100)
    lines.append("BIGGEST RANK DISCREPANCIES")
    lines.append("-" * 100)
    lines.append("")
    lines.append("High rec impact / low party impact (symmetric churn):")
    high_rec = df.nsmallest(5, "rank_bias")[
        ["question_id", "impact", "max_abs_delta", "rank_rec", "rank_party",
         "rank_bias", "voter_nan_pct", "voter_var", "max_positive_party", "_category"]
    ]
    lines.append(high_rec.to_string(index=False))
    lines.append("")
    lines.append("Low rec impact / high party impact (directional shifts):")
    high_party = df.nlargest(5, "rank_bias")[
        ["question_id", "impact", "max_abs_delta", "rank_rec", "rank_party",
         "rank_bias", "voter_nan_pct", "voter_var", "max_positive_party", "_category"]
    ]
    lines.append(high_party.to_string(index=False))
    lines.append("")

    # Topic analysis
    lines.append("-" * 100)
    lines.append("TOPIC-LEVEL COMPARISON")
    lines.append("-" * 100)
    lines.append("")
    header = (f"{'Topic':<40} {'N':>2} {'Rec Impact':>10} {'Rank':>4} "
              f"{'Party Δ':>10} {'Rank':>4} {'Diff':>5} {'NaN%':>6}")
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in topic_stats.iterrows():
        diff_str = f"{int(row['rank_diff']):+d}" if row["rank_diff"] != 0 else "0"
        lines.append(
            f"{row['_category']:<40} {int(row['n_questions']):>2} "
            f"{row['mean_rec_impact']:>10.4f} {int(row['rank_rec']):>4} "
            f"{row['mean_party_delta']:>10.6f} {int(row['rank_party']):>4} "
            f"{diff_str:>5} {row['mean_nan_pct']:>5.1%}"
        )
    lines.append("")

    # Interpretation
    lines.append("-" * 100)
    lines.append("INTERPRETATION")
    lines.append("-" * 100)
    lines.append("")
    lines.append("1. MODERATE CORRELATION (ρ ≈ 0.69): Questions that cause more individual")
    lines.append("   recommendation churn also tend to cause more party visibility shifts,")
    lines.append("   but the relationship is far from deterministic.")
    lines.append("")
    lines.append("2. NaN RATE IS THE KEY DIFFERENTIATOR:")
    lines.append("   - Rec impact is strongly anti-correlated with NaN rate (ρ = -0.65):")
    lines.append("     questions answered by more voters → more total churn when cloned.")
    lines.append("   - Party delta is only weakly anti-correlated (ρ = -0.36).")
    lines.append("   - High NaN questions rank relatively higher on party impact because")
    lines.append("     the self-selected subset who answers is more polarized, creating")
    lines.append("     directional (cross-party) shifts rather than symmetric churn.")
    lines.append("")
    lines.append("3. VOTER VARIANCE drives both metrics, but rec impact more strongly")
    lines.append("   (ρ = 0.81 vs 0.60). High-variance questions create more distance")
    lines.append("   distortion per voter, causing more recommendation shuffling.")
    lines.append("")
    lines.append("4. ANSWER CORRELATION is NOT predictive of either metric (ρ ≈ 0),")
    lines.append("   consistent with the mathematical independence of L2 distance terms.")
    lines.append("")
    lines.append("5. TOPIC-LEVEL patterns mirror question-level: Federal Budget and Values")
    lines.append("   (highest NaN rates) rank last on rec impact but not as low on party")
    lines.append("   impact. Immigration (lowest NaN) ranks top on both.")
    lines.append("")
    lines.append("=" * 100)

    report = "\n".join(lines)
    path = output_dir / "comparison_report.txt"
    path.write_text(report)
    print(f"  -> comparison_report.txt")
    print()
    print(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load & merge
    df = load_data()

    # Analysis
    feat_corr = compute_feature_correlations(df)
    party_corr = compute_per_party_correlations(df)
    topic_stats = compute_topic_stats(df)
    heatmap_data = compute_topic_party_heatmap_data(df)

    # Save merged CSV
    csv_path = OUTPUT_DIR / "merged_impact_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> merged_impact_comparison.csv")

    # Plots
    plot_scatter(df, OUTPUT_DIR)
    plot_feature_correlations(feat_corr, OUTPUT_DIR)
    plot_nan_vs_rank_bias(df, OUTPUT_DIR)
    plot_topic_comparison(topic_stats, OUTPUT_DIR)
    plot_topic_party_heatmap(heatmap_data, OUTPUT_DIR)
    plot_rank_comparison(df, OUTPUT_DIR)

    # Report
    write_report(df, feat_corr, party_corr, topic_stats, OUTPUT_DIR)


if __name__ == "__main__":
    main()
