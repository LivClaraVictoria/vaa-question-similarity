"""
Analyze intra-topic vs inter-topic distances for E5-INSTRUCT and ANSWER-CORRELATION-ARCCOS.

For each metric, computes per-topic intra-topic distance statistics, per-topic-pair
inter-topic distance statistics, and overall intra vs inter comparison with effect sizes.

Outputs:
    experiment_results/topic_distances/e5_instruct/         — per-metric analysis
    experiment_results/topic_distances/answer_corr_arccos/   — per-metric analysis
    experiment_results/topic_distances/comparison/            — cross-metric comparison

Usage:
    python -m scripts.analyze_topic_distances
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/cleaned")
CACHE_DIR = Path("cache/distance_calculations")
OUTPUT_DIR = Path("experiment_results/topic_distances")

METRICS = {
    "e5_instruct": {
        "label": "E5-INSTRUCT",
        "path": CACHE_DIR / "dist_2023_E5-INSTRUCT_ba053f9f59a3.parquet",
    },
    "answer_corr_arccos": {
        "label": "ANSWER-CORRELATION-ARCCOS",
        "path": CACHE_DIR / "dist_2023_ANSWER-CORRELATION-ARCCOS_790a61671ea5.parquet",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_distances(path: Path) -> pd.DataFrame:
    """Load distance parquet, normalize Similarity -> Distance if needed."""
    df = pd.read_parquet(path)
    if "Similarity" in df.columns and "Distance" not in df.columns:
        df["Distance"] = np.sqrt(np.maximum(0, 2 * (1 - df["Similarity"])))
    # Keep only original questions
    df = df[(df["ID1"] < 9_000_000) & (df["ID2"] < 9_000_000)]
    return df


def to_pair_dict(df: pd.DataFrame) -> dict:
    """Convert distance DataFrame to {(min_id, max_id): distance} dict."""
    result = {}
    for _, row in df.iterrows():
        id1, id2 = int(row["ID1"]), int(row["ID2"])
        key = (min(id1, id2), max(id1, id2))
        result[key] = row["Distance"]
    return result


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation."""
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return (b.mean() - a.mean()) / np.sqrt(pooled_var)


# ---------------------------------------------------------------------------
# Per-metric analysis
# ---------------------------------------------------------------------------

def compute_intra_topic(pairs: dict, cat_map: dict, topic_questions: dict) -> pd.DataFrame:
    """Compute per-topic intra-topic distance statistics."""
    rows = []
    for topic, qids in sorted(topic_questions.items()):
        dists = []
        for i, id1 in enumerate(qids):
            for id2 in qids[i + 1:]:
                key = (min(id1, id2), max(id1, id2))
                if key in pairs:
                    dists.append(pairs[key])
        if not dists:
            continue
        d = np.array(dists)
        rows.append({
            "topic": topic,
            "n_questions": len(qids),
            "n_pairs": len(dists),
            "mean": d.mean(),
            "median": np.median(d),
            "std": d.std(ddof=1) if len(d) > 1 else 0.0,
            "min": d.min(),
            "max": d.max(),
            "q25": np.percentile(d, 25),
            "q75": np.percentile(d, 75),
        })
    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)


def compute_inter_topic(pairs: dict, cat_map: dict, topic_questions: dict) -> pd.DataFrame:
    """Compute per-topic-pair inter-topic distance statistics."""
    topics = sorted(topic_questions.keys())
    rows = []
    for i, t_a in enumerate(topics):
        for t_b in topics[i + 1:]:
            dists = []
            for id1 in topic_questions[t_a]:
                for id2 in topic_questions[t_b]:
                    key = (min(id1, id2), max(id1, id2))
                    if key in pairs:
                        dists.append(pairs[key])
            if not dists:
                continue
            d = np.array(dists)
            rows.append({
                "topic_a": t_a,
                "topic_b": t_b,
                "n_pairs": len(dists),
                "mean": d.mean(),
                "median": np.median(d),
                "std": d.std(ddof=1) if len(d) > 1 else 0.0,
                "min": d.min(),
                "max": d.max(),
            })
    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)


def compute_summary(pairs: dict, cat_map: dict, topic_questions: dict) -> dict:
    """Overall intra vs inter comparison with statistical tests."""
    intra_dists = []
    inter_dists = []
    for (id1, id2), dist in pairs.items():
        c1 = cat_map.get(id1)
        c2 = cat_map.get(id2)
        if c1 and c2:
            if c1 == c2:
                intra_dists.append(dist)
            else:
                inter_dists.append(dist)

    intra = np.array(intra_dists)
    inter = np.array(inter_dists)

    U, p_val = mannwhitneyu(intra, inter, alternative="less")
    n_intra, n_inter = len(intra), len(inter)
    auroc = U / (n_intra * n_inter)
    r_biserial = 2 * U / (n_intra * n_inter) - 1

    return {
        "intra_mean": intra.mean(),
        "intra_median": np.median(intra),
        "intra_std": intra.std(ddof=1),
        "intra_n_pairs": n_intra,
        "inter_mean": inter.mean(),
        "inter_median": np.median(inter),
        "inter_std": inter.std(ddof=1),
        "inter_n_pairs": n_inter,
        "mann_whitney_U": U,
        "mann_whitney_p": p_val,
        "cohens_d": cohens_d(intra, inter),
        "auroc": auroc,
        "rank_biserial_r": r_biserial,
    }


def build_heatmap_matrix(pairs: dict, cat_map: dict, topics_sorted: list,
                         topic_questions: dict) -> np.ndarray:
    """Build NxN topic distance heatmap (diagonal = intra, off-diagonal = inter)."""
    n = len(topics_sorted)
    mat = np.full((n, n), np.nan)
    topic_idx = {t: i for i, t in enumerate(topics_sorted)}

    for i, t_a in enumerate(topics_sorted):
        # Diagonal: intra-topic mean
        dists = []
        qids = topic_questions[t_a]
        for qi in range(len(qids)):
            for qj in range(qi + 1, len(qids)):
                key = (min(qids[qi], qids[qj]), max(qids[qi], qids[qj]))
                if key in pairs:
                    dists.append(pairs[key])
        if dists:
            mat[i, i] = np.mean(dists)

        # Off-diagonal: inter-topic means
        for j, t_b in enumerate(topics_sorted):
            if j <= i:
                continue
            dists = []
            for id1 in topic_questions[t_a]:
                for id2 in topic_questions[t_b]:
                    key = (min(id1, id2), max(id1, id2))
                    if key in pairs:
                        dists.append(pairs[key])
            if dists:
                mat[i, j] = np.mean(dists)
                mat[j, i] = np.mean(dists)

    return mat


def collect_intra_dists_by_topic(pairs: dict, topic_questions: dict) -> dict:
    """Return {topic: np.array of within-topic distances}."""
    result = {}
    for topic, qids in sorted(topic_questions.items()):
        dists = []
        for i, id1 in enumerate(qids):
            for id2 in qids[i + 1:]:
                key = (min(id1, id2), max(id1, id2))
                if key in pairs:
                    dists.append(pairs[key])
        if dists:
            result[topic] = np.array(dists)
    return result


def collect_all_inter_dists(pairs: dict, cat_map: dict) -> np.ndarray:
    """Return array of all cross-topic distances."""
    dists = []
    for (id1, id2), dist in pairs.items():
        c1 = cat_map.get(id1)
        c2 = cat_map.get(id2)
        if c1 and c2 and c1 != c2:
            dists.append(dist)
    return np.array(dists)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_boxplot(intra_by_topic: dict, inter_dists: np.ndarray,
                 metric_label: str, output_path: Path):
    """Horizontal box/strip plot: one row per topic + INTER-TOPIC."""
    # Sort topics by median intra distance
    sorted_topics = sorted(intra_by_topic.keys(),
                           key=lambda t: np.median(intra_by_topic[t]))

    labels = []
    data = []
    for t in sorted_topics:
        d = intra_by_topic[t]
        labels.append(f"{t} ({len(d)} pairs)")
        data.append(d)
    labels.append(f"INTER-TOPIC ({len(inter_dists)} pairs)")
    data.append(inter_dists)

    fig, ax = plt.subplots(figsize=(14, 8))
    n = len(data)
    y_positions = list(range(n))

    for i, (d, label) in enumerate(zip(data, labels)):
        is_inter = (i == n - 1)
        color = "#95a5a6" if is_inter else "#3498db"
        alpha = 0.6 if is_inter else 0.8

        if len(d) < 6:
            # Strip plot for small samples
            ax.scatter(d, [i] * len(d), color=color, alpha=alpha, s=40,
                       edgecolors="white", zorder=3)
            ax.plot([np.median(d)], [i], marker="|", color="red",
                    markersize=15, markeredgewidth=2, zorder=4)
        else:
            bp = ax.boxplot(d, positions=[i], vert=False, widths=0.5,
                            patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor=color, alpha=alpha),
                            medianprops=dict(color="red", linewidth=1.5),
                            whiskerprops=dict(color="gray"),
                            capprops=dict(color="gray"))

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"Distance ({metric_label})", fontsize=11)
    ax.set_title(f"Within-Topic vs Cross-Topic Distances — {metric_label}", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(mat: np.ndarray, topics: list, metric_label: str, output_path: Path):
    """Annotated heatmap of topic x topic mean distances."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(mat, cmap="viridis_r", aspect="equal")

    # Annotate cells
    for i in range(len(topics)):
        for j in range(len(topics)):
            val = mat[i, j]
            if np.isnan(val):
                continue
            fontweight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, fontweight=fontweight,
                    color="white" if val < np.nanmedian(mat) else "black")

    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=8)
    ax.set_title(f"Mean Pairwise Distance: Topic x Topic — {metric_label}", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Distance", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(e5_mat: np.ndarray, corr_mat: np.ndarray,
                    e5_topics: list, corr_topics: list,
                    e5_intra: pd.DataFrame, corr_intra: pd.DataFrame,
                    output_path: Path):
    """2x2 comparison: two heatmaps + scatter + rank bar chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # (0,0) E5 heatmap
    ax = axes[0, 0]
    im1 = ax.imshow(e5_mat, cmap="viridis_r", aspect="equal")
    for i in range(len(e5_topics)):
        for j in range(len(e5_topics)):
            val = e5_mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if val < np.nanmedian(e5_mat) else "black")
    ax.set_xticks(range(len(e5_topics)))
    ax.set_xticklabels(e5_topics, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(e5_topics)))
    ax.set_yticklabels(e5_topics, fontsize=6)
    ax.set_title("E5-INSTRUCT", fontsize=10)
    fig.colorbar(im1, ax=ax, shrink=0.7)

    # (0,1) CORR heatmap — same topic order as E5 for comparability
    ax = axes[0, 1]
    # Reorder corr_mat to match e5_topics order
    corr_topic_idx = {t: i for i, t in enumerate(corr_topics)}
    reorder = [corr_topic_idx[t] for t in e5_topics]
    corr_reordered = corr_mat[np.ix_(reorder, reorder)]

    im2 = ax.imshow(corr_reordered, cmap="viridis_r", aspect="equal")
    for i in range(len(e5_topics)):
        for j in range(len(e5_topics)):
            val = corr_reordered[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if val < np.nanmedian(corr_reordered) else "black")
    ax.set_xticks(range(len(e5_topics)))
    ax.set_xticklabels(e5_topics, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(e5_topics)))
    ax.set_yticklabels(e5_topics, fontsize=6)
    ax.set_title("ANSWER-CORRELATION-ARCCOS", fontsize=10)
    fig.colorbar(im2, ax=ax, shrink=0.7)

    # (1,0) Scatter: per-topic intra means
    ax = axes[1, 0]
    merged = e5_intra[["topic", "mean"]].merge(
        corr_intra[["topic", "mean"]], on="topic", suffixes=("_e5", "_corr"))
    ax.scatter(merged["mean_e5"], merged["mean_corr"], s=60, c="#3498db",
               edgecolors="white", zorder=3)
    for _, row in merged.iterrows():
        ax.annotate(row["topic"], (row["mean_e5"], row["mean_corr"]),
                     fontsize=6, ha="left", va="bottom",
                     xytext=(3, 3), textcoords="offset points")

    rho, p_val = spearmanr(merged["mean_e5"], merged["mean_corr"])
    ax.set_xlabel("E5-INSTRUCT intra-topic mean distance", fontsize=9)
    ax.set_ylabel("ANSWER-CORR-ARCCOS intra-topic mean distance", fontsize=9)
    ax.set_title(f"Per-Topic Intra Mean: E5 vs CORR (Spearman ρ={rho:.3f}, p={p_val:.3f})", fontsize=9)
    ax.grid(alpha=0.3)

    # (1,1) Paired rank bar chart
    ax = axes[1, 1]
    master = merged.copy()
    master["e5_rank"] = master["mean_e5"].rank().astype(int)
    master["corr_rank"] = master["mean_corr"].rank().astype(int)
    master = master.sort_values("e5_rank")

    y_pos = np.arange(len(master))
    bar_h = 0.35
    ax.barh(y_pos - bar_h / 2, master["e5_rank"], bar_h,
            label="E5-INSTRUCT rank", color="#3498db", edgecolor="white")
    ax.barh(y_pos + bar_h / 2, master["corr_rank"], bar_h,
            label="CORR-ARCCOS rank", color="#e74c3c", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(master["topic"].values, fontsize=7)
    ax.set_xlabel("Rank (1 = tightest cluster)", fontsize=9)
    ax.set_title("Topic Ranking: Intra-Topic Cohesion", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()

    fig.suptitle("E5-INSTRUCT vs ANSWER-CORRELATION-ARCCOS: Topic Distance Comparison",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_metric_report(metric_label: str, intra_df: pd.DataFrame,
                        inter_df: pd.DataFrame, summary: dict,
                        output_path: Path):
    """Write human-readable report for one metric."""
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"{'=' * 80}")
    p(f"TOPIC DISTANCE ANALYSIS — {metric_label}")
    p(f"{'=' * 80}")
    p(f"  Total intra-topic pairs: {summary['intra_n_pairs']}")
    p(f"  Total inter-topic pairs: {summary['inter_n_pairs']}")
    p()

    # Intra-topic table
    p(f"INTRA-TOPIC DISTANCES (sorted by mean, ascending = tightest cluster)")
    p(f"  {'Topic':<38} {'N_q':>3} {'Pairs':>5} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    p(f"  {'-' * 87}")
    for _, row in intra_df.iterrows():
        p(f"  {row['topic']:<38} {int(row['n_questions']):>3} {int(row['n_pairs']):>5} "
          f"{row['mean']:>8.4f} {row['median']:>8.4f} {row['std']:>8.4f} "
          f"{row['min']:>8.4f} {row['max']:>8.4f}")
    p()

    # Top-10 closest inter-topic pairs
    p(f"TOP-10 CLOSEST INTER-TOPIC PAIRS")
    p(f"  {'Topic A':<30} {'Topic B':<30} {'Pairs':>5} {'Mean':>8}")
    p(f"  {'-' * 75}")
    for _, row in inter_df.head(10).iterrows():
        p(f"  {row['topic_a']:<30} {row['topic_b']:<30} {int(row['n_pairs']):>5} {row['mean']:>8.4f}")
    p()

    # Top-10 most distant inter-topic pairs
    p(f"TOP-10 MOST DISTANT INTER-TOPIC PAIRS")
    p(f"  {'Topic A':<30} {'Topic B':<30} {'Pairs':>5} {'Mean':>8}")
    p(f"  {'-' * 75}")
    for _, row in inter_df.tail(10).iloc[::-1].iterrows():
        p(f"  {row['topic_a']:<30} {row['topic_b']:<30} {int(row['n_pairs']):>5} {row['mean']:>8.4f}")
    p()

    # Overall comparison
    p(f"OVERALL COMPARISON: INTRA vs INTER")
    p(f"  {'':>20} {'Intra':>12} {'Inter':>12}")
    p(f"  {'-' * 46}")
    p(f"  {'Mean':>20} {summary['intra_mean']:>12.4f} {summary['inter_mean']:>12.4f}")
    p(f"  {'Median':>20} {summary['intra_median']:>12.4f} {summary['inter_median']:>12.4f}")
    p(f"  {'Std':>20} {summary['intra_std']:>12.4f} {summary['inter_std']:>12.4f}")
    p(f"  {'N pairs':>20} {summary['intra_n_pairs']:>12} {summary['inter_n_pairs']:>12}")
    p()
    p(f"  Mann-Whitney U:     {summary['mann_whitney_U']:.1f}")
    p(f"  p-value (one-sided): {summary['mann_whitney_p']:.2e}")
    p(f"  Cohen's d:          {summary['cohens_d']:.4f}")
    p(f"  AUROC:              {summary['auroc']:.4f}")
    p(f"    (probability that random inter-topic dist > random intra-topic dist)")
    p(f"  Rank-biserial r:    {summary['rank_biserial_r']:.4f}")
    p()
    p(f"{'=' * 80}")

    output_path.write_text("\n".join(lines))


def write_comparison_report(master_df: pd.DataFrame,
                            e5_summary: dict, corr_summary: dict,
                            e5_inter: pd.DataFrame, corr_inter: pd.DataFrame,
                            output_path: Path):
    """Write comparison report between the two metrics."""
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"{'=' * 90}")
    p(f"COMPARISON: E5-INSTRUCT vs ANSWER-CORRELATION-ARCCOS")
    p(f"{'=' * 90}")
    p()

    # Side-by-side overall stats
    p(f"OVERALL INTRA vs INTER")
    p(f"  {'Metric':<30} {'Intra Mean':>11} {'Inter Mean':>11} {'Cohen d':>9} {'AUROC':>7} {'p-value':>12}")
    p(f"  {'-' * 82}")
    p(f"  {'E5-INSTRUCT':<30} {e5_summary['intra_mean']:>11.4f} {e5_summary['inter_mean']:>11.4f} "
      f"{e5_summary['cohens_d']:>9.4f} {e5_summary['auroc']:>7.4f} {e5_summary['mann_whitney_p']:>12.2e}")
    p(f"  {'ANSWER-CORR-ARCCOS':<30} {corr_summary['intra_mean']:>11.4f} {corr_summary['inter_mean']:>11.4f} "
      f"{corr_summary['cohens_d']:>9.4f} {corr_summary['auroc']:>7.4f} {corr_summary['mann_whitney_p']:>12.2e}")
    p()

    # Per-topic master table
    p(f"PER-TOPIC INTRA-TOPIC DISTANCES (sorted by E5 mean)")
    p(f"  {'Topic':<30} {'N_q':>3} {'E5 Mean':>8} {'E5 Rank':>8} {'Corr Mean':>10} {'Corr Rank':>10} {'Rank Diff':>10}")
    p(f"  {'-' * 82}")
    for _, row in master_df.sort_values("e5_rank").iterrows():
        p(f"  {row['topic']:<30} {int(row['n_questions']):>3} "
          f"{row['e5_intra_mean']:>8.4f} {int(row['e5_rank']):>8} "
          f"{row['corr_intra_mean']:>10.4f} {int(row['corr_rank']):>10} "
          f"{int(row['rank_diff']):>10}")
    p()

    # Spearman correlations
    rho_intra, p_intra = spearmanr(master_df["e5_intra_mean"], master_df["corr_intra_mean"])
    p(f"AGREEMENT BETWEEN METRICS")
    p(f"  Spearman ρ (per-topic intra means):     {rho_intra:.4f} (p={p_intra:.4f})")

    # Per-topic-pair agreement
    merged_inter = e5_inter[["topic_a", "topic_b", "mean"]].merge(
        corr_inter[["topic_a", "topic_b", "mean"]], on=["topic_a", "topic_b"],
        suffixes=("_e5", "_corr"))
    if len(merged_inter) > 2:
        rho_inter, p_inter = spearmanr(merged_inter["mean_e5"], merged_inter["mean_corr"])
        p(f"  Spearman ρ (per-topic-pair inter means): {rho_inter:.4f} (p={p_inter:.4f})")
    p()

    # Notable disagreements
    p(f"NOTABLE RANK DISAGREEMENTS (|rank_diff| >= 4)")
    disagree = master_df[master_df["rank_diff"].abs() >= 4].sort_values("rank_diff", key=abs, ascending=False)
    if len(disagree) == 0:
        p(f"  None — metrics largely agree on topic cohesion ranking")
    else:
        for _, row in disagree.iterrows():
            direction = "tighter in E5" if row["rank_diff"] > 0 else "tighter in CORR"
            p(f"  {row['topic']:<30} E5 rank={int(row['e5_rank'])}, CORR rank={int(row['corr_rank'])} ({direction})")
    p()
    p(f"{'=' * 90}")

    output_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load question metadata
    questions = pd.read_parquet(DATA_DIR / "df_questions.parquet")
    cat_map = questions.set_index("ID_question")["_category"].to_dict()
    topic_questions = questions.groupby("_category")["ID_question"].apply(sorted).to_dict()
    print(f"Loaded {len(questions)} questions across {len(topic_questions)} topics")

    # Store per-metric results for comparison
    all_intra = {}
    all_inter = {}
    all_summaries = {}
    all_heatmaps = {}
    all_topics_sorted = {}

    for metric_key, metric_info in METRICS.items():
        label = metric_info["label"]
        dist_path = metric_info["path"]
        out_dir = OUTPUT_DIR / metric_key

        print(f"\n{'=' * 60}")
        print(f"Processing {label}")
        print(f"{'=' * 60}")

        if not dist_path.exists():
            print(f"  ERROR: {dist_path} not found — skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        # Load distances
        dist_df = load_distances(dist_path)
        pairs = to_pair_dict(dist_df)
        print(f"  Loaded {len(pairs)} pairs")

        # Compute statistics
        intra_df = compute_intra_topic(pairs, cat_map, topic_questions)
        inter_df = compute_inter_topic(pairs, cat_map, topic_questions)
        summary = compute_summary(pairs, cat_map, topic_questions)

        # Store for comparison
        all_intra[metric_key] = intra_df
        all_inter[metric_key] = inter_df
        all_summaries[metric_key] = summary

        # Heatmap matrix (topics sorted by intra mean for this metric)
        topics_sorted = intra_df["topic"].tolist()
        all_topics_sorted[metric_key] = topics_sorted
        heatmap_mat = build_heatmap_matrix(pairs, cat_map, topics_sorted, topic_questions)
        all_heatmaps[metric_key] = heatmap_mat

        # Collect per-topic distance arrays for boxplot
        intra_by_topic = collect_intra_dists_by_topic(pairs, topic_questions)
        inter_dists = collect_all_inter_dists(pairs, cat_map)

        # Save CSVs
        intra_df.to_csv(out_dir / "topic_distances_intra.csv", index=False)
        inter_df.to_csv(out_dir / "topic_distances_inter.csv", index=False)
        pd.DataFrame([summary]).to_csv(out_dir / "topic_distances_summary.csv", index=False)

        # Plots
        plot_boxplot(intra_by_topic, inter_dists, label, out_dir / "topic_distances_boxplot.png")
        plot_heatmap(heatmap_mat, topics_sorted, label, out_dir / "topic_distances_heatmap.png")

        # Report
        write_metric_report(label, intra_df, inter_df, summary,
                            out_dir / "topic_distances_report.txt")

        print(f"  Saved to {out_dir}/")

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    if len(all_intra) < 2:
        print("\nSkipping comparison — need both metrics")
        return

    comp_dir = OUTPUT_DIR / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Generating comparison")
    print(f"{'=' * 60}")

    # Master CSV
    e5_intra = all_intra["e5_instruct"][["topic", "n_questions", "n_pairs", "mean", "median", "std"]].rename(
        columns={"mean": "e5_intra_mean", "median": "e5_intra_median",
                 "std": "e5_intra_std", "n_pairs": "n_intra_pairs"})
    corr_intra = all_intra["answer_corr_arccos"][["topic", "mean", "median", "std"]].rename(
        columns={"mean": "corr_intra_mean", "median": "corr_intra_median", "std": "corr_intra_std"})

    master = e5_intra.merge(corr_intra, on="topic")
    master["e5_rank"] = master["e5_intra_mean"].rank().astype(int)
    master["corr_rank"] = master["corr_intra_mean"].rank().astype(int)
    master["rank_diff"] = master["corr_rank"] - master["e5_rank"]
    master = master.sort_values("e5_intra_mean").reset_index(drop=True)

    master.to_csv(comp_dir / "topic_distances_master.csv", index=False)

    # Comparison plot
    plot_comparison(
        all_heatmaps["e5_instruct"], all_heatmaps["answer_corr_arccos"],
        all_topics_sorted["e5_instruct"], all_topics_sorted["answer_corr_arccos"],
        all_intra["e5_instruct"], all_intra["answer_corr_arccos"],
        comp_dir / "topic_distances_comparison.png",
    )

    # Comparison report
    write_comparison_report(
        master, all_summaries["e5_instruct"], all_summaries["answer_corr_arccos"],
        all_inter["e5_instruct"], all_inter["answer_corr_arccos"],
        comp_dir / "topic_distances_comparison_report.txt",
    )

    print(f"  Saved to {comp_dir}/")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
