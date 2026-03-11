"""
Validate embedding models against correlation-based ground truth.

Compares each embedding model's pairwise question distance matrix against
the ANSWER-CORRELATION distance matrix (1 - |Pearson_r| of voter answers).

For each model, computes Spearman rank correlation between the embedding
distances and the correlation distances across all question pairs.

Usage:
    python -m scripts.validate_embeddings_vs_correlation
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from main import load_config

CACHE_DIR = Path("cache/distance_calculations")
OUTPUT_DIR = Path("experiment_results/correlation_metric_results/correlation_analysis")

# All base ZH configs (one per embedding model)
MODEL_CONFIGS = {
    "SBERT": "configs/full_pipeline/base_data/pipeline_sbert_ZH.py",
    "E5": "configs/full_pipeline/base_data/pipeline_e5_ZH.py",
    "E5-ASYMMETRIC": "configs/full_pipeline/base_data/pipeline_e5_asym_ZH.py",
    "E5-INSTRUCT": "configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py",
    "E5-ASYMMETRIC-INSTRUCT": "configs/full_pipeline/base_data/pipeline_e5_asym_instruct_ZH.py",
    "JINA-V3": "configs/full_pipeline/base_data/pipeline_jina_v3_ZH.py",
    "BGE-M3": "configs/full_pipeline/base_data/pipeline_bge_m3_ZH.py",
    "GTE": "configs/full_pipeline/base_data/pipeline_gte_ZH.py",
    "NOMIC-V2": "configs/full_pipeline/base_data/pipeline_nomic_v2_ZH.py",
    "QWEN3": "configs/full_pipeline/base_data/pipeline_qwen3_ZH.py",
}

CORR_CONFIG = "configs/full_pipeline/base_data/pipeline_answer_corr_ZH.py"

DISTANCE_HASH_PARAMS = [
    "data_year", "dist", "data_choice", "clone_id",
    "embedding_instruction", "embedding_task", "correlation_answer_source",
]


def compute_hash(config) -> str:
    params = {k: getattr(config, k, None) for k in DISTANCE_HASH_PARAMS}
    param_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode()).hexdigest()[:12]


def find_distance_file(config) -> Path | None:
    """Find cached distance file. Falls back to any file for this model if exact hash missing."""
    h = compute_hash(config)
    # Try exact hash first
    matches = list(CACHE_DIR.glob(f"*{h}.parquet"))
    if matches:
        return matches[0]

    # Fallback: any cached file for this model (cloned configs include original pairs too)
    dist_name = config.dist
    fallbacks = sorted(CACHE_DIR.glob(f"dist_2023_{dist_name}_*.parquet"))
    if fallbacks:
        print(f"  (exact hash {h} not found, using fallback with original-pair filtering)")
        return fallbacks[0]

    return None


def load_distances(path: Path, original_only: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "Similarity" in df.columns and "Distance" not in df.columns:
        df["Distance"] = np.sqrt(np.maximum(0, 2 * (1 - df["Similarity"])))
    if original_only:
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


def main():
    # Load correlation ground truth
    print("Loading ANSWER-CORRELATION ground truth...")
    corr_config = load_config(Path(CORR_CONFIG))
    corr_path = find_distance_file(corr_config)
    if not corr_path:
        raise FileNotFoundError(
            f"No cached distance file for ANSWER-CORRELATION. "
            f"Hash: {compute_hash(corr_config)}. Run the pipeline first."
        )
    print(f"  Found: {corr_path.name}")
    corr_df = load_distances(corr_path, original_only=True)
    corr_pairs = to_pair_dict(corr_df)
    print(f"  {len(corr_pairs)} question pairs")

    # Load question metadata for topic-based analysis
    from vqs.data_loader import load_dataset
    metadata_config = load_config(Path(CORR_CONFIG))
    dataset = load_dataset(metadata_config)
    q_info = dataset["questions"]
    cat_map = q_info.set_index("ID_question")["_category"].to_dict()

    # Build within-topic and cross-topic pair sets
    within_topic_keys = set()
    cross_topic_keys = set()
    for (id1, id2) in corr_pairs:
        cat1 = cat_map.get(id1)
        cat2 = cat_map.get(id2)
        if cat1 and cat2:
            key = (min(id1, id2), max(id1, id2))
            if cat1 == cat2:
                within_topic_keys.add(key)
            else:
                cross_topic_keys.add(key)
    print(f"\n  Within-topic pairs: {len(within_topic_keys)}, Cross-topic pairs: {len(cross_topic_keys)}")

    # Load each embedding model and store pair data for topic analysis
    results = []
    model_pair_data = {}  # model_name -> (common_keys, corr_dists, emb_dists)
    for model_name, config_path in MODEL_CONFIGS.items():
        print(f"\n--- {model_name} ---")
        config = load_config(Path(config_path))
        dist_path = find_distance_file(config)

        if not dist_path:
            print(f"  MISSING: hash={compute_hash(config)}")
            continue

        print(f"  Found: {dist_path.name}")
        emb_df = load_distances(dist_path, original_only=True)
        emb_pairs = to_pair_dict(emb_df)

        # Find common pairs
        common_keys = sorted(set(corr_pairs.keys()) & set(emb_pairs.keys()))
        if len(common_keys) < 10:
            print(f"  Only {len(common_keys)} common pairs — skipping")
            continue

        corr_dists = np.array([corr_pairs[k] for k in common_keys])
        emb_dists = np.array([emb_pairs[k] for k in common_keys])

        model_pair_data[model_name] = (common_keys, corr_dists, emb_dists)

        # Compute correlations
        spearman_r, spearman_p = spearmanr(corr_dists, emb_dists)
        pearson_r, pearson_p = pearsonr(corr_dists, emb_dists)

        # Within-topic and cross-topic correlations
        within_idx = [i for i, k in enumerate(common_keys) if k in within_topic_keys]
        cross_idx = [i for i, k in enumerate(common_keys) if k in cross_topic_keys]

        within_sp = spearmanr(corr_dists[within_idx], emb_dists[within_idx])[0] if len(within_idx) > 2 else np.nan
        cross_sp = spearmanr(corr_dists[cross_idx], emb_dists[cross_idx])[0] if len(cross_idx) > 2 else np.nan

        results.append({
            "model": model_name,
            "n_pairs": len(common_keys),
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "within_topic_spearman": within_sp,
            "cross_topic_spearman": cross_sp,
            "n_within": len(within_idx),
            "n_cross": len(cross_idx),
        })

        print(f"  Spearman r={spearman_r:.4f} (p={spearman_p:.2e})")
        print(f"  Pearson  r={pearson_r:.4f} (p={pearson_p:.2e})")
        print(f"  Within-topic ρ={within_sp:.4f} ({len(within_idx)} pairs), Cross-topic ρ={cross_sp:.4f} ({len(cross_idx)} pairs)")

    if not results:
        print("\nNo embedding models found! Run the pipelines first.")
        return

    df_results = pd.DataFrame(results).sort_values("spearman_r", ascending=False)

    # Print ranking
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"\n{'='*70}")
    p(f"EMBEDDING MODEL VALIDATION vs ANSWER-CORRELATION")
    p(f"  Ground truth: 1 - |Pearson_r| of voter answers (75 original questions)")
    p(f"  Metric: Spearman rank correlation between distance matrices")
    p(f"{'='*70}")

    p(f"\n  {'Rank':<5} {'Model':<30} {'Spearman r':>11} {'Pearson r':>11} {'N pairs':>8}")
    p(f"  {'-'*67}")
    for rank, (_, row) in enumerate(df_results.iterrows(), 1):
        p(f"  {rank:<5} {row['model']:<30} {row['spearman_r']:>+11.4f} "
          f"{row['pearson_r']:>+11.4f} {int(row['n_pairs']):>8}")

    # Within-topic vs cross-topic analysis
    p(f"\n{'='*70}")
    p(f"WITHIN-TOPIC vs CROSS-TOPIC ANALYSIS")
    p(f"  Within-topic: both questions in the same category ({len(within_topic_keys)} pairs)")
    p(f"  Cross-topic: questions from different categories ({len(cross_topic_keys)} pairs)")
    p(f"{'='*70}")

    p(f"\n  {'Rank':<5} {'Model':<30} {'All':>8} {'Within':>8} {'Cross':>8} {'W-C':>8}")
    p(f"  {'-'*69}")
    for rank, (_, row) in enumerate(df_results.iterrows(), 1):
        w = row['within_topic_spearman']
        c = row['cross_topic_spearman']
        delta = w - c if not (np.isnan(w) or np.isnan(c)) else np.nan
        p(f"  {rank:<5} {row['model']:<30} {row['spearman_r']:>+8.4f} "
          f"{w:>+8.4f} {c:>+8.4f} {delta:>+8.4f}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = f"emb_vs_corr_{timestamp}"

    csv_path = OUTPUT_DIR / f"{base}_ranking.csv"
    df_results.to_csv(csv_path, index=False)

    report_path = OUTPUT_DIR / f"{base}_report.txt"
    report_path.write_text("\n".join(lines))

    # Plot 1: Grouped bar chart — All / Within-topic / Cross-topic
    fig, ax = plt.subplots(figsize=(12, 6))
    models = df_results["model"].tolist()
    y_pos = np.arange(len(models))
    bar_h = 0.25

    all_vals = df_results["spearman_r"].values
    within_vals = df_results["within_topic_spearman"].values
    cross_vals = df_results["cross_topic_spearman"].values

    ax.barh(y_pos - bar_h, all_vals, bar_h, label="All pairs", color="#3498db", edgecolor="white")
    ax.barh(y_pos, within_vals, bar_h, label="Within-topic", color="#e74c3c", edgecolor="white")
    ax.barh(y_pos + bar_h, cross_vals, bar_h, label="Cross-topic", color="#95a5a6", edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Spearman rank correlation with ANSWER-CORRELATION distances")
    ax.set_title("Embedding Model Validation: Agreement with Voter-Answer Redundancy")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)

    for i, (a, w, c) in enumerate(zip(all_vals, within_vals, cross_vals)):
        ax.text(max(a, w, c) + 0.005, i, f"all={a:.3f}  within={w:.3f}  cross={c:.3f}",
                va="center", fontsize=7, color="black")

    fig.tight_layout()
    bar_path = OUTPUT_DIR / f"{base}_ranking.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Scatter plots (embedding distance vs correlation distance)
    n_models = len(df_results)
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (_, row) in enumerate(df_results.iterrows()):
        ax = axes[idx]
        model_name = row["model"]

        config = load_config(Path(MODEL_CONFIGS[model_name]))
        dist_path = find_distance_file(config)
        emb_df = load_distances(dist_path)
        emb_pairs = to_pair_dict(emb_df)

        common_keys = sorted(set(corr_pairs.keys()) & set(emb_pairs.keys()))
        corr_dists = np.array([corr_pairs[k] for k in common_keys])
        emb_dists = np.array([emb_pairs[k] for k in common_keys])

        ax.scatter(corr_dists, emb_dists, alpha=0.3, s=8, color="steelblue")
        ax.set_xlabel("Correlation distance", fontsize=7)
        ax.set_ylabel("Embedding distance", fontsize=7)
        ax.set_title(f"{model_name}\nρ={row['spearman_r']:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Embedding Distance vs Voter-Answer Correlation Distance", fontsize=11, y=1.02)
    fig.tight_layout()
    scatter_path = OUTPUT_DIR / f"{base}_scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Rank-based scatter (shows Spearman signal more clearly)
    from scipy.stats import rankdata
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (_, row) in enumerate(df_results.iterrows()):
        ax = axes[idx]
        model_name = row["model"]

        config = load_config(Path(MODEL_CONFIGS[model_name]))
        dist_path = find_distance_file(config)
        emb_df = load_distances(dist_path, original_only=True)
        emb_pairs = to_pair_dict(emb_df)

        common_keys = sorted(set(corr_pairs.keys()) & set(emb_pairs.keys()))
        corr_dists = np.array([corr_pairs[k] for k in common_keys])
        emb_dists = np.array([emb_pairs[k] for k in common_keys])

        corr_ranks = rankdata(corr_dists)
        emb_ranks = rankdata(emb_dists)

        ax.scatter(corr_ranks, emb_ranks, alpha=0.15, s=5, color="steelblue")
        # Add trend line
        z = np.polyfit(corr_ranks, emb_ranks, 1)
        x_line = np.array([corr_ranks.min(), corr_ranks.max()])
        ax.plot(x_line, z[0] * x_line + z[1], color="red", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Correlation distance rank", fontsize=7)
        ax.set_ylabel("Embedding distance rank", fontsize=7)
        ax.set_title(f"{model_name}\n\u03c1={row['spearman_r']:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Rank-Based: Embedding vs Correlation Distance", fontsize=11, y=1.02)
    fig.tight_layout()
    rank_scatter_path = OUTPUT_DIR / f"{base}_rank_scatter.png"
    fig.savefig(rank_scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =========================================================
    # Per-topic correlation analysis (pure, no embeddings)
    # =========================================================
    p(f"\n{'='*70}")
    p(f"PER-TOPIC ANSWER CORRELATION ANALYSIS")
    p(f"  Metric: 1 - |Pearson_r| of voter answers within each topic")
    p(f"{'='*70}")

    topic_questions = q_info.groupby("_category")["ID_question"].apply(list).to_dict()
    topic_rows = []
    for topic, qids in sorted(topic_questions.items()):
        n_q = len(qids)
        topic_dists = []
        for i, id1 in enumerate(qids):
            for id2 in qids[i+1:]:
                key = (min(id1, id2), max(id1, id2))
                if key in corr_pairs:
                    topic_dists.append(corr_pairs[key])
        if topic_dists:
            dists = np.array(topic_dists)
            abs_r_vals = 1 - dists
            topic_rows.append({
                "category": topic,
                "n_questions": n_q,
                "n_pairs": len(topic_dists),
                "mean_abs_r": abs_r_vals.mean(),
                "max_abs_r": abs_r_vals.max(),
                "min_distance": dists.min(),
                "mean_distance": dists.mean(),
                "std_distance": dists.std(),
            })

    df_topics = pd.DataFrame(topic_rows).sort_values("mean_abs_r", ascending=False)

    cross_dists = np.array([corr_pairs[k] for k in cross_topic_keys])
    cross_abs_r = 1 - cross_dists

    p(f"\n  {'Category':<38} {'N_q':>3} {'Pairs':>5} {'Mean |r|':>9} {'Max |r|':>9} {'Mean dist':>10}")
    p(f"  {'-'*78}")
    for _, row in df_topics.iterrows():
        p(f"  {row['category']:<38} {int(row['n_questions']):>3} {int(row['n_pairs']):>5} "
          f"{row['mean_abs_r']:>9.4f} {row['max_abs_r']:>9.4f} "
          f"{row['mean_distance']:>10.4f}")
    p(f"\n  {'CROSS-TOPIC (all)':<38} {'':>3} {len(cross_dists):>5} "
      f"{cross_abs_r.mean():>9.4f} {cross_abs_r.max():>9.4f} "
      f"{cross_dists.mean():>10.4f}")

    topic_csv_path = OUTPUT_DIR / f"{base}_topic_correlations.csv"
    df_topics.to_csv(topic_csv_path, index=False)

    # Plot 4: Per-topic bar chart of mean |r|
    fig, ax = plt.subplots(figsize=(10, 6))
    topics = df_topics["category"].tolist()
    mean_r = df_topics["mean_abs_r"].values
    n_qs = df_topics["n_questions"].values
    max_scale = max(mean_r.max(), 0.3)
    colors_t = plt.cm.RdYlGn(mean_r / max_scale)

    ax.barh(range(len(topics)), mean_r, color=colors_t, edgecolor="white")
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels([f"{t} ({n}q)" for t, n in zip(topics, n_qs)], fontsize=9)
    ax.set_xlabel("Mean |Pearson r| within topic (higher = more correlated answers)")
    ax.set_title("Within-Topic Answer Correlation by Category")
    ax.axvline(cross_abs_r.mean(), color="gray", linestyle="--", linewidth=1,
               label=f"Cross-topic mean |r|={cross_abs_r.mean():.3f}")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    for i, (v, mx) in enumerate(zip(mean_r, df_topics["max_abs_r"].values)):
        ax.text(v + 0.003, i, f"mean={v:.3f}, max={mx:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    topic_bar_path = OUTPUT_DIR / f"{base}_topic_correlations.png"
    fig.savefig(topic_bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    p(f"\n--- Saved to {OUTPUT_DIR}/ ---")
    p(f"  {csv_path.name}")
    p(f"  {bar_path.name}")
    p(f"  {scatter_path.name}")
    p(f"  {rank_scatter_path.name}")
    p(f"  {topic_csv_path.name}")
    p(f"  {topic_bar_path.name}")
    p(f"  {report_path.name}")


if __name__ == "__main__":
    main()
