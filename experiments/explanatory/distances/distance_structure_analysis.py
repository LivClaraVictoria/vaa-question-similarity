"""
Distance and correlation analysis: what is the functional redundancy structure of
the questionnaire, and how well do embedding models capture it?

Four sections, each telling one part of the story:

  §1 Raw correlation structure
       Pairwise Pearson correlations across all 75 question pairs from voter
       answers.  Produces a density plot, sorted CSV, and top-N most correlated
       pairs.  Config required (load_voters=True).

  §2 Embedding vs correlation agreement
       For each of 10 embedding models, compute Spearman rank correlation between
       the model's distance matrix and the ANSWER-CORRELATION ground-truth matrix.
       Also splits into within-topic / cross-topic pairs.
       Uses cached distance parquets — no config needed.

  §3 Within-topic vs cross-topic distances (E5-INSTRUCT & ANSWER-CORR-ARCCOS)
       For both metrics: per-topic intra-topic statistics, topic × topic heatmap,
       and overall intra vs inter comparison with Mann-Whitney U and effect sizes.
       Uses hardcoded parquet paths — no config needed.

  §4 Mini vs full-only correlation distributions
       Computes |Pearson r| and arccos distance for all question pairs classified
       as within-mini, within-full-only, or cross.  Config required
       (load_voters=True, full questionnaire data).

Usage:
    # Run all sections (§1 and §4 require --config)
    python -m experiments.exploratory.analysis.distance_correlation_analysis \\
        --config configs/base_pipeline/pipeline_answer_corr_ZH.py

    # Run only §2 and §3 (no config needed)
    python -m experiments.exploratory.analysis.distance_correlation_analysis \\
        --section 2,3

    # Run §1 with 30 top pairs
    python -m experiments.exploratory.analysis.distance_correlation_analysis \\
        --section 1 --config configs/base_pipeline/pipeline_answer_corr_ZH.py \\
        --top 30
"""

import argparse
import hashlib
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, mannwhitneyu, pearsonr, rankdata, spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR = Path("cache/distance_calculations")
DATA_DIR = Path("data/cleaned")

_SEC1_OUTPUT_DIR = Path("experiment_results/distance_analysis/PURE_CORRELATION_2023")
_SEC2_OUTPUT_DIR = Path("experiment_results/correlation_metric_results/correlation_analysis")
_SEC3_OUTPUT_DIR = Path("experiment_results/topic_distances")
_SEC4_OUTPUT_DIR = Path("experiment_results/correlation_metric_results/correlation_analysis")

# §2 — one base config per embedding model
_MODEL_CONFIGS = {
    "SBERT": "configs/base_pipeline/pipeline_sbert_ZH.py",
    "E5": "configs/base_pipeline/pipeline_e5_ZH.py",
    "E5-ASYMMETRIC": "configs/base_pipeline/pipeline_e5_asym_ZH.py",
    "E5-INSTRUCT": "configs/base_pipeline/pipeline_e5_instruct_ZH.py",
    "E5-ASYMMETRIC-INSTRUCT": "configs/base_pipeline/pipeline_e5_asym_instruct_ZH.py",
    "JINA-V3": "configs/base_pipeline/pipeline_jina_v3_ZH.py",
    "BGE-M3": "configs/base_pipeline/pipeline_bge_m3_ZH.py",
    "GTE": "configs/base_pipeline/pipeline_gte_ZH.py",
    "NOMIC-V2": "configs/base_pipeline/pipeline_nomic_v2_ZH.py",
    "QWEN3": "configs/base_pipeline/pipeline_qwen3_ZH.py",
}
_CORR_CONFIG = "configs/base_pipeline/pipeline_answer_corr_ZH.py"

# §3 — specific cached distance files
_SEC3_METRICS = {
    "e5_instruct": {
        "label": "E5-INSTRUCT",
        "path": CACHE_DIR / "dist_2023_E5-INSTRUCT_ba053f9f59a3.parquet",
    },
    "answer_corr_arccos": {
        "label": "ANSWER-CORRELATION-ARCCOS",
        "path": CACHE_DIR / "dist_2023_ANSWER-CORRELATION-ARCCOS_790a61671ea5.parquet",
    },
}

_DISTANCE_HASH_PARAMS = [
    "data_year", "dist", "data_choice", "clone_id",
    "embedding_instruction", "embedding_task", "correlation_answer_source",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_distances(path: Path, original_only: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "Similarity" in df.columns and "Distance" not in df.columns:
        df["Distance"] = np.sqrt(np.maximum(0, 2 * (1 - df["Similarity"])))
    if original_only:
        df = df[(df["ID1"] < 9_000_000) & (df["ID2"] < 9_000_000)]
    return df


def _to_pair_dict(df: pd.DataFrame) -> dict:
    result = {}
    for _, row in df.iterrows():
        id1, id2 = int(row["ID1"]), int(row["ID2"])
        result[(min(id1, id2), max(id1, id2))] = row["Distance"]
    return result


def _compute_dist_hash(config) -> str:
    params = {k: getattr(config, k, None) for k in _DISTANCE_HASH_PARAMS}
    return hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _find_distance_file(config) -> Path | None:
    h = _compute_dist_hash(config)
    matches = list(CACHE_DIR.glob(f"*{h}.parquet"))
    if matches:
        return matches[0]
    dist_name = config.dist
    fallbacks = sorted(CACHE_DIR.glob(f"dist_2023_{dist_name}_*.parquet"))
    if fallbacks:
        print(f"  (exact hash {h} not found, using fallback with original-pair filtering)")
        return fallbacks[0]
    return None


# ---------------------------------------------------------------------------
# §1  Raw Pearson correlation structure
# ---------------------------------------------------------------------------


def section1_raw_correlations(config, top: int):
    """Pairwise Pearson correlations between all question pairs from voter answers."""
    from vqs.data_loader import load_dataset

    print(f"\n{'=' * 70}")
    print("§1  RAW PEARSON CORRELATION STRUCTURE")
    print(f"{'=' * 70}")

    dataset = load_dataset(config)
    questions_df = dataset["questions"]
    question_ids = questions_df["ID_question"].tolist()

    # Determine question text column
    col_candidates = ["question_EN", "question_en", "question_DE", "question_de"]
    text_col = next((c for c in col_candidates if c in questions_df.columns), None)
    if text_col:
        id_to_text = dict(zip(questions_df["ID_question"], questions_df[text_col]))
    else:
        id_to_text = {qid: str(qid) for qid in question_ids}

    # Compute pairwise Pearson correlations
    answer_cols = [f"answer_{qid}" for qid in question_ids]
    voters_df = dataset["voters"][answer_cols]
    corr_matrix = voters_df.corr()

    rows = []
    for i, j in combinations(range(len(question_ids)), 2):
        id1, id2 = question_ids[i], question_ids[j]
        r = corr_matrix.iloc[i, j]
        rows.append({
            "ID1": id1, "ID2": id2,
            "Qu1": id_to_text[id1], "Qu2": id_to_text[id2],
            "correlation": r, "abs_correlation": abs(r),
        })

    df = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    correlations = df["correlation"].values

    stats = {
        "n_pairs": len(df), "n_questions": len(question_ids),
        "mean_r": correlations.mean(), "median_r": float(np.median(correlations)),
        "std_r": correlations.std(), "min_r": correlations.min(), "max_r": correlations.max(),
        "mean_abs_r": df["abs_correlation"].mean(),
        "median_abs_r": float(df["abs_correlation"].median()),
        "max_abs_r": df["abs_correlation"].max(),
        "n_above_03": int((df["abs_correlation"] > 0.3).sum()),
        "n_above_05": int((df["abs_correlation"] > 0.5).sum()),
        "n_negative": int((correlations < 0).sum()),
        "n_positive": int((correlations > 0).sum()),
    }

    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"  Questions: {stats['n_questions']}  |  Pairs: {stats['n_pairs']}")
    p()
    for key in ["mean_r", "median_r", "std_r", "min_r", "max_r",
                "mean_abs_r", "median_abs_r", "max_abs_r"]:
        p(f"  {key:<25} {stats[key]:>10.4f}")
    p()
    p(f"  Positive correlations:  {stats['n_positive']}")
    p(f"  Negative correlations:  {stats['n_negative']}")
    p(f"  |r| > 0.3:             {stats['n_above_03']}")
    p(f"  |r| > 0.5:             {stats['n_above_05']}")
    p()
    p(f"TOP {top} MOST CORRELATED PAIRS (by |r|)")
    p("-" * 70)
    for _, row in df.head(top).iterrows():
        r = row["correlation"]
        p(f"  {int(row['ID1']):>8} — {int(row['ID2']):>8}  r={r:+.4f}  |r|={abs(r):.4f}")
        p(f"    Q1: {str(row['Qu1'])[:60]}")
        p(f"    Q2: {str(row['Qu2'])[:60]}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(correlations, bins=50, alpha=0.7, color="steelblue", edgecolor="white", density=True)
    xs = np.linspace(correlations.min() - 0.05, correlations.max() + 0.05, 200)
    ax.plot(xs, gaussian_kde(correlations)(xs), color="darkblue", linewidth=1.5)
    ax.axvline(0, color="black", linestyle=":", alpha=0.3)
    ax.axvline(float(np.median(correlations)), color="red", linestyle="--",
               label=f"median={np.median(correlations):.3f}")
    ax.axvline(float(correlations.mean()), color="orange", linestyle="--",
               label=f"mean={correlations.mean():.3f}")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Density")
    ax.set_title("Raw correlation distribution")
    ax.legend(fontsize=8)

    ax = axes[1]
    abs_corr = df["abs_correlation"].values
    ax.hist(abs_corr, bins=50, alpha=0.7, color="coral", edgecolor="white", density=True)
    xs2 = np.linspace(0, abs_corr.max() + 0.05, 200)
    ax.plot(xs2, gaussian_kde(abs_corr)(xs2), color="darkred", linewidth=1.5)
    ax.axvline(float(np.median(abs_corr)), color="red", linestyle="--",
               label=f"median={np.median(abs_corr):.3f}")
    ax.axvline(float(abs_corr.mean()), color="orange", linestyle="--",
               label=f"mean={abs_corr.mean():.3f}")
    ax.set_xlabel("|Pearson r|")
    ax.set_ylabel("Density")
    ax.set_title("|Correlation| distribution")
    ax.legend(fontsize=8)

    fig.suptitle("Pure Correlation Analysis — All Question Pairs", fontsize=12, y=1.02)
    fig.tight_layout()

    out = _SEC1_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M")
    base = f"pure_correlation_{ts}"

    csv_path = out / f"{base}_all_correlations.csv"
    df.to_csv(csv_path, index=False)
    plot_path = out / f"{base}_distribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    report_path = out / f"{base}_report.txt"
    report_path.write_text("\n".join(lines))

    print(f"\n  -> {out}/  ({csv_path.name}, {plot_path.name})")


# ---------------------------------------------------------------------------
# §2  Embedding model validation against correlation ground truth
# ---------------------------------------------------------------------------


def section2_embedding_validation():
    """Spearman rank correlation between each embedding model and ANSWER-CORRELATION."""
    from vqs.config_utils import load_config

    print(f"\n{'=' * 70}")
    print("§2  EMBEDDING MODEL VALIDATION vs ANSWER-CORRELATION GROUND TRUTH")
    print(f"{'=' * 70}")

    # Load correlation ground truth
    corr_config = load_config(Path(_CORR_CONFIG))
    corr_path = _find_distance_file(corr_config)
    if not corr_path:
        print("  ERROR: No cached distance file for ANSWER-CORRELATION. Run the pipeline first.")
        return
    print(f"  Correlation ground truth: {corr_path.name}")
    corr_df = _load_distances(corr_path, original_only=True)
    corr_pairs = _to_pair_dict(corr_df)
    print(f"  {len(corr_pairs)} question pairs")

    # Load question metadata for within/cross-topic analysis
    from vqs.data_loader import load_dataset
    dataset = load_dataset(corr_config)
    q_info = dataset["questions"]
    cat_map = q_info.set_index("ID_question")["_category"].to_dict()

    within_topic_keys = set()
    cross_topic_keys = set()
    for (id1, id2) in corr_pairs:
        c1 = cat_map.get(id1)
        c2 = cat_map.get(id2)
        if c1 and c2:
            key = (min(id1, id2), max(id1, id2))
            (within_topic_keys if c1 == c2 else cross_topic_keys).add(key)
    print(f"  Within-topic pairs: {len(within_topic_keys)}, Cross-topic: {len(cross_topic_keys)}")

    results = []
    for model_name, config_path in _MODEL_CONFIGS.items():
        print(f"\n--- {model_name} ---")
        config = load_config(Path(config_path))
        dist_path = _find_distance_file(config)
        if not dist_path:
            print(f"  MISSING: hash={_compute_dist_hash(config)}")
            continue

        emb_df = _load_distances(dist_path, original_only=True)
        emb_pairs = _to_pair_dict(emb_df)
        common_keys = sorted(set(corr_pairs.keys()) & set(emb_pairs.keys()))
        if len(common_keys) < 10:
            print(f"  Only {len(common_keys)} common pairs — skipping")
            continue

        corr_dists = np.array([corr_pairs[k] for k in common_keys])
        emb_dists = np.array([emb_pairs[k] for k in common_keys])

        sp_r, sp_p = spearmanr(corr_dists, emb_dists)
        pe_r, pe_p = pearsonr(corr_dists, emb_dists)

        within_idx = [i for i, k in enumerate(common_keys) if k in within_topic_keys]
        cross_idx = [i for i, k in enumerate(common_keys) if k in cross_topic_keys]
        within_sp = spearmanr(corr_dists[within_idx], emb_dists[within_idx])[0] if len(within_idx) > 2 else np.nan
        cross_sp = spearmanr(corr_dists[cross_idx], emb_dists[cross_idx])[0] if len(cross_idx) > 2 else np.nan

        results.append({
            "model": model_name, "n_pairs": len(common_keys),
            "spearman_r": sp_r, "spearman_p": sp_p,
            "pearson_r": pe_r, "pearson_p": pe_p,
            "within_topic_spearman": within_sp, "cross_topic_spearman": cross_sp,
            "n_within": len(within_idx), "n_cross": len(cross_idx),
        })
        print(f"  Spearman r={sp_r:.4f}  Pearson r={pe_r:.4f}")
        print(f"  Within-topic ρ={within_sp:.4f} ({len(within_idx)} pairs)  "
              f"Cross-topic ρ={cross_sp:.4f} ({len(cross_idx)} pairs)")

    if not results:
        print("\n  No embedding models found — run the pipelines first.")
        return

    df_results = pd.DataFrame(results).sort_values("spearman_r", ascending=False)
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"\n{'='*70}")
    p("RANKING — Spearman ρ with voter-answer correlation distances")
    p(f"{'='*70}")
    p(f"\n  {'Rank':<5} {'Model':<30} {'Spearman r':>11} {'Pearson r':>11}")
    p(f"  {'-'*60}")
    for rank, (_, row) in enumerate(df_results.iterrows(), 1):
        p(f"  {rank:<5} {row['model']:<30} {row['spearman_r']:>+11.4f} {row['pearson_r']:>+11.4f}")

    p(f"\n{'='*70}")
    p("WITHIN-TOPIC vs CROSS-TOPIC BREAKDOWN")
    p(f"{'='*70}")
    p(f"\n  {'Rank':<5} {'Model':<30} {'All':>8} {'Within':>8} {'Cross':>8} {'W-C':>8}")
    p(f"  {'-'*69}")
    for rank, (_, row) in enumerate(df_results.iterrows(), 1):
        w, c = row["within_topic_spearman"], row["cross_topic_spearman"]
        delta = w - c if not (np.isnan(w) or np.isnan(c)) else np.nan
        p(f"  {rank:<5} {row['model']:<30} {row['spearman_r']:>+8.4f} "
          f"{w:>+8.4f} {c:>+8.4f} {delta:>+8.4f}")

    out = _SEC2_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M")
    base = f"emb_vs_corr_{ts}"

    df_results.to_csv(out / f"{base}_ranking.csv", index=False)
    (out / f"{base}_report.txt").write_text("\n".join(lines))

    # Bar chart: All / Within / Cross
    fig, ax = plt.subplots(figsize=(12, 6))
    models = df_results["model"].tolist()
    y_pos = np.arange(len(models))
    bar_h = 0.25
    ax.barh(y_pos - bar_h, df_results["spearman_r"].values, bar_h,
            label="All pairs", color="#3498db", edgecolor="white")
    ax.barh(y_pos, df_results["within_topic_spearman"].values, bar_h,
            label="Within-topic", color="#e74c3c", edgecolor="white")
    ax.barh(y_pos + bar_h, df_results["cross_topic_spearman"].values, bar_h,
            label="Cross-topic", color="#95a5a6", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Spearman ρ with ANSWER-CORRELATION distances")
    ax.set_title("Embedding Model Agreement with Voter-Answer Redundancy")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    bar_path = out / f"{base}_ranking.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Scatter plots
    n_models = len(df_results)
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols

    for suffix, rank_based in [("_scatter.png", False), ("_rank_scatter.png", True)]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes = np.array([axes]).flatten() if n_models == 1 else np.array(axes).flatten()

        for idx, (_, row) in enumerate(df_results.iterrows()):
            ax = axes[idx]
            model_name = row["model"]
            config = load_config(Path(_MODEL_CONFIGS[model_name]))
            dist_path = _find_distance_file(config)
            emb_df = _load_distances(dist_path, original_only=(not rank_based))
            emb_pairs_l = _to_pair_dict(emb_df)
            common_keys = sorted(set(corr_pairs.keys()) & set(emb_pairs_l.keys()))
            cd = np.array([corr_pairs[k] for k in common_keys])
            ed = np.array([emb_pairs_l[k] for k in common_keys])

            if rank_based:
                cd, ed = rankdata(cd), rankdata(ed)
                z = np.polyfit(cd, ed, 1)
                x_line = np.array([cd.min(), cd.max()])
                ax.scatter(cd, ed, alpha=0.15, s=5, color="steelblue")
                ax.plot(x_line, z[0] * x_line + z[1], color="red", linewidth=1.5, alpha=0.8)
                ax.set_xlabel("Corr rank", fontsize=7)
                ax.set_ylabel("Emb rank", fontsize=7)
            else:
                ax.scatter(cd, ed, alpha=0.3, s=8, color="steelblue")
                ax.set_xlabel("Corr dist", fontsize=7)
                ax.set_ylabel("Emb dist", fontsize=7)

            ax.set_title(f"{model_name}\nρ={row['spearman_r']:.3f}", fontsize=9)
            ax.tick_params(labelsize=7)

        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        title = ("Rank-Based: Embedding vs Correlation Distance" if rank_based
                 else "Embedding Distance vs Correlation Distance")
        fig.suptitle(title, fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"{base}{suffix}", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\n  -> {out}/  ({base}_ranking.csv, .png, _scatter.png, _rank_scatter.png)")


# ---------------------------------------------------------------------------
# §3  Within-topic vs cross-topic distances (E5-INSTRUCT & ANSWER-CORR-ARCCOS)
# ---------------------------------------------------------------------------


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return (b.mean() - a.mean()) / np.sqrt(pooled_var)


def _compute_intra_topic(pairs: dict, topic_questions: dict) -> pd.DataFrame:
    rows = []
    for topic, qids in sorted(topic_questions.items()):
        dists = [pairs[(min(i, j), max(i, j))]
                 for idx, i in enumerate(qids) for j in qids[idx + 1:]
                 if (min(i, j), max(i, j)) in pairs]
        if not dists:
            continue
        d = np.array(dists)
        rows.append({
            "topic": topic, "n_questions": len(qids), "n_pairs": len(dists),
            "mean": d.mean(), "median": float(np.median(d)),
            "std": d.std(ddof=1) if len(d) > 1 else 0.0,
            "min": d.min(), "max": d.max(),
            "q25": float(np.percentile(d, 25)), "q75": float(np.percentile(d, 75)),
        })
    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)


def _compute_inter_topic(pairs: dict, topic_questions: dict) -> pd.DataFrame:
    topics = sorted(topic_questions.keys())
    rows = []
    for i, t_a in enumerate(topics):
        for t_b in topics[i + 1:]:
            dists = [pairs[(min(i1, i2), max(i1, i2))]
                     for i1 in topic_questions[t_a] for i2 in topic_questions[t_b]
                     if (min(i1, i2), max(i1, i2)) in pairs]
            if not dists:
                continue
            d = np.array(dists)
            rows.append({
                "topic_a": t_a, "topic_b": t_b, "n_pairs": len(dists),
                "mean": d.mean(), "median": float(np.median(d)),
                "std": d.std(ddof=1) if len(d) > 1 else 0.0,
                "min": d.min(), "max": d.max(),
            })
    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)


def _compute_intra_inter_summary(pairs: dict, cat_map: dict) -> dict:
    intra = [d for (id1, id2), d in pairs.items()
             if cat_map.get(id1) and cat_map.get(id2) and cat_map[id1] == cat_map[id2]]
    inter = [d for (id1, id2), d in pairs.items()
             if cat_map.get(id1) and cat_map.get(id2) and cat_map[id1] != cat_map[id2]]
    intra, inter = np.array(intra), np.array(inter)
    U, p_val = mannwhitneyu(intra, inter, alternative="less")
    n_intra, n_inter = len(intra), len(inter)
    auroc = U / (n_intra * n_inter)
    return {
        "intra_mean": intra.mean(), "intra_median": float(np.median(intra)),
        "intra_std": intra.std(ddof=1), "intra_n_pairs": n_intra,
        "inter_mean": inter.mean(), "inter_median": float(np.median(inter)),
        "inter_std": inter.std(ddof=1), "inter_n_pairs": n_inter,
        "mann_whitney_U": float(U), "mann_whitney_p": float(p_val),
        "cohens_d": _cohens_d(intra, inter), "auroc": auroc,
        "rank_biserial_r": 2 * U / (n_intra * n_inter) - 1,
    }


def _build_heatmap_matrix(pairs: dict, topics_sorted: list, topic_questions: dict) -> np.ndarray:
    n = len(topics_sorted)
    mat = np.full((n, n), np.nan)
    for i, t_a in enumerate(topics_sorted):
        qids_a = topic_questions[t_a]
        dists = [pairs[(min(qa, qb), max(qa, qb))]
                 for qi, qa in enumerate(qids_a) for qb in qids_a[qi + 1:]
                 if (min(qa, qb), max(qa, qb)) in pairs]
        if dists:
            mat[i, i] = np.mean(dists)
        for j, t_b in enumerate(topics_sorted):
            if j <= i:
                continue
            dists = [pairs[(min(i1, i2), max(i1, i2))]
                     for i1 in qids_a for i2 in topic_questions[t_b]
                     if (min(i1, i2), max(i1, i2)) in pairs]
            if dists:
                mat[i, j] = mat[j, i] = np.mean(dists)
    return mat


def _plot_topic_boxplot(pairs: dict, cat_map: dict, topic_questions: dict,
                        metric_label: str, output_path: Path):
    intra_by_topic = {}
    for topic, qids in sorted(topic_questions.items()):
        dists = [pairs[(min(i, j), max(i, j))]
                 for idx, i in enumerate(qids) for j in qids[idx + 1:]
                 if (min(i, j), max(i, j)) in pairs]
        if dists:
            intra_by_topic[topic] = np.array(dists)
    inter_dists = np.array([d for (id1, id2), d in pairs.items()
                            if cat_map.get(id1) and cat_map.get(id2)
                            and cat_map[id1] != cat_map[id2]])

    sorted_topics = sorted(intra_by_topic.keys(), key=lambda t: np.median(intra_by_topic[t]))
    labels = [f"{t} ({len(intra_by_topic[t])} pairs)" for t in sorted_topics]
    data = [intra_by_topic[t] for t in sorted_topics]
    labels.append(f"INTER-TOPIC ({len(inter_dists)} pairs)")
    data.append(inter_dists)

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, (d, label) in enumerate(zip(data, labels)):
        color = "#95a5a6" if i == len(data) - 1 else "#3498db"
        alpha = 0.6 if i == len(data) - 1 else 0.8
        if len(d) < 6:
            ax.scatter(d, [i] * len(d), color=color, alpha=alpha, s=40, edgecolors="white", zorder=3)
            ax.plot([np.median(d)], [i], marker="|", color="red", markersize=15, markeredgewidth=2, zorder=4)
        else:
            ax.boxplot(d, positions=[i], vert=False, widths=0.5, patch_artist=True,
                       showfliers=False,
                       boxprops=dict(facecolor=color, alpha=alpha),
                       medianprops=dict(color="red", linewidth=1.5),
                       whiskerprops=dict(color="gray"), capprops=dict(color="gray"))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"Distance ({metric_label})", fontsize=11)
    ax.set_title(f"Within-Topic vs Cross-Topic Distances — {metric_label}", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_sec3(mat: np.ndarray, topics: list, metric_label: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(mat, cmap="viridis_r", aspect="equal")
    for i in range(len(topics)):
        for j in range(len(topics)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, fontweight="bold" if i == j else "normal",
                        color="white" if val < float(np.nanmedian(mat)) else "black")
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=8)
    ax.set_title(f"Mean Pairwise Distance: Topic × Topic — {metric_label}", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Mean Distance", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def section3_topic_distances():
    """Within-topic vs cross-topic distance analysis for E5-INSTRUCT and ANSWER-CORR-ARCCOS."""
    print(f"\n{'=' * 70}")
    print("§3  WITHIN-TOPIC vs CROSS-TOPIC DISTANCES")
    print(f"{'=' * 70}")

    questions = pd.read_parquet(DATA_DIR / "df_questions.parquet")
    cat_map = questions.set_index("ID_question")["_category"].to_dict()
    topic_questions = questions.groupby("_category")["ID_question"].apply(sorted).to_dict()
    print(f"  {len(questions)} questions, {len(topic_questions)} topics")

    all_intra: dict = {}
    all_inter: dict = {}
    all_summaries: dict = {}
    all_heatmaps: dict = {}
    all_topics_sorted: dict = {}

    for metric_key, metric_info in _SEC3_METRICS.items():
        label = metric_info["label"]
        dist_path = metric_info["path"]
        out_dir = _SEC3_OUTPUT_DIR / metric_key

        print(f"\n--- {label} ---")
        if not dist_path.exists():
            print(f"  ERROR: {dist_path} not found — skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        pairs = _to_pair_dict(_load_distances(dist_path))
        print(f"  {len(pairs)} pairs loaded")

        intra_df = _compute_intra_topic(pairs, topic_questions)
        inter_df = _compute_inter_topic(pairs, topic_questions)
        summary = _compute_intra_inter_summary(pairs, cat_map)

        all_intra[metric_key] = intra_df
        all_inter[metric_key] = inter_df
        all_summaries[metric_key] = summary

        topics_sorted = intra_df["topic"].tolist()
        all_topics_sorted[metric_key] = topics_sorted
        heatmap_mat = _build_heatmap_matrix(pairs, topics_sorted, topic_questions)
        all_heatmaps[metric_key] = heatmap_mat

        intra_df.to_csv(out_dir / "topic_distances_intra.csv", index=False)
        inter_df.to_csv(out_dir / "topic_distances_inter.csv", index=False)
        pd.DataFrame([summary]).to_csv(out_dir / "topic_distances_summary.csv", index=False)

        _plot_topic_boxplot(pairs, cat_map, topic_questions, label,
                            out_dir / "topic_distances_boxplot.png")
        _plot_heatmap_sec3(heatmap_mat, topics_sorted, label,
                           out_dir / "topic_distances_heatmap.png")

        # Print summary
        print(f"  Intra mean={summary['intra_mean']:.4f}  Inter mean={summary['inter_mean']:.4f}")
        print(f"  Cohen's d={summary['cohens_d']:.4f}  AUROC={summary['auroc']:.4f}  "
              f"p={summary['mann_whitney_p']:.2e}")
        print(f"  -> {out_dir}/")

    # Cross-metric comparison
    if len(all_intra) == 2 and "e5_instruct" in all_intra and "answer_corr_arccos" in all_intra:
        comp_dir = _SEC3_OUTPUT_DIR / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)

        e5_intra = all_intra["e5_instruct"][["topic", "n_questions", "n_pairs", "mean"]].rename(
            columns={"mean": "e5_intra_mean", "n_pairs": "n_intra_pairs"})
        corr_intra = all_intra["answer_corr_arccos"][["topic", "mean"]].rename(
            columns={"mean": "corr_intra_mean"})
        master = e5_intra.merge(corr_intra, on="topic")
        master["e5_rank"] = master["e5_intra_mean"].rank().astype(int)
        master["corr_rank"] = master["corr_intra_mean"].rank().astype(int)
        master["rank_diff"] = master["corr_rank"] - master["e5_rank"]

        # Comparison plot (2×2)
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        e5_topics = all_topics_sorted["e5_instruct"]
        corr_topics = all_topics_sorted["answer_corr_arccos"]
        e5_mat = all_heatmaps["e5_instruct"]
        corr_mat = all_heatmaps["answer_corr_arccos"]

        for ax, mat, label in [(axes[0, 0], e5_mat, "E5-INSTRUCT"),
                               (axes[0, 1], corr_mat, "ANSWER-CORRELATION-ARCCOS")]:
            topics = e5_topics if label == "E5-INSTRUCT" else corr_topics
            im = ax.imshow(mat, cmap="viridis_r", aspect="equal")
            for i in range(len(topics)):
                for j in range(len(topics)):
                    val = mat[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=5.5,
                                color="white" if val < float(np.nanmedian(mat)) else "black")
            ax.set_xticks(range(len(topics)))
            ax.set_xticklabels(topics, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(len(topics)))
            ax.set_yticklabels(topics, fontsize=6)
            ax.set_title(label, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.7)

        ax = axes[1, 0]
        ax.scatter(master["e5_intra_mean"], master["corr_intra_mean"],
                   s=60, c="#3498db", edgecolors="white", zorder=3)
        for _, row in master.iterrows():
            ax.annotate(row["topic"], (row["e5_intra_mean"], row["corr_intra_mean"]),
                        fontsize=6, ha="left", va="bottom", xytext=(3, 3),
                        textcoords="offset points")
        rho_val, p_val = spearmanr(master["e5_intra_mean"], master["corr_intra_mean"])
        ax.set_xlabel("E5-INSTRUCT intra-topic mean distance", fontsize=9)
        ax.set_ylabel("ANSWER-CORR-ARCCOS intra-topic mean distance", fontsize=9)
        ax.set_title(f"Per-Topic Intra Mean: E5 vs CORR (ρ={rho_val:.3f}, p={p_val:.3f})", fontsize=9)
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        master_s = master.sort_values("e5_rank")
        y_pos = np.arange(len(master_s))
        bar_h = 0.35
        ax.barh(y_pos - bar_h / 2, master_s["e5_rank"], bar_h,
                label="E5-INSTRUCT rank", color="#3498db", edgecolor="white")
        ax.barh(y_pos + bar_h / 2, master_s["corr_rank"], bar_h,
                label="CORR-ARCCOS rank", color="#e74c3c", edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(master_s["topic"].values, fontsize=7)
        ax.set_xlabel("Rank (1 = tightest cluster)", fontsize=9)
        ax.set_title("Topic Ranking: Intra-Topic Cohesion", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.invert_yaxis()

        fig.suptitle("E5-INSTRUCT vs ANSWER-CORR-ARCCOS: Topic Distance Comparison",
                     fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(comp_dir / "topic_distances_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        master.to_csv(comp_dir / "topic_distances_master.csv", index=False)
        print(f"\n  -> {comp_dir}/  (comparison)")


# ---------------------------------------------------------------------------
# §4  Mini vs full-only correlation distributions
# ---------------------------------------------------------------------------


def section4_mini_maxi_correlations(config, top: int):
    """Within-mini vs within-full-only vs cross pairwise |Pearson r| and arccos distance."""
    from vqs.data_loader import load_dataset

    print(f"\n{'=' * 70}")
    print("§4  MINI vs FULL-ONLY CORRELATION DISTRIBUTIONS")
    print(f"{'=' * 70}")

    dataset = load_dataset(config)
    questions_df = dataset["questions"]
    voters_df = dataset["voters"]

    mini_ids = set(questions_df.loc[questions_df["rapide"] == 1, "ID_question"].tolist())
    full_only_ids = set(questions_df.loc[questions_df["rapide"] == 0, "ID_question"].tolist())
    all_ids = sorted(mini_ids | full_only_ids)
    cat_map = questions_df.set_index("ID_question")["_category"].to_dict()
    print(f"  Mini: {len(mini_ids)}  Full-only: {len(full_only_ids)}")

    # Compute pairwise |Pearson r|
    answer_cols = {qid: voters_df[f"answer_{qid}"]
                   for qid in all_ids if f"answer_{qid}" in voters_df.columns}
    pairs = []
    ids = list(answer_cols.keys())
    for i, id1 in enumerate(ids):
        for id2 in ids[i + 1:]:
            mask = answer_cols[id1].notna() & answer_cols[id2].notna()
            if mask.sum() < 10:
                continue
            r, _ = pearsonr(answer_cols[id1][mask], answer_cols[id2][mask])
            abs_r = abs(r)
            pairs.append({
                "ID1": id1, "ID2": id2, "abs_r": abs_r,
                "arccos_dist": float(np.arccos(np.clip(abs_r, 0.0, 1.0))),
                "signed_r": r,
            })

    df_pairs = pd.DataFrame(pairs)
    print(f"  {len(df_pairs)} pairs computed")

    def classify(id1, id2):
        a, b = id1 in mini_ids, id2 in mini_ids
        if a and b:
            return "within-mini"
        if not a and not b:
            return "within-full-only"
        return "cross"

    df_pairs["group"] = df_pairs.apply(lambda r: classify(int(r["ID1"]), int(r["ID2"])), axis=1)
    df_pairs["cat1"] = df_pairs["ID1"].map(cat_map)
    df_pairs["cat2"] = df_pairs["ID2"].map(cat_map)

    # Per-group stats
    groups_order = ["within-mini", "within-full-only", "cross", "all"]
    stats_rows = []
    for g in groups_order:
        subset = df_pairs if g == "all" else df_pairs[df_pairs["group"] == g]
        row = {"group": g, "n_pairs": len(subset)}
        for col in ["abs_r", "arccos_dist"]:
            v = subset[col].values
            row[f"{col}_mean"] = v.mean()
            row[f"{col}_median"] = float(np.median(v))
            row[f"{col}_max"] = v.max()
        stats_rows.append(row)
    df_stats = pd.DataFrame(stats_rows)

    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p(f"  {'Group':<22} {'N':>6} {'Mean |r|':>9} {'Med |r|':>9} {'Max |r|':>9} "
      f"{'Mean dist':>10}")
    p(f"  {'-' * 70}")
    for _, row in df_stats.iterrows():
        p(f"  {row['group']:<22} {int(row['n_pairs']):>6} "
          f"{row['abs_r_mean']:>9.4f} {row['abs_r_median']:>9.4f} {row['abs_r_max']:>9.4f} "
          f"{row['arccos_dist_mean']:>10.4f}")

    top_df = df_pairs.nlargest(top, "abs_r")
    p(f"\nTOP {top} MOST CORRELATED PAIRS")
    p(f"  {'Rank':>4} {'ID1':>8} {'ID2':>8} {'|r|':>7} {'dist':>7} {'Group':<20} {'Categories'}")
    p("-" * 85)
    for rank, (_, row) in enumerate(top_df.iterrows(), 1):
        p(f"  {rank:>4} {int(row['ID1']):>8} {int(row['ID2']):>8} "
          f"{row['abs_r']:>7.4f} {row['arccos_dist']:>7.4f} "
          f"{row['group']:<20} {row['cat1']} / {row['cat2']}")

    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    groups = ["within-mini", "within-full-only", "cross"]
    colors = {"within-mini": "#3498db", "within-full-only": "#e67e22", "cross": "#2ecc71"}
    labels_map = {
        "within-mini": f"Within mini ({len(mini_ids)}q)",
        "within-full-only": f"Within full-only ({len(full_only_ids)}q)",
        "cross": "Cross (mini↔full)",
    }
    box_colors = [colors[g] for g in groups] + ["#95a5a6"]

    for col_idx, col in enumerate(["abs_r", "arccos_dist"]):
        xlabel = "|Pearson r|" if col == "abs_r" else "Arccos distance"
        # Histogram
        ax = axes[0, col_idx]
        for g in groups:
            sub = df_pairs.loc[df_pairs["group"] == g, col]
            ax.hist(sub, bins=40, alpha=0.5, color=colors[g],
                    label=f"{labels_map[g]} (n={len(sub)})", edgecolor="white")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"Pairwise {xlabel} by Group")
        ax.legend(fontsize=8)

        # Box plot
        ax = axes[1, col_idx]
        data = [df_pairs.loc[df_pairs["group"] == g, col].values for g in groups]
        data.append(df_pairs[col].values)
        bp = ax.boxplot(data, labels=[labels_map[g] for g in groups] + ["All"],
                        vert=True, patch_artist=True)
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_ylabel(xlabel)
        ax.set_title(f"Distribution Comparison: {xlabel}")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Mini vs Full-Only Question Correlations (Voter Answers)", fontsize=13, y=1.01)
    fig.tight_layout()

    out = _SEC4_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M")
    base = f"mini_maxi_corr_{ts}"

    df_stats.to_csv(out / f"{base}_stats.csv", index=False)
    df_pairs.sort_values("abs_r", ascending=False).to_csv(out / f"{base}_all_pairs.csv", index=False)
    top_df.to_csv(out / f"{base}_top_pairs.csv", index=False)

    dist_plot_path = out / f"{base}_distributions.png"
    fig.savefig(dist_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 75×75 heatmap
    ordered_ids = sorted(mini_ids) + sorted(full_only_ids)
    n = len(ordered_ids)
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)
    pair_lookup = {}
    for _, row in df_pairs.iterrows():
        pair_lookup[(int(row["ID1"]), int(row["ID2"]))] = row["abs_r"]
        pair_lookup[(int(row["ID2"]), int(row["ID1"]))] = row["abs_r"]
    for i in range(n):
        for j in range(i + 1, n):
            val = pair_lookup.get((ordered_ids[i], ordered_ids[j]), np.nan)
            matrix[i, j] = matrix[j, i] = val

    tick_labels = [f"{qid} ({cat_map.get(qid, '?')[:15]})" for qid in ordered_ids]
    fig2, ax2 = plt.subplots(figsize=(18, 16))
    im = ax2.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=0.8, aspect="equal")
    n_mini = len(mini_ids)
    ax2.axhline(n_mini - 0.5, color="black", linewidth=2)
    ax2.axvline(n_mini - 0.5, color="black", linewidth=2)
    ax2.text(n_mini / 2, -2, "Mini", ha="center", fontsize=11, fontweight="bold")
    ax2.text(n_mini + (n - n_mini) / 2, -2, "Full-only", ha="center", fontsize=11, fontweight="bold")
    ax2.text(-4, n_mini / 2, "Mini", va="center", fontsize=11, fontweight="bold", rotation=90)
    ax2.text(-4, n_mini + (n - n_mini) / 2, "Full-only", va="center", fontsize=11,
             fontweight="bold", rotation=90)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(tick_labels, rotation=90, fontsize=5)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(tick_labels, fontsize=5)
    fig2.colorbar(im, ax=ax2, shrink=0.8).set_label("|Pearson r|")
    ax2.set_title("Pairwise |Pearson r| — Mini (top-left) vs Full-only (bottom-right)", fontsize=12)
    fig2.tight_layout()
    heatmap_path = out / f"{base}_heatmap.png"
    fig2.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    (out / f"{base}_report.txt").write_text("\n".join(lines))
    print(f"\n  -> {out}/  ({base}_stats.csv, _distributions.png, _heatmap.png)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Distance and correlation analysis — four sections (run all or select)"
    )
    parser.add_argument(
        "--section", type=str, default="1,2,3,4",
        help="Comma-separated section numbers to run (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Pipeline config path (required for §1 and §4 — must have load_voters=True)",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top correlated pairs to show in §1 and §4 (default: 20)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    sections = [int(s.strip()) for s in args.section.split(",")]

    config = None
    if any(s in sections for s in [1, 4]):
        if not args.config:
            print("ERROR: --config is required for §1 and/or §4")
            return
        from vqs.config_utils import load_config
        config = load_config(Path(args.config))

    print("=" * 70)
    print(f"Distance & Correlation Analysis — sections: {sections}")
    print("=" * 70)

    if 1 in sections:
        section1_raw_correlations(config, top=args.top)

    if 2 in sections:
        section2_embedding_validation()

    if 3 in sections:
        section3_topic_distances()

    if 4 in sections:
        section4_mini_maxi_correlations(config, top=args.top)

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
