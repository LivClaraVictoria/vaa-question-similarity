# -*- coding: utf-8 -*-
"""
Approximate clone alpha sweep: tests whether CRW can detect and correct for
real question redundancy by adding the top-K most correlated full-only questions
to the mini (rapide) questionnaire and measuring recommendation distortion.

Selection criterion: top-K full-only questions by max |Pearson r| to any mini
question (using voter answers).

Metrics tested:
  - ANSWER-CORRELATION-ARCCOS: full alpha sweep (ground truth distance metric)
  - E5-INSTRUCT: single alpha=0.4 (known optimal from exp1)
  - QWEN3: single alpha=0.6 (known optimal from exp1)

Usage:
    python -m approx_clone_alpha_sweep_main \
        --config configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py

    python -m approx_clone_alpha_sweep_main \
        --config configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py \
        --top-k 5 -n 36

Outputs (saved to experiment_results/exp1/approx_clone_alpha_sweep/):
    - approx_clone_sweep_<timestamp>.csv
    - approx_clone_sweep_<timestamp>_selection.csv
    - approx_clone_sweep_<timestamp>_metrics.png
    - approx_clone_sweep_<timestamp>_report.txt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.manifold import MDS

from experiments._common import _get_clean_name, _get_question_text_col, _resolve_n, DEFAULT_ALPHAS
from cross_run_analysis.analyzer import CrossRunAnalyzer
from vqs.config_utils import load_config
from experiments.approximate_clones.partisan_distortion import (
    add_questions_to_mini,
    compute_redundancy_scores,
    filter_to_mini,
)
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine
from vqs.similarity_metrics import get_calculator

sys.path.insert(0, str(Path(__file__).resolve().parent / "dependencies" / "rsfp"))
from dependencies import SVDataFrame

RESULTS_DIR = Path("experiment_results/exp1/approx_clone_alpha_sweep")

# (label, config_path, alpha_or_list)
# None → full alpha sweep with DEFAULT_ALPHAS
METRIC_CONFIGS = [
    ("ANSWER-CORR-ARCCOS-SMOOTHED", "configs/base_pipeline/pipeline_answer_corr_arccos_ZH_smoothed.py", None),
    ("E5-INSTRUCT-SMOOTHED", "configs/base_pipeline/pipeline_e5_instruct_ZH_a03_smoothed.py", [0.4]),
    ("QWEN3-SMOOTHED", "configs/base_pipeline/pipeline_qwen3_ZH_smoothed.py", [0.6]),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Approximate clone alpha sweep: add high-correlation "
        "full-only questions to mini questionnaire and test CRW correction"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Base pipeline config for data loading (e.g. pipeline_e5_instruct_ZH_a03.py)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of highest-correlation questions to add (default: 5)",
    )
    parser.add_argument(
        "-n", type=int, default=None,
        help="Override top-k for Jaccard (default: derived from config)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Question selection
# ---------------------------------------------------------------------------


def _select_top_k_questions(
    full_dataset: dict,
    mini_ids: set[int],
    full_only_ids: list[int],
    top_k: int,
) -> pd.DataFrame:
    """Select top-K full-only questions by max |Pearson r| to any mini question.

    Returns a DataFrame with columns:
        question_id, question_text, category, max_abs_r, mean_abs_r,
        n_high_corr, most_correlated_mini_id
    """
    voters_df = full_dataset["voters"]
    questions_df = full_dataset["questions"]
    text_col = _get_question_text_col(questions_df)

    # Compute redundancy scores
    redundancy = compute_redundancy_scores(voters_df, mini_ids, full_only_ids)

    # Also find which mini question each full-only question is most correlated with
    from scipy.stats import pearsonr

    mini_cols = sorted(f"answer_{qid}" for qid in mini_ids)
    mini_cols = [c for c in mini_cols if c in voters_df.columns]

    best_mini_partner = {}
    for q_id in full_only_ids:
        q_col = f"answer_{q_id}"
        if q_col not in voters_df.columns:
            best_mini_partner[q_id] = None
            continue

        q_vals = voters_df[q_col]
        best_r = 0.0
        best_partner = None
        for mc in mini_cols:
            m_vals = voters_df[mc]
            mask = q_vals.notna() & m_vals.notna()
            if mask.sum() < 10:
                continue
            r, _ = pearsonr(q_vals[mask], m_vals[mask])
            if abs(r) > best_r:
                best_r = abs(r)
                best_partner = int(mc.replace("answer_", ""))
        best_mini_partner[q_id] = best_partner

    # Build selection DataFrame
    rows = []
    for q_id in full_only_ids:
        q_row = questions_df[questions_df["ID_question"] == q_id]
        if q_row.empty:
            continue
        r = redundancy.get(q_id, {})
        rows.append({
            "question_id": q_id,
            "question_text": q_row[text_col].iloc[0],
            "category": q_row["_category"].iloc[0] if "_category" in q_row.columns else "",
            "max_abs_r": r.get("max_abs_r", 0.0),
            "mean_abs_r": r.get("mean_abs_r", 0.0),
            "n_high_corr": r.get("n_high_corr", 0),
            "most_correlated_mini_id": best_mini_partner.get(q_id),
        })

    sel_df = pd.DataFrame(rows).sort_values("max_abs_r", ascending=False)
    return sel_df.head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Correlation overview: pairwise stats + MDS map + heatmap
# ---------------------------------------------------------------------------


def _compute_full_correlation_matrix(
    voters_df: pd.DataFrame,
    all_question_ids: list[int],
) -> pd.DataFrame:
    """Compute pairwise |Pearson r| matrix for all questions using voter answers."""
    n = len(all_question_ids)
    corr_matrix = np.zeros((n, n))

    answer_cols = {
        qid: voters_df[f"answer_{qid}"]
        for qid in all_question_ids
        if f"answer_{qid}" in voters_df.columns
    }

    for i in range(n):
        corr_matrix[i, i] = 1.0
        q_i = all_question_ids[i]
        if q_i not in answer_cols:
            continue
        vals_i = answer_cols[q_i]

        for j in range(i + 1, n):
            q_j = all_question_ids[j]
            if q_j not in answer_cols:
                continue
            vals_j = answer_cols[q_j]
            mask = vals_i.notna() & vals_j.notna()
            if mask.sum() < 10:
                continue
            r, _ = pearsonr(vals_i[mask], vals_j[mask])
            corr_matrix[i, j] = abs(r)
            corr_matrix[j, i] = abs(r)

    return pd.DataFrame(
        corr_matrix,
        index=all_question_ids,
        columns=all_question_ids,
    )


def _compute_correlation_overview(
    full_dataset: dict,
    questions_df: pd.DataFrame,
    mini_ids: set[int],
    selected_ids: list[int],
    all_question_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-question correlation stats and MDS coordinates.

    Returns (overview_df, corr_matrix_df).
    overview_df has one row per question with all stats needed for thesis plots.
    """
    voters_df = full_dataset["voters"]
    text_col = _get_question_text_col(questions_df)
    selected_set = set(selected_ids)

    # Full pairwise |r| matrix
    print("  Computing full pairwise correlation matrix...")
    corr_df = _compute_full_correlation_matrix(voters_df, all_question_ids)

    # Arccos distance matrix for MDS
    abs_r_clipped = np.clip(corr_df.values, 0, 1)
    dist_matrix = np.arccos(abs_r_clipped)
    np.fill_diagonal(dist_matrix, 0.0)

    # MDS to 2D
    print("  Running MDS projection to 2D...")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4)
    coords = mds.fit_transform(dist_matrix)
    stress = mds.stress_

    print(f"  MDS stress: {stress:.2f}")

    # Per-question stats
    rows = []
    for idx, q_id in enumerate(all_question_ids):
        q_row = questions_df[questions_df["ID_question"] == q_id]
        if q_row.empty:
            continue

        is_mini = q_id in mini_ids
        is_selected = q_id in selected_set
        if is_mini:
            group = "mini"
        elif is_selected:
            group = "selected"
        else:
            group = "full_only"

        # Correlations with mini questions
        mini_list = sorted(mini_ids)
        r_to_mini = [corr_df.loc[q_id, m] for m in mini_list if m != q_id and m in corr_df.index]
        max_r_to_mini = max(r_to_mini) if r_to_mini else 0.0
        mean_r_to_mini = np.mean(r_to_mini) if r_to_mini else 0.0

        # Best mini partner
        if r_to_mini:
            best_mini_idx = np.argmax(r_to_mini)
            candidates = [m for m in mini_list if m != q_id and m in corr_df.index]
            best_mini_partner = candidates[best_mini_idx] if candidates else None
        else:
            best_mini_partner = None

        # Correlations with full-only questions (excluding self)
        full_only_list = [q for q in all_question_ids if q not in mini_ids and q != q_id]
        r_to_full_only = [corr_df.loc[q_id, f] for f in full_only_list if f in corr_df.index]
        max_r_to_full_only = max(r_to_full_only) if r_to_full_only else 0.0
        mean_r_to_full_only = np.mean(r_to_full_only) if r_to_full_only else 0.0

        # Correlations with ALL other questions
        all_others = [q for q in all_question_ids if q != q_id]
        r_to_all = [corr_df.loc[q_id, a] for a in all_others if a in corr_df.index]
        max_r_overall = max(r_to_all) if r_to_all else 0.0
        mean_r_overall = np.mean(r_to_all) if r_to_all else 0.0
        n_high_corr = sum(1 for r in r_to_all if r > 0.3)

        # Arccos distance to best mini partner
        arccos_to_best_mini = np.arccos(np.clip(max_r_to_mini, 0, 1))

        rows.append({
            "question_id": q_id,
            "question_text": q_row[text_col].iloc[0],
            "category": q_row["_category"].iloc[0] if "_category" in q_row.columns else "",
            "group": group,
            "is_mini": is_mini,
            "is_selected": is_selected,
            "max_abs_r_to_mini": max_r_to_mini,
            "mean_abs_r_to_mini": mean_r_to_mini,
            "best_mini_partner": best_mini_partner,
            "arccos_to_best_mini": arccos_to_best_mini,
            "max_abs_r_to_full_only": max_r_to_full_only,
            "mean_abs_r_to_full_only": mean_r_to_full_only,
            "max_abs_r_overall": max_r_overall,
            "mean_abs_r_overall": mean_r_overall,
            "n_high_corr_all": n_high_corr,
            "mds_x": coords[idx, 0],
            "mds_y": coords[idx, 1],
            "mds_stress": stress,
        })

    overview_df = pd.DataFrame(rows)
    return overview_df, corr_df


def _plot_question_map(
    overview_df: pd.DataFrame,
    output_dir: Path,
    base: str,
):
    """2D MDS scatter: mini=red, selected=green, rest=blue."""
    fig, ax = plt.subplots(figsize=(12, 10))

    group_config = {
        "mini": {"color": "#E53935", "label": "Mini (rapide)", "marker": "o", "s": 80, "zorder": 3},
        "selected": {"color": "#43A047", "label": "Added (top-k correlated)", "marker": "D", "s": 120, "zorder": 4},
        "full_only": {"color": "#1E88E5", "label": "Full-only (not added)", "marker": "o", "s": 50, "zorder": 2},
    }

    for group, cfg in group_config.items():
        mask = overview_df["group"] == group
        if not mask.any():
            continue
        subset = overview_df[mask]
        ax.scatter(
            subset["mds_x"], subset["mds_y"],
            c=cfg["color"], label=cfg["label"],
            marker=cfg["marker"], s=cfg["s"],
            alpha=0.8, edgecolors="white", linewidth=0.5,
            zorder=cfg["zorder"],
        )

    # Label selected questions
    selected = overview_df[overview_df["group"] == "selected"]
    for _, row in selected.iterrows():
        ax.annotate(
            f"Q{int(row['question_id'])}",
            (row["mds_x"], row["mds_y"]),
            fontsize=7, fontweight="bold",
            xytext=(5, 5), textcoords="offset points",
            color="#2E7D32",
        )

    # Draw lines from selected to their best mini partner
    for _, row in selected.iterrows():
        partner = row["best_mini_partner"]
        if pd.isna(partner):
            continue
        partner_row = overview_df[overview_df["question_id"] == int(partner)]
        if partner_row.empty:
            continue
        ax.plot(
            [row["mds_x"], partner_row["mds_x"].iloc[0]],
            [row["mds_y"], partner_row["mds_y"].iloc[0]],
            color="#43A047", alpha=0.4, linewidth=1, linestyle="--",
            zorder=1,
        )

    stress = overview_df["mds_stress"].iloc[0]
    ax.set_title(
        "Question Map: MDS Projection of Voter-Answer Correlation Distances\n"
        f"(arccos(|Pearson r|), stress={stress:.1f})",
        fontsize=12,
    )
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    path = output_dir / f"{base}_question_map.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Question map: {path.name}")


def _plot_corr_heatmap(
    corr_df: pd.DataFrame,
    mini_ids: set[int],
    selected_ids: list[int],
    questions_df: pd.DataFrame,
    output_dir: Path,
    base: str,
):
    """Correlation heatmap grouped by mini / selected / rest."""
    text_col = _get_question_text_col(questions_df)
    selected_set = set(selected_ids)
    all_ids = list(corr_df.index)

    # Sort: mini first, then selected, then rest
    def sort_key(q_id):
        if q_id in mini_ids:
            return (0, q_id)
        elif q_id in selected_set:
            return (1, q_id)
        return (2, q_id)

    sorted_ids = sorted(all_ids, key=sort_key)
    reordered = corr_df.loc[sorted_ids, sorted_ids]

    # Labels
    labels = []
    for q_id in sorted_ids:
        q_row = questions_df[questions_df["ID_question"] == q_id]
        text = str(q_row[text_col].iloc[0])[:25] if not q_row.empty else ""
        prefix = ""
        if q_id in mini_ids:
            prefix = "[M] "
        elif q_id in selected_set:
            prefix = "[+] "
        labels.append(f"{prefix}Q{q_id} {text}")

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        reordered.values,
        annot=False,
        cmap="YlOrRd",
        vmin=0, vmax=0.8,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.1,
        cbar_kws={"label": "|Pearson r|"},
    )

    # Draw group separator lines
    n_mini = sum(1 for q in sorted_ids if q in mini_ids)
    n_selected = sum(1 for q in sorted_ids if q in selected_set)
    for boundary in [n_mini, n_mini + n_selected]:
        ax.axhline(boundary, color="black", linewidth=1.5)
        ax.axvline(boundary, color="black", linewidth=1.5)

    ax.set_title(
        "Pairwise |Pearson r| Between All Questions\n"
        "[M] = Mini (rapide), [+] = Added to mini, others = full-only",
        fontsize=12,
    )
    ax.tick_params(axis="both", labelsize=5)
    fig.tight_layout()

    path = output_dir / f"{base}_corr_heatmap.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Correlation heatmap: {path.name}")


def _plot_corr_distributions(
    overview_df: pd.DataFrame,
    output_dir: Path,
    base: str,
):
    """Box plots showing correlation distributions by group."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Max |r| to mini by group
    ax = axes[0]
    for group, color in [("mini", "#E53935"), ("selected", "#43A047"), ("full_only", "#1E88E5")]:
        subset = overview_df[overview_df["group"] == group]
        if subset.empty:
            continue
        label = {"mini": "Mini", "selected": "Added", "full_only": "Full-only"}[group]
        ax.boxplot(
            subset["max_abs_r_to_mini"].dropna(),
            positions=[list({"mini": 0, "selected": 1, "full_only": 2}.keys()).index(group)],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="black"),
        )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Mini\n(within-mini)", "Added\n(to mini)", "Full-only\n(to mini)"])
    ax.set_ylabel("Max |Pearson r| to nearest mini question")
    ax.set_title("Redundancy with Mini Questionnaire")

    # 2. Mean |r| overall by group
    ax = axes[1]
    for i, (group, color) in enumerate([("mini", "#E53935"), ("selected", "#43A047"), ("full_only", "#1E88E5")]):
        subset = overview_df[overview_df["group"] == group]
        if subset.empty:
            continue
        ax.boxplot(
            subset["mean_abs_r_overall"].dropna(),
            positions=[i],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="black"),
        )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Mini", "Added", "Full-only"])
    ax.set_ylabel("Mean |Pearson r| to all other questions")
    ax.set_title("Overall Correlation Level")

    # 3. Arccos distance to best mini partner (full-only + selected only)
    ax = axes[2]
    non_mini = overview_df[overview_df["group"] != "mini"]
    for i, (group, color) in enumerate([("selected", "#43A047"), ("full_only", "#1E88E5")]):
        subset = non_mini[non_mini["group"] == group]
        if subset.empty:
            continue
        ax.boxplot(
            subset["arccos_to_best_mini"].dropna(),
            positions=[i],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="black"),
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Added", "Full-only"])
    ax.set_ylabel("Arccos distance to nearest mini question")
    ax.set_title("CRW-Relevant Distance to Mini")
    ax.axhline(np.pi / 2, color="gray", linestyle=":", alpha=0.5, label="max (π/2)")
    ax.legend(fontsize=8)

    fig.suptitle("Correlation Overview: Mini vs Added vs Full-Only Questions", fontsize=13)
    fig.tight_layout()

    path = output_dir / f"{base}_corr_distributions.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Correlation distributions: {path.name}")


# ---------------------------------------------------------------------------
# Per-metric alpha sweep
# ---------------------------------------------------------------------------


def _run_metric_sweep(
    metric_label: str,
    metric_config_path: str,
    alphas: list[float],
    mini_dataset: dict,
    augmented_dataset: dict,
    mini_rec_engine: RecommendationEngine,
    aug_rec_engine: RecommendationEngine,
    mini_baseline: pd.DataFrame,
    aug_baseline: pd.DataFrame,
    n_jaccard: int,
    aug_clone_tag: str,
) -> list[dict]:
    """Run alpha sweep for one metric. Returns list of per-alpha result dicts."""
    metric_config = load_config(Path(metric_config_path))
    calculator = get_calculator(metric_config)

    # Compute distances (once per metric, reused across alphas)
    print(f"  Computing mini distances ({metric_label})...")
    mini_dist = calculator.calculate_distance(mini_dataset, metric_config)

    print(f"  Computing augmented distances ({metric_label})...")
    saved_clone_id = metric_config.clone_id
    metric_config.clone_id = f"approx_clone_{aug_clone_tag}"
    aug_dist = calculator.calculate_distance(augmented_dataset, metric_config)
    metric_config.clone_id = saved_clone_id

    analyzer = CrossRunAnalyzer.from_n(n_jaccard)
    rows = []

    for i, alpha in enumerate(alphas):
        # Mini CRW
        metric_config.alpha = alpha
        mini_reweighter = CloneRobustReweighter(metric_config)
        mini_weights = mini_reweighter.reweight(mini_dist)
        mini_crw = mini_rec_engine.run_crw(mini_weights)

        mini_match_cols = [c for c in mini_crw.columns if "match" in c or "Dist" in c]
        mini_combined = mini_baseline.join(mini_crw[mini_match_cols].add_prefix("CRW_"))

        # Augmented CRW
        aug_config = SimpleNamespace(**vars(metric_config))
        aug_config.alpha = alpha
        aug_config.clone_id = f"approx_clone_{aug_clone_tag}"
        aug_reweighter = CloneRobustReweighter(aug_config)
        aug_weights = aug_reweighter.reweight(aug_dist)
        aug_crw = aug_rec_engine.run_crw(aug_weights)

        aug_match_cols = [c for c in aug_crw.columns if "match" in c or "Dist" in c]
        aug_combined = aug_baseline.join(aug_crw[aug_match_cols].add_prefix("CRW_"))

        # Cross-run analysis (in-memory)
        results = analyzer.analyze_from_dfs(mini_combined, aug_combined)

        row = {
            "metric": metric_label,
            "alpha": alpha,
            "baseline_jaccard_mean": results["base_jaccard"].mean(),
            "baseline_jaccard_median": results["base_jaccard"].median(),
            "baseline_jaccard_p10": results["base_jaccard"].quantile(0.1),
            "baseline_spearman_mean": results["base_spearman"].mean(),
            "baseline_kendall_mean": results["base_kendall"].mean(),
            "crw_jaccard_mean": results["crw_jaccard"].mean(),
            "crw_jaccard_median": results["crw_jaccard"].median(),
            "crw_jaccard_p10": results["crw_jaccard"].quantile(0.1),
            "crw_spearman_mean": results["crw_spearman"].mean(),
            "crw_kendall_mean": results["crw_kendall"].mean(),
        }
        rows.append(row)

        print(
            f"    alpha={alpha:.2f}: "
            f"baseline Jaccard={row['baseline_jaccard_mean']:.4f}, "
            f"CRW Jaccard={row['crw_jaccard_mean']:.4f}, "
            f"CRW Spearman={row['crw_spearman_mean']:.4f}"
        )

    # Also extract CRW weights for the added questions at the last alpha
    weight_lookup = aug_weights.set_index("ID_question")["Weight"].to_dict()

    return rows, weight_lookup


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_metrics(
    all_rows: list[dict],
    output_dir: Path,
    base: str,
):
    """Plot alpha sweep curve for ANSWER-CORR-ARCCOS with E5/QWEN3 as reference points."""
    df = pd.DataFrame(all_rows)

    # Separate sweep metric from single-point metrics
    sweep_df = df[df["metric"] == "ANSWER-CORR-ARCCOS-SMOOTHED"].sort_values("alpha")
    point_dfs = {
        label: df[df["metric"] == label]
        for label in ["E5-INSTRUCT-SMOOTHED", "QWEN3-SMOOTHED"]
        if label in df["metric"].values
    }

    metrics = [
        ("crw_jaccard_mean", "CRW Jaccard (mean)", "baseline_jaccard_mean"),
        ("crw_spearman_mean", "CRW Spearman (mean)", "baseline_spearman_mean"),
        ("crw_kendall_mean", "CRW Kendall (mean)", "baseline_kendall_mean"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    point_colors = {"E5-INSTRUCT-SMOOTHED": "#2196F3", "QWEN3-SMOOTHED": "#FF9800"}
    point_markers = {"E5-INSTRUCT-SMOOTHED": "D", "QWEN3-SMOOTHED": "s"}

    for ax, (crw_col, label, base_col) in zip(axes, metrics):
        # Sweep curve
        if not sweep_df.empty:
            ax.plot(
                sweep_df["alpha"], sweep_df[crw_col],
                "o-", color="#4CAF50", markersize=4, linewidth=1.5,
                label="ANSWER-CORR-ARCCOS-SMOOTHED (CRW)",
            )

            # Baseline as dashed line
            baseline_val = sweep_df[base_col].iloc[0]
            ax.axhline(
                baseline_val, color="gray", linestyle="--", linewidth=1,
                label=f"Baseline ({baseline_val:.4f})",
            )

        # Single-point metrics
        for pt_label, pt_df in point_dfs.items():
            if not pt_df.empty:
                ax.scatter(
                    pt_df["alpha"].values, pt_df[crw_col].values,
                    color=point_colors.get(pt_label, "red"),
                    marker=point_markers.get(pt_label, "o"),
                    s=100, zorder=5,
                    label=f"{pt_label} (α={pt_df['alpha'].iloc[0]:.1f})",
                )

        ax.set_xlabel("Alpha")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Approximate Clone Experiment: CRW Correction of High-Correlation Question Additions\n"
        "(mini questionnaire + top-5 most correlated full-only questions)",
        fontsize=12,
    )
    fig.tight_layout()

    path = output_dir / f"{base}_metrics.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Metrics plot: {path.name}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _save_report(
    selection_df: pd.DataFrame,
    all_rows: list[dict],
    weight_lookups: dict[str, dict],
    n_jaccard: int,
    output_dir: Path,
    base: str,
):
    """Save human-readable summary report."""
    df = pd.DataFrame(all_rows)
    q_ids = selection_df["question_id"].tolist()

    lines = [
        "=" * 80,
        "APPROXIMATE CLONE EXPERIMENT — REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Top-k for Jaccard: {n_jaccard}",
        "",
        "SELECTED QUESTIONS (top-5 by max |Pearson r| to mini questions):",
        "-" * 80,
    ]

    for _, row in selection_df.iterrows():
        lines.append(
            f"  Q{int(row['question_id'])}: max_abs_r={row['max_abs_r']:.3f} "
            f"(with mini Q{int(row['most_correlated_mini_id']) if pd.notna(row['most_correlated_mini_id']) else '?'}), "
            f"mean_abs_r={row['mean_abs_r']:.3f}"
        )
        lines.append(f'    "{str(row["question_text"])[:70]}"')
        lines.append(f"    Category: {row.get('category', 'N/A')}")
        lines.append("")

    # Per-metric results
    for metric_label in df["metric"].unique():
        mdf = df[df["metric"] == metric_label]
        lines.extend([
            "=" * 80,
            f"METRIC: {metric_label}",
            "-" * 80,
        ])

        baseline_jac = mdf["baseline_jaccard_mean"].iloc[0]
        lines.append(f"  Baseline distortion (Jaccard mean): {baseline_jac:.4f}")
        lines.append(f"  Baseline distortion (Spearman mean): {mdf['baseline_spearman_mean'].iloc[0]:.4f}")

        if len(mdf) > 1:
            # Alpha sweep
            best_idx = mdf["crw_jaccard_mean"].idxmax()
            best = mdf.loc[best_idx]
            lines.append(f"  Best CRW alpha: {best['alpha']:.2f}")
            lines.append(f"  Best CRW Jaccard mean: {best['crw_jaccard_mean']:.4f}")
            lines.append(f"  Best CRW Spearman mean: {best['crw_spearman_mean']:.4f}")
            improvement = best["crw_jaccard_mean"] - baseline_jac
            lines.append(f"  Jaccard improvement over baseline: {improvement:+.4f}")
        else:
            row = mdf.iloc[0]
            lines.append(f"  Alpha: {row['alpha']:.2f}")
            lines.append(f"  CRW Jaccard mean: {row['crw_jaccard_mean']:.4f}")
            lines.append(f"  CRW Spearman mean: {row['crw_spearman_mean']:.4f}")
            improvement = row["crw_jaccard_mean"] - baseline_jac
            lines.append(f"  Jaccard improvement over baseline: {improvement:+.4f}")

        # CRW weights for added questions
        if metric_label in weight_lookups:
            wl = weight_lookups[metric_label]
            lines.append(f"  CRW weights for added questions:")
            for q_id in q_ids:
                w = wl.get(q_id, None)
                lines.append(f"    Q{q_id}: {w:.4f}" if w is not None else f"    Q{q_id}: N/A")

        lines.append("")

    lines.append("=" * 80)

    path = output_dir / f"{base}_report.txt"
    path.write_text("\n".join(lines))
    print(f"  -> Report: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = _parse_args(argv)
    config = load_config(Path(args.config))
    n_jaccard = _resolve_n(config, args.n)
    top_k = args.top_k

    print("\n=== Approximate Clone Alpha Sweep ===")
    print(f"  Config: {args.config}")
    print(f"  Top-k questions: {top_k}")
    print(f"  Jaccard n: {n_jaccard}")

    # 1. Load dataset and filter to mini
    print("\n--- Loading full dataset ---")
    full_dataset = load_dataset(config)

    print("\n--- Filtering to mini questionnaire ---")
    mini_dataset, mini_ids, full_only_ids = filter_to_mini(full_dataset)
    print(f"  Mini questions: {len(mini_ids)}, Full-only: {len(full_only_ids)}")

    # 2. Select top-K questions by max |r| to mini
    print(f"\n--- Selecting top-{top_k} high-correlation questions ---")
    selection_df = _select_top_k_questions(
        full_dataset, mini_ids, full_only_ids, top_k
    )
    q_ids = selection_df["question_id"].tolist()

    print("  Selected questions:")
    for _, row in selection_df.iterrows():
        print(
            f"    Q{int(row['question_id'])}: max_abs_r={row['max_abs_r']:.3f} "
            f"(with mini Q{int(row['most_correlated_mini_id']) if pd.notna(row['most_correlated_mini_id']) else '?'}), "
            f"category={row['category']}"
        )

    # 3. Correlation overview (before building augmented dataset)
    print("\n--- Computing correlation overview ---")
    all_question_ids = sorted(mini_ids | set(full_only_ids))
    overview_df, corr_matrix_df = _compute_correlation_overview(
        full_dataset,
        full_dataset["questions"],
        mini_ids,
        q_ids,
        all_question_ids,
    )

    # 4. Build augmented dataset
    print("\n--- Building augmented dataset ---")
    augmented_dataset = add_questions_to_mini(mini_dataset, full_dataset, q_ids)
    n_aug_q = len(augmented_dataset["questions"])
    print(f"  Augmented: {n_aug_q} questions ({len(mini_ids)} mini + {top_k} added)")

    # 5. Compute baselines (shared across all metrics)
    print("\n--- Computing mini baseline recommendations ---")
    mini_rec_engine = RecommendationEngine(config=config, data_map=mini_dataset)
    mini_baseline = mini_rec_engine.run_baseline()

    print("\n--- Computing augmented baseline recommendations ---")
    aug_clone_tag = "_".join(str(q) for q in sorted(q_ids))
    aug_config = SimpleNamespace(**vars(config))
    aug_config.clone_id = f"approx_clone_{aug_clone_tag}"
    aug_rec_engine = RecommendationEngine(config=aug_config, data_map=augmented_dataset)
    aug_baseline = aug_rec_engine.run_baseline()

    # 6. Run per-metric sweeps
    all_rows = []
    weight_lookups = {}

    for metric_label, config_path, alphas_spec in METRIC_CONFIGS:
        alphas = DEFAULT_ALPHAS if alphas_spec is None else alphas_spec
        print(f"\n{'='*60}")
        print(f"  Metric: {metric_label} — {len(alphas)} alpha(s)")
        print(f"{'='*60}")

        rows, wl = _run_metric_sweep(
            metric_label=metric_label,
            metric_config_path=config_path,
            alphas=alphas,
            mini_dataset=mini_dataset,
            augmented_dataset=augmented_dataset,
            mini_rec_engine=mini_rec_engine,
            aug_rec_engine=aug_rec_engine,
            mini_baseline=mini_baseline,
            aug_baseline=aug_baseline,
            n_jaccard=n_jaccard,
            aug_clone_tag=aug_clone_tag,
        )
        all_rows.extend(rows)
        weight_lookups[metric_label] = wl

    # 7. Save outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = f"approx_clone_sweep_{timestamp}"

    # Correlation overview CSV (single file with all per-question data)
    overview_path = RESULTS_DIR / f"{base}_corr_overview.csv"
    overview_df.to_csv(overview_path, index=False)
    print(f"\n  -> Correlation overview CSV: {overview_path}")

    # Alpha sweep results CSV
    results_df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / f"{base}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  -> Results CSV: {csv_path}")

    # Selection CSV
    sel_path = RESULTS_DIR / f"{base}_selection.csv"
    selection_df.to_csv(sel_path, index=False)
    print(f"  -> Selection CSV: {sel_path}")

    # Plots
    sns.set_theme(style="whitegrid")
    _plot_question_map(overview_df, RESULTS_DIR, base)
    _plot_corr_heatmap(
        corr_matrix_df, mini_ids, q_ids,
        full_dataset["questions"], RESULTS_DIR, base,
    )
    _plot_corr_distributions(overview_df, RESULTS_DIR, base)
    _plot_metrics(all_rows, RESULTS_DIR, base)

    # Report
    _save_report(
        selection_df, all_rows, weight_lookups, n_jaccard,
        RESULTS_DIR, base,
    )

    print("\n=== Approximate Clone Alpha Sweep Complete ===")


if __name__ == "__main__":
    main()
