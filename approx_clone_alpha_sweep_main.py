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
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py

    python -m approx_clone_alpha_sweep_main \
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py \
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

from alpha_sweep_main import DEFAULT_ALPHAS, _resolve_n
from cross_run_analysis.analyzer import CrossRunAnalyzer
from main import load_config
from mini_maxi_party_impact_main import (
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
    ("ANSWER-CORR-ARCCOS", "configs/full_pipeline/base_data/pipeline_answer_corr_arccos_ZH.py", None),
    ("E5-INSTRUCT", "configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py", [0.4]),
    ("QWEN3", "configs/full_pipeline/base_data/pipeline_qwen3_ZH.py", [0.6]),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_question_text_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "question_en" in col.lower():
            return col
    raise ValueError("No question text column found")


def _get_clean_name(config) -> str:
    base = Path(config.__file__).stem
    overrides = getattr(config, "overrides", [])
    if overrides:
        suffix = "_".join(overrides).replace("~", "").replace("=", "")
        return f"{base}_{suffix}"
    return base


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
    sweep_df = df[df["metric"] == "ANSWER-CORR-ARCCOS"].sort_values("alpha")
    point_dfs = {
        label: df[df["metric"] == label]
        for label in ["E5-INSTRUCT", "QWEN3"]
        if label in df["metric"].values
    }

    metrics = [
        ("crw_jaccard_mean", "CRW Jaccard (mean)", "baseline_jaccard_mean"),
        ("crw_spearman_mean", "CRW Spearman (mean)", "baseline_spearman_mean"),
        ("crw_kendall_mean", "CRW Kendall (mean)", "baseline_kendall_mean"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    point_colors = {"E5-INSTRUCT": "#2196F3", "QWEN3": "#FF9800"}
    point_markers = {"E5-INSTRUCT": "D", "QWEN3": "s"}

    for ax, (crw_col, label, base_col) in zip(axes, metrics):
        # Sweep curve
        if not sweep_df.empty:
            ax.plot(
                sweep_df["alpha"], sweep_df[crw_col],
                "o-", color="#4CAF50", markersize=4, linewidth=1.5,
                label="ANSWER-CORR-ARCCOS (CRW)",
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


def main():
    args = _parse_args()
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

    # 3. Build augmented dataset
    print("\n--- Building augmented dataset ---")
    augmented_dataset = add_questions_to_mini(mini_dataset, full_dataset, q_ids)
    n_aug_q = len(augmented_dataset["questions"])
    print(f"  Augmented: {n_aug_q} questions ({len(mini_ids)} mini + {top_k} added)")

    # 4. Compute baselines (shared across all metrics)
    print("\n--- Computing mini baseline recommendations ---")
    mini_rec_engine = RecommendationEngine(config=config, data_map=mini_dataset)
    mini_baseline = mini_rec_engine.run_baseline()

    print("\n--- Computing augmented baseline recommendations ---")
    aug_clone_tag = "_".join(str(q) for q in sorted(q_ids))
    aug_config = SimpleNamespace(**vars(config))
    aug_config.clone_id = f"approx_clone_{aug_clone_tag}"
    aug_rec_engine = RecommendationEngine(config=aug_config, data_map=augmented_dataset)
    aug_baseline = aug_rec_engine.run_baseline()

    # 5. Run per-metric sweeps
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

    # 6. Save outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = f"approx_clone_sweep_{timestamp}"

    # CSV — all results
    results_df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / f"{base}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  -> Results CSV: {csv_path}")

    # Selection CSV
    sel_path = RESULTS_DIR / f"{base}_selection.csv"
    selection_df.to_csv(sel_path, index=False)
    print(f"  -> Selection CSV: {sel_path}")

    # Plot
    sns.set_theme(style="whitegrid")
    _plot_metrics(all_rows, RESULTS_DIR, base)

    # Report
    _save_report(
        selection_df, all_rows, weight_lookups, n_jaccard,
        RESULTS_DIR, base,
    )

    print("\n=== Approximate Clone Alpha Sweep Complete ===")


if __name__ == "__main__":
    main()
