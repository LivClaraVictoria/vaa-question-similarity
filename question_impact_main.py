"""
Question impact sweep: identifies which questions, when cloned 10 times,
cause the most dramatic change in voter recommendations.

Supports three modes:
    - sweep (default): Run all questions sequentially in one process.
    - worker:  Compute a single question (by --task-id index). For SLURM job arrays.
    - collect: Read per-question worker CSVs from --sweep-dir, aggregate, and plot.

Usage:
    # Sequential:
    python -m question_impact_main \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py

    # Worker (one question, for SLURM array):
    python -m question_impact_main --mode worker --task-id 3 \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --sweep-dir /path/to/sweep_dir

    # Collect (aggregate workers + plot):
    python -m question_impact_main --mode collect \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --sweep-dir /path/to/sweep_dir

Outputs (saved to experiment_results/question_impact_results/):
    - question_impact_*.csv          — per-question metrics sorted by composite rank
    - question_impact_*_ranking.png  — impact ranking bar chart
    - question_impact_*_correlation_analysis.png — scatter plots of predictors vs impact
    - question_impact_*_corr_matrices.png — answer correlation heatmaps
    - question_impact_*_report.txt   — top-10 worst-case summary
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, kendalltau

from clone_pipeline.applicator import apply_specs
from clone_pipeline.spec import CloneSpec
from configs import base_constants as default_config
from cross_run_analysis.analyzer import CrossRunAnalyzer
from main import load_config
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine

N_CLONES = 10
RESULTS_DIR = default_config.RESULTS_DIR / "question_impact_results"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Question impact sweep: find worst-case questions to clone"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Base pipeline config (e.g. configs/full_pipeline/base_data/pipeline_e5_ZH.py)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["sweep", "worker", "collect"], default="sweep",
        help="Execution mode: sweep (sequential), worker (single question), collect (aggregate + plot)",
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="Question index for worker mode (typically SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--sweep-dir", type=str, default=None,
        help="Directory for per-question worker CSVs (worker writes here, collect reads from here)",
    )
    parser.add_argument(
        "-n", type=int, default=None,
        help="Override top-k for Jaccard (default: derived from config)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_clean_name(config) -> str:
    base = Path(config.__file__).stem
    overrides = getattr(config, "overrides", [])
    if overrides:
        suffix = "_".join(overrides).replace("~", "").replace("=", "")
        return f"{base}_{suffix}"
    return base


def _resolve_n(config, n_override: int | None) -> int:
    if n_override is not None:
        return n_override
    if config.n_recommendations == "all":
        if config.district != "all":
            seats = (
                config.SEATS_PER_CANTON.get(config.district)
                if config.data_year == 2023
                else config.SEATS_PER_CANTON19.get(config.district)
            )
            return seats if seats else 30
        return 30
    elif config.n_recommendations is not None:
        return config.n_recommendations
    return 30


def _get_question_text_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "question_en" in col.lower():
            return col
    raise ValueError("No question text column found")


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _setup_pipeline(config):
    """Load dataset, compute base baseline recs, pre-compute question stats."""
    print("\n--- Loading dataset ---")
    dataset = load_dataset(config)

    print("\n--- Computing base baseline recommendations ---")
    rec_engine = RecommendationEngine(config=config, data_map=dataset)
    base_recs = rec_engine.run_baseline()

    # Extract ranked lists from base recs (only standard, no CRW columns)
    base_rankings = CrossRunAnalyzer._extract_rankings(base_recs)

    # Get sorted question IDs (excluding clones) for deterministic task-id mapping
    questions_df = dataset["questions"]
    question_ids = sorted(
        questions_df.loc[questions_df["ID_question"] < 9_000_000, "ID_question"].tolist()
    )

    print(f"\n  Questions: {len(question_ids)}")

    # Pre-compute NaN rates
    nan_info = {}
    df_voters = dataset["voters"]
    df_candidates = dataset["candidates"]
    for q_id in question_ids:
        ans_col = f"answer_{q_id}"
        voter_nan = df_voters[ans_col].isna().mean() if ans_col in df_voters.columns else 1.0
        cand_nan = df_candidates[ans_col].isna().mean() if ans_col in df_candidates.columns else 1.0
        nan_info[q_id] = {"voter_nan_pct": voter_nan, "candidate_nan_pct": cand_nan}

    # Pre-compute variance stats
    var_info = {}
    for q_id in question_ids:
        ans_col = f"answer_{q_id}"
        cand_var = df_candidates[ans_col].var() if ans_col in df_candidates.columns else np.nan
        voter_var = df_voters[ans_col].var() if ans_col in df_voters.columns else np.nan
        combined = cand_var * voter_var if pd.notna(cand_var) and pd.notna(voter_var) else np.nan
        var_info[q_id] = {
            "candidate_var": cand_var, "voter_var": voter_var, "combined_var": combined,
        }

    return {
        "dataset": dataset,
        "base_rankings": base_rankings,
        "question_ids": question_ids,
        "questions_df": questions_df,
        "nan_info": nan_info,
        "var_info": var_info,
    }


# ---------------------------------------------------------------------------
# Per-question comparison
# ---------------------------------------------------------------------------


def _compare_baseline_only(
    base_rankings: pd.DataFrame,
    cloned_recs: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """Compare two baseline-only recommendation DataFrames using Jaccard, Spearman, Kendall."""
    cloned_rankings = CrossRunAnalyzer._extract_rankings(cloned_recs)

    # Join on voterID
    merged = base_rankings.join(cloned_rankings, lsuffix="_a", rsuffix="_b", how="inner")

    # Create a lightweight analyzer just for its metric methods
    analyzer = CrossRunAnalyzer.from_n(n)

    results = []
    for vid, std_a, std_b in zip(
        merged.index, merged["ranked_standard_a"], merged["ranked_standard_b"]
    ):
        jac = analyzer._jaccard(std_a, std_b, n)
        rank = analyzer._rank_stats(std_a, std_b)
        pos = analyzer._position_stats(std_a, std_b)

        results.append({
            "voterID": vid,
            "jaccard": jac,
            "swaps": analyzer._swaps(jac),
            "spearman": rank.get("spearman", np.nan),
            "kendall": rank.get("kendall", np.nan),
            "any_rank_change": pos.get("any_change", np.nan),
            "n_changed": pos.get("n_changed", np.nan),
            "avg_pos_moved": pos.get("avg_pos_moved", np.nan),
            "max_pos_moved": pos.get("max_pos_moved", np.nan),
        })

    return pd.DataFrame(results)


def _compute_question_impact(q_id: int, config, pipeline: dict, n: int) -> dict:
    """Clone a single question 10x, compute baseline recs, compare with base."""
    dataset = pipeline["dataset"]
    text_col = _get_question_text_col(pipeline["questions_df"])

    # 1. Clone in-memory (10 identical clones)
    spec = CloneSpec(source_q_id=q_id, clone_type="identical", n_clones=N_CLONES)
    cloned_data = apply_specs(
        specs=[spec],
        dataframes={
            "questions": dataset["questions"],
            "voters": dataset["voters"],
            "candidates": dataset["candidates"],
        },
    )

    # 2. Compute cloned baseline recs
    cloned_engine = RecommendationEngine(config=config, data_map=cloned_data)
    cloned_recs = cloned_engine.run_baseline()

    sys.stdout.flush()

    # 3. Compare
    per_voter = _compare_baseline_only(pipeline["base_rankings"], cloned_recs, n)

    # 4. Build summary row
    q_text = pipeline["questions_df"].loc[
        pipeline["questions_df"]["ID_question"] == q_id, text_col
    ].iloc[0]
    nan = pipeline["nan_info"][q_id]
    var = pipeline["var_info"][q_id]
    # 1-based position when questions are ordered by ID (reflects survey order)
    question_order = pipeline["question_ids"].index(q_id) + 1

    return {
        "question_id": q_id,
        "question_text": q_text,
        "question_order": question_order,
        "jaccard_mean": per_voter["jaccard"].mean(),
        "jaccard_median": per_voter["jaccard"].median(),
        "jaccard_p10": per_voter["jaccard"].quantile(0.1),
        "spearman_mean": per_voter["spearman"].mean(),
        "kendall_mean": per_voter["kendall"].mean(),
        "any_rank_change_pct": per_voter["any_rank_change"].mean(),
        "n_changed_mean": per_voter["n_changed"].mean(),
        "avg_pos_moved_mean": per_voter["avg_pos_moved"].mean(),
        "max_pos_moved_mean": per_voter["max_pos_moved"].mean(),
        "voter_nan_pct": nan["voter_nan_pct"],
        "candidate_nan_pct": nan["candidate_nan_pct"],
        "candidate_var": var["candidate_var"],
        "voter_var": var["voter_var"],
        "combined_var": var["combined_var"],
    }


# ---------------------------------------------------------------------------
# Mode: sweep (sequential)
# ---------------------------------------------------------------------------


def _run_sweep(args, config, n: int):
    pipeline = _setup_pipeline(config)
    question_ids = pipeline["question_ids"]

    rows = []
    for i, q_id in enumerate(question_ids):
        print(f"\n--- Question {q_id} ({i + 1}/{len(question_ids)}) ---")
        row = _compute_question_impact(q_id, config, pipeline, n)
        rows.append(row)
        print(
            f"  Jaccard={row['jaccard_mean']:.4f}, "
            f"Spearman={row['spearman_mean']:.4f}, "
            f"Kendall={row['kendall_mean']:.4f}"
        )

    sweep_df = pd.DataFrame(rows)

    # Use collect logic for output
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_collect_outputs(sweep_df, config, n, output_dir, pipeline["dataset"])

    print("\n=== Question Impact Sweep Complete ===")


# ---------------------------------------------------------------------------
# Mode: worker (single question for SLURM array)
# ---------------------------------------------------------------------------


def _run_worker(args, config, n: int):
    task_id = args.task_id
    if task_id is None:
        print("ERROR: --task-id is required in worker mode", file=sys.stderr)
        sys.exit(1)

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "workers"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _setup_pipeline(config)
    question_ids = pipeline["question_ids"]

    if task_id < 0 or task_id >= len(question_ids):
        print(
            f"ERROR: --task-id {task_id} out of range [0, {len(question_ids) - 1}]",
            file=sys.stderr,
        )
        sys.exit(1)

    q_id = question_ids[task_id]
    out_path = sweep_dir / f"question_worker_{task_id:03d}_q{q_id}.csv"

    # Skip if already computed
    if out_path.exists():
        print(f"  [SKIP] Worker CSV already exists: {out_path}")
        return

    print(f"\n=== Worker: question {q_id} (task {task_id}/{len(question_ids) - 1}) ===")

    row = _compute_question_impact(q_id, config, pipeline, n)

    worker_df = pd.DataFrame([row])
    worker_df.to_csv(out_path, index=False)

    print(f"\n  -> Worker CSV: {out_path}")
    print(
        f"  Jaccard={row['jaccard_mean']:.4f}, "
        f"Spearman={row['spearman_mean']:.4f}, "
        f"Kendall={row['kendall_mean']:.4f}"
    )
    print(f"\n=== Worker Complete ===")


# ---------------------------------------------------------------------------
# Mode: collect (aggregate + plot + correlation analysis)
# ---------------------------------------------------------------------------


def _run_collect(args, config, n: int):
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "workers"

    worker_files = sorted(sweep_dir.glob("question_worker_*.csv"))
    if not worker_files:
        print(f"ERROR: No worker CSVs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Collect: reading {len(worker_files)} worker CSVs from {sweep_dir} ===")

    dfs = [pd.read_csv(f) for f in worker_files]
    combined = pd.concat(dfs, ignore_index=True)

    # Load dataset for correlation analysis
    print("\n--- Loading dataset for correlation analysis ---")
    dataset = load_dataset(config)

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_collect_outputs(combined, config, n, output_dir, dataset)

    print("\n=== Collect Complete ===")


# ---------------------------------------------------------------------------
# Collect output generation
# ---------------------------------------------------------------------------


def _save_collect_outputs(
    df: pd.DataFrame,
    config,
    n: int,
    output_dir: Path,
    dataset: dict,
):
    """Compute composite ranks, correlation analysis, and save all outputs."""
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = f"question_impact_{name}_{timestamp}"

    # --- Composite rank ---
    df["rank_jaccard"] = df["jaccard_mean"].rank()  # lower jaccard = rank 1
    df["rank_spearman"] = df["spearman_mean"].rank()
    df["rank_kendall"] = df["kendall_mean"].rank()
    df["composite_rank"] = (
        df["rank_jaccard"] + df["rank_spearman"] + df["rank_kendall"]
    ) / 3
    df["impact"] = 1 - df["jaccard_mean"]

    # --- Answer correlation analysis ---
    print("\n--- Computing answer correlation analysis ---")
    df_voters = dataset["voters"]
    df_candidates = dataset["candidates"]
    question_ids = sorted(df["question_id"].tolist())

    voter_corr, cand_corr, voter_avg_corr, cand_avg_corr = _compute_answer_correlations(
        df_voters, df_candidates, question_ids
    )

    # Merge correlation features into main df
    df["voter_avg_abs_corr"] = df["question_id"].map(voter_avg_corr)
    df["candidate_avg_abs_corr"] = df["question_id"].map(cand_avg_corr)

    # Sort by composite rank
    df = df.sort_values("composite_rank").reset_index(drop=True)

    # --- Save CSV ---
    csv_path = output_dir / f"{base}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> CSV: {csv_path.name}")

    # --- Plots ---
    sns.set_theme(style="whitegrid")

    _plot_ranking(df, output_dir, base)
    _plot_correlation_analysis(df, output_dir, base)
    _plot_corr_matrices(voter_corr, cand_corr, question_ids, output_dir, base)
    _save_report(df, output_dir, base, n)


def _compute_answer_correlations(
    df_voters: pd.DataFrame,
    df_candidates: pd.DataFrame,
    question_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Compute Pearson correlation matrices and per-question average |correlation|."""
    voter_ans_cols = [f"answer_{q}" for q in question_ids if f"answer_{q}" in df_voters.columns]
    cand_ans_cols = [f"answer_{q}" for q in question_ids if f"answer_{q}" in df_candidates.columns]

    voter_corr = df_voters[voter_ans_cols].corr()
    cand_corr = df_candidates[cand_ans_cols].corr()

    # Per-question average absolute correlation (excluding self-correlation on diagonal)
    voter_avg_corr = {}
    cand_avg_corr = {}

    for q_id in question_ids:
        col = f"answer_{q_id}"
        if col in voter_corr.columns:
            vals = voter_corr[col].drop(col, errors="ignore").abs()
            voter_avg_corr[q_id] = vals.mean() if len(vals) > 0 else np.nan
        else:
            voter_avg_corr[q_id] = np.nan

        if col in cand_corr.columns:
            vals = cand_corr[col].drop(col, errors="ignore").abs()
            cand_avg_corr[q_id] = vals.mean() if len(vals) > 0 else np.nan
        else:
            cand_avg_corr[q_id] = np.nan

    return voter_corr, cand_corr, voter_avg_corr, cand_avg_corr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_ranking(df: pd.DataFrame, output_dir: Path, base: str):
    """Horizontal bar chart: questions ranked by composite impact."""
    df_sorted = df.sort_values("composite_rank").head(30)  # top 30

    fig, ax = plt.subplots(figsize=(14, max(8, len(df_sorted) * 0.35)))

    y = np.arange(len(df_sorted))
    height = 0.25

    # Plot three metrics side by side (as "distortion" = 1 - metric)
    ax.barh(y - height, 1 - df_sorted["jaccard_mean"], height, label="1 - Jaccard", color="#E53935")
    ax.barh(y, 1 - df_sorted["spearman_mean"], height, label="1 - Spearman", color="#1E88E5")
    ax.barh(y + height, 1 - df_sorted["kendall_mean"], height, label="1 - Kendall", color="#43A047")

    # Labels: question ID + truncated text
    labels = []
    for _, row in df_sorted.iterrows():
        text = str(row["question_text"])[:60]
        labels.append(f"Q{int(row['question_id'])}  {text}")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Distortion (higher = more impact from cloning)")
    ax.set_title(f"Question Impact Ranking (top {len(df_sorted)}, {N_CLONES} identical clones)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = output_dir / f"{base}_ranking.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Ranking plot: {path.name}")


def _plot_correlation_analysis(df: pd.DataFrame, output_dir: Path, base: str):
    """Multi-panel scatter: predictors vs impact."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    scatter_configs = [
        ("combined_var", "Combined Variance", axes[0, 0]),
        ("voter_var", "Voter Variance", axes[0, 1]),
        ("candidate_var", "Candidate Variance", axes[0, 2]),
        ("voter_nan_pct", "Voter NaN %\n(fraction who skipped this question)", axes[1, 0]),
        ("question_order", "Question Position in Survey\n(1 = first question)", axes[1, 1]),
        ("voter_avg_abs_corr", "Question Redundancy\n(mean |Pearson r| of voter answers vs all other questions)", axes[1, 2]),
    ]

    for col, label, ax in scatter_configs:
        if col not in df.columns or df[col].isna().all():
            ax.set_visible(False)
            continue

        x = df[col].values
        y = df["impact"].values
        mask = ~(np.isnan(x) | np.isnan(y))

        ax.scatter(x[mask], y[mask], alpha=0.6, s=40, color="#1565C0")

        # Add Spearman correlation
        if mask.sum() >= 3:
            rho, p = spearmanr(x[mask], y[mask])
            ax.set_title(f"{label}\n(Spearman r={rho:.3f}, p={p:.3f})", fontsize=10)
        else:
            ax.set_title(label, fontsize=10)

        ax.set_xlabel(label)
        ax.set_ylabel("Impact (1 - Jaccard mean)")

        # Annotate top-3 impact questions
        top3 = df.nlargest(3, "impact")
        for _, row in top3.iterrows():
            if pd.notna(row[col]):
                ax.annotate(
                    f"Q{int(row['question_id'])}",
                    (row[col], row["impact"]),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points",
                )

    fig.suptitle(
        f"Predictors of Clone Impact ({N_CLONES} identical clones)\n"
        f"Y-axis: impact = 1 − mean Jaccard (higher = more disruption to recommendations)",
        fontsize=12,
    )
    fig.tight_layout()

    path = output_dir / f"{base}_correlation_analysis.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Correlation analysis: {path.name}")


def _plot_corr_matrices(
    voter_corr: pd.DataFrame,
    cand_corr: pd.DataFrame,
    question_ids: list[int],
    output_dir: Path,
    base: str,
):
    """Heatmaps of voter and candidate answer correlation matrices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Shorten column labels to question IDs
    def _shorten(cols):
        return [c.replace("answer_", "Q") for c in cols]

    v = voter_corr.copy()
    v.columns = _shorten(v.columns)
    v.index = _shorten(v.index)
    sns.heatmap(v, ax=ax1, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.6},
                xticklabels=True, yticklabels=True)
    ax1.set_title("Voter Answer Correlations")
    ax1.tick_params(labelsize=6)

    c = cand_corr.copy()
    c.columns = _shorten(c.columns)
    c.index = _shorten(c.index)
    sns.heatmap(c, ax=ax2, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.6},
                xticklabels=True, yticklabels=True)
    ax2.set_title("Candidate Answer Correlations")
    ax2.tick_params(labelsize=6)

    fig.suptitle("Answer Correlation Matrices (Pearson)", fontsize=13)
    fig.tight_layout()

    path = output_dir / f"{base}_corr_matrices.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Correlation matrices: {path.name}")


def _save_report(df: pd.DataFrame, output_dir: Path, base: str, n: int, top_n: int = 10):
    """Save human-readable top-N worst-case report."""
    lines = [
        "=" * 80,
        "QUESTION IMPACT SWEEP REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Clone type: identical, n_clones: {N_CLONES}",
        f"Jaccard top-k (n): {n}",
        f"Questions tested: {len(df)}",
        "",
        f"TOP {top_n} WORST-CASE QUESTIONS (by composite rank):",
        "-" * 80,
        f"{'Rank':>4}  {'QID':>6}  {'Jaccard':>8}  {'Spearman':>8}  {'Kendall':>8}  "
        f"{'V.NaN%':>6}  {'C.NaN%':>6}  {'CombVar':>8}  Text",
        "-" * 80,
    ]

    for i, (_, row) in enumerate(df.head(top_n).iterrows()):
        text = str(row["question_text"])[:50]
        lines.append(
            f"{i + 1:>4}  {int(row['question_id']):>6}  "
            f"{row['jaccard_mean']:>8.4f}  {row['spearman_mean']:>8.4f}  "
            f"{row['kendall_mean']:>8.4f}  "
            f"{row['voter_nan_pct'] * 100:>5.1f}%  {row['candidate_nan_pct'] * 100:>5.1f}%  "
            f"{row['combined_var']:>8.1f}  {text}"
        )

    lines.extend([
        "",
        "=" * 80,
        "FULL STATISTICS:",
        f"  Mean Jaccard:   {df['jaccard_mean'].mean():.4f}",
        f"  Median Jaccard: {df['jaccard_mean'].median():.4f}",
        f"  Min Jaccard:    {df['jaccard_mean'].min():.4f} (Q{int(df.loc[df['jaccard_mean'].idxmin(), 'question_id'])})",
        f"  Max Jaccard:    {df['jaccard_mean'].max():.4f} (Q{int(df.loc[df['jaccard_mean'].idxmax(), 'question_id'])})",
        "",
    ])

    # Correlation summary
    if "voter_avg_abs_corr" in df.columns:
        for feat in ["combined_var", "voter_nan_pct", "candidate_nan_pct",
                      "voter_avg_abs_corr", "candidate_avg_abs_corr"]:
            if feat in df.columns:
                mask = df[feat].notna() & df["impact"].notna()
                if mask.sum() >= 3:
                    rho, p = spearmanr(df.loc[mask, feat], df.loc[mask, "impact"])
                    lines.append(f"  Spearman(impact, {feat}): r={rho:.3f}, p={p:.3f}")

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
    n = _resolve_n(config, args.n)
    name = _get_clean_name(config)

    print(f"\n=== Question Impact Sweep ({args.mode} mode) ===")
    print(f"  Config : {name}")
    print(f"  Top-k (n): {n}")
    print(f"  Clones per question: {N_CLONES}")

    if args.mode == "sweep":
        _run_sweep(args, config, n)
    elif args.mode == "worker":
        _run_worker(args, config, n)
    elif args.mode == "collect":
        _run_collect(args, config, n)


if __name__ == "__main__":
    main()
