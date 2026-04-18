"""
Clone count sweep (all questions): measures how recommendation distortion
scales with the number of identical clones for every question in the dataset.

2D sweep: question_id × n_clones. Baseline-only comparison (no CRW).

Supports three modes:
    - sweep (default): Run all questions sequentially in one process.
    - worker:  Compute a single question (by --task-id index). For SLURM job arrays.
    - collect: Read per-question worker CSVs from --sweep-dir, aggregate, and plot.

Usage:
    # Sequential (all questions):
    python -m clone_count_sweep_main \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py

    # Worker (one question, for SLURM array):
    python -m clone_count_sweep_main --mode worker --task-id 3 \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --sweep-dir /path/to/sweep_dir

    # Collect (aggregate workers + plot):
    python -m clone_count_sweep_main --mode collect \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --sweep-dir /path/to/sweep_dir

Outputs (saved to experiment_results/clone_count_sweep/):
    - clone_count_sweep_all_*.csv                  — one row per (question, n_clones)
    - clone_count_sweep_all_*_metric_comparison.png — 1×3: Jaccard/Spearman/Kendall panels
    - clone_count_sweep_all_*_heatmap.png           — questions × n_clones heatmap
    - clone_count_sweep_all_*_report.txt            — top-10 worst-case summary
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clone_pipeline.applicator import apply_specs
from clone_pipeline.spec import CloneSpec
from cross_run_analysis.analyzer import CrossRunAnalyzer
from experiments._common import _get_clean_name, _get_question_text_col, _resolve_n
from vqs.config_utils import load_config
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine

DEFAULT_N_VALUES = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
RESULTS_DIR = Path("experiment_results/clone_count_sweep")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Clone count sweep (all questions): 2D sweep over question_id × n_clones"
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
        "--n-values", type=str, default=None,
        help="Comma-separated n_clones values (default: 1,2,3,4,5,7,10,15,20,30)",
    )
    parser.add_argument(
        "-n", type=int, default=None,
        help="Override top-k for Jaccard (default: derived from config)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

    base_rankings = CrossRunAnalyzer._extract_rankings(base_recs)

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
# Per-(question, n_clones) comparison
# ---------------------------------------------------------------------------


def _compare_baseline_only(
    base_rankings: pd.DataFrame,
    cloned_recs: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """Compare two baseline-only recommendation DataFrames."""
    cloned_rankings = CrossRunAnalyzer._extract_rankings(cloned_recs)
    merged = base_rankings.join(cloned_rankings, lsuffix="_a", rsuffix="_b", how="inner")

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
            "spearman": rank.get("spearman", np.nan),
            "kendall": rank.get("kendall", np.nan),
            "n_changed": pos.get("n_changed", np.nan),
            "avg_pos_moved": pos.get("avg_pos_moved", np.nan),
        })

    return pd.DataFrame(results)


def _compute_for_question(
    q_id: int,
    config,
    pipeline: dict,
    n_values: list[int],
    n_jaccard: int,
) -> list[dict]:
    """Clone one question at each n_value, return list of metric dicts."""
    dataset = pipeline["dataset"]
    text_col = _get_question_text_col(pipeline["questions_df"])
    q_text = pipeline["questions_df"].loc[
        pipeline["questions_df"]["ID_question"] == q_id, text_col
    ].iloc[0]
    nan = pipeline["nan_info"][q_id]
    var = pipeline["var_info"][q_id]
    question_order = pipeline["question_ids"].index(q_id) + 1

    # Baseline point: no clones added, perfect agreement
    rows = [{
        "question_id": q_id,
        "question_text": q_text,
        "question_order": question_order,
        "n_clones": 0,
        "jaccard_mean": 1.0,
        "jaccard_median": 1.0,
        "jaccard_p10": 1.0,
        "spearman_mean": 1.0,
        "kendall_mean": 1.0,
        "n_changed_mean": 0.0,
        "avg_pos_moved_mean": 0.0,
        "voter_nan_pct": nan["voter_nan_pct"],
        "candidate_nan_pct": nan["candidate_nan_pct"],
        "candidate_var": var["candidate_var"],
        "voter_var": var["voter_var"],
        "combined_var": var["combined_var"],
    }]

    for n_clones in n_values:
        spec = CloneSpec(source_q_id=q_id, clone_type="identical", n_clones=n_clones)
        cloned_data = apply_specs(
            specs=[spec],
            dataframes={
                "questions": dataset["questions"],
                "voters": dataset["voters"],
                "candidates": dataset["candidates"],
            },
        )

        cloned_engine = RecommendationEngine(config=config, data_map=cloned_data)
        cloned_recs = cloned_engine.run_baseline()
        sys.stdout.flush()

        per_voter = _compare_baseline_only(pipeline["base_rankings"], cloned_recs, n_jaccard)

        rows.append({
            "question_id": q_id,
            "question_text": q_text,
            "question_order": question_order,
            "n_clones": n_clones,
            "jaccard_mean": per_voter["jaccard"].mean(),
            "jaccard_median": per_voter["jaccard"].median(),
            "jaccard_p10": per_voter["jaccard"].quantile(0.1),
            "spearman_mean": per_voter["spearman"].mean(),
            "kendall_mean": per_voter["kendall"].mean(),
            "n_changed_mean": per_voter["n_changed"].mean(),
            "avg_pos_moved_mean": per_voter["avg_pos_moved"].mean(),
            "voter_nan_pct": nan["voter_nan_pct"],
            "candidate_nan_pct": nan["candidate_nan_pct"],
            "candidate_var": var["candidate_var"],
            "voter_var": var["voter_var"],
            "combined_var": var["combined_var"],
        })

        print(
            f"    n_clones={n_clones:>2}: "
            f"Jaccard={rows[-1]['jaccard_mean']:.4f}, "
            f"Spearman={rows[-1]['spearman_mean']:.4f}, "
            f"Kendall={rows[-1]['kendall_mean']:.4f}"
        )

    return rows


# ---------------------------------------------------------------------------
# Mode: sweep (sequential)
# ---------------------------------------------------------------------------


def _run_sweep(args, config, n_values: list[int], n_jaccard: int):
    pipeline = _setup_pipeline(config)
    question_ids = pipeline["question_ids"]

    all_rows = []
    for i, q_id in enumerate(question_ids):
        print(f"\n--- Question {q_id} ({i + 1}/{len(question_ids)}) ---")
        rows = _compute_for_question(q_id, config, pipeline, n_values, n_jaccard)
        all_rows.extend(rows)

    sweep_df = pd.DataFrame(all_rows)

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_collect_outputs(sweep_df, config, n_values, n_jaccard, output_dir)

    print("\n=== Clone Count Sweep (All Questions) Complete ===")


# ---------------------------------------------------------------------------
# Mode: worker (single question for SLURM array)
# ---------------------------------------------------------------------------


def _run_worker(args, config, n_values: list[int], n_jaccard: int):
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
    out_path = sweep_dir / f"ccs_worker_{task_id:03d}_q{q_id}.csv"

    # Skip if already computed
    if out_path.exists():
        print(f"  [SKIP] Worker CSV already exists: {out_path}")
        return

    print(f"\n=== Worker: question {q_id} (task {task_id}/{len(question_ids) - 1}) ===")

    rows = _compute_for_question(q_id, config, pipeline, n_values, n_jaccard)

    worker_df = pd.DataFrame(rows)
    worker_df.to_csv(out_path, index=False)

    print(f"\n  -> Worker CSV: {out_path}")
    print(f"\n=== Worker Complete ===")


# ---------------------------------------------------------------------------
# Mode: collect (aggregate + plot)
# ---------------------------------------------------------------------------


def _run_collect(args, config, n_values: list[int], n_jaccard: int):
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "workers"

    worker_files = sorted(sweep_dir.glob("ccs_worker_*.csv"))
    if not worker_files:
        print(f"ERROR: No worker CSVs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Collect: reading {len(worker_files)} worker CSVs from {sweep_dir} ===")

    dfs = [pd.read_csv(f) for f in worker_files]
    combined = pd.concat(dfs, ignore_index=True)

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_collect_outputs(combined, config, n_values, n_jaccard, output_dir)

    print("\n=== Collect Complete ===")


# ---------------------------------------------------------------------------
# Collect output generation
# ---------------------------------------------------------------------------


def _save_collect_outputs(
    df: pd.DataFrame,
    config,
    n_values: list[int],
    n_jaccard: int,
    output_dir: Path,
):
    """Save CSV, plots, and report."""
    # Inject n_clones=0 baseline rows if not already present
    if 0 not in df["n_clones"].values:
        baseline_rows = []
        for q_id, grp in df.groupby("question_id"):
            row = grp.iloc[0].to_dict()
            row["n_clones"] = 0
            row["jaccard_mean"] = 1.0
            row["jaccard_median"] = 1.0
            row["jaccard_p10"] = 1.0
            row["spearman_mean"] = 1.0
            row["kendall_mean"] = 1.0
            row["n_changed_mean"] = 0.0
            row["avg_pos_moved_mean"] = 0.0
            baseline_rows.append(row)
        df = pd.concat([pd.DataFrame(baseline_rows), df], ignore_index=True)
        df = df.sort_values(["question_id", "n_clones"]).reset_index(drop=True)

    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")

    subfolder_name = f"clone_count_sweep_all_{name}"
    subfolder = output_dir / subfolder_name
    subfolder.mkdir(parents=True, exist_ok=True)

    base = f"clone_count_sweep_all_{name}_{timestamp}"

    # --- Save CSV ---
    csv_path = subfolder / f"{base}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> CSV: {subfolder_name}/{csv_path.name}")

    # --- Determine representative questions ---
    max_n = df["n_clones"].max()
    at_max = df[df["n_clones"] == max_n].copy()
    at_max_sorted = at_max.sort_values("jaccard_mean")

    most_impacted_qid = int(at_max_sorted.iloc[0]["question_id"])
    least_impacted_qid = int(at_max_sorted.iloc[-1]["question_id"])
    median_idx = len(at_max_sorted) // 2
    median_impacted_qid = int(at_max_sorted.iloc[median_idx]["question_id"])

    print(f"\n  Representatives (by Jaccard at n={max_n}):")
    print(f"    Most impacted:  Q{most_impacted_qid} (Jaccard={at_max_sorted.iloc[0]['jaccard_mean']:.4f})")
    print(f"    Median:         Q{median_impacted_qid} (Jaccard={at_max_sorted.iloc[median_idx]['jaccard_mean']:.4f})")
    print(f"    Least impacted: Q{least_impacted_qid} (Jaccard={at_max_sorted.iloc[-1]['jaccard_mean']:.4f})")

    # --- Plots ---
    sns.set_theme(style="whitegrid")

    _plot_metric_comparison(
        df, most_impacted_qid, median_impacted_qid, least_impacted_qid,
        config, n_jaccard, subfolder, base,
    )
    _plot_heatmap(df, config, n_jaccard, subfolder, base)
    _save_report(df, at_max_sorted, n_values, n_jaccard, config, subfolder, base)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_metric_comparison(
    df: pd.DataFrame,
    most_q: int,
    median_q: int,
    least_q: int,
    config,
    n_jaccard: int,
    output_dir: Path,
    base: str,
):
    """1×3 subplot: one panel per metric, 3 overlaid question lines + shaded spread."""
    colors = {
        "most": "#E53935",
        "median": "#1E88E5",
        "least": "#43A047",
    }

    metrics = [
        ("jaccard_mean", "Jaccard Similarity"),
        ("spearman_mean", "Spearman Correlation"),
        ("kendall_mean", "Kendall Tau"),
    ]

    representatives = [
        (most_q, "most", "Most Impacted"),
        (median_q, "median", "Median"),
        (least_q, "least", "Least Impacted"),
    ]

    # Get question texts for legend
    q_texts = {}
    for q_id, _, _ in representatives:
        q_data = df[df["question_id"] == q_id]
        text = str(q_data["question_text"].iloc[0])[:45]
        q_texts[q_id] = text

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)

    for ax, (metric_col, metric_label) in zip(axes, metrics):
        for q_id, color_key, label in representatives:
            q_data = df[df["question_id"] == q_id].sort_values("n_clones")
            x = q_data["n_clones"].values
            y = q_data[metric_col].values

            ax.plot(
                x, y, color=colors[color_key],
                marker="o", linewidth=2, markersize=5,
                label=f"{label} (Q{q_id})",
            )

        # Shade area between most and least impacted
        most_data = df[df["question_id"] == most_q].sort_values("n_clones")
        least_data = df[df["question_id"] == least_q].sort_values("n_clones")
        ax.fill_between(
            most_data["n_clones"].values,
            most_data[metric_col].values,
            least_data[metric_col].values,
            alpha=0.1, color="grey",
        )

        ax.axhline(1.0, color="grey", linestyle=":", alpha=0.3)
        ax.set_xlabel("Number of Clones Added", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(most_data["n_clones"].values)
        ax.set_title(metric_label, fontsize=13)

    axes[0].set_ylabel("Metric Value (higher = less distortion)", fontsize=11)
    axes[0].legend(loc="lower left", fontsize=9)

    fig.suptitle(
        f"Recommendation Distortion vs Number of Clones — All Questions\n"
        f"(identical clones, top-{n_jaccard}, {config.district})",
        fontsize=14,
    )
    fig.tight_layout()

    path = output_dir / f"{base}_metric_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Metric comparison: {subfolder_stem(output_dir)}/{path.name}")


def _plot_heatmap(
    df: pd.DataFrame,
    config,
    n_jaccard: int,
    output_dir: Path,
    base: str,
):
    """Heatmap: questions (y, sorted by impact) × n_clones (x), colored by Jaccard mean."""
    max_n = df["n_clones"].max()
    at_max = df[df["n_clones"] == max_n].sort_values("jaccard_mean")
    question_order = at_max["question_id"].tolist()

    # Build pivot table
    pivot = df.pivot_table(index="question_id", columns="n_clones", values="jaccard_mean")
    pivot = pivot.loc[question_order]

    # Y-axis labels
    y_labels = []
    for q_id in question_order:
        q_rows = df[df["question_id"] == q_id]
        q_text = str(q_rows["question_text"].iloc[0])[:40]
        y_labels.append(f"Q{int(q_id)}  {q_text}")

    fig, ax = plt.subplots(figsize=(14, max(10, len(question_order) * 0.28)))

    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Jaccard Similarity (mean)", "shrink": 0.6},
        xticklabels=True, yticklabels=y_labels,
        annot=False,
    )

    ax.set_xlabel("Number of Clones Added", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(
        f"Clone Count Sweep: Jaccard vs (Question, n_clones)\n"
        f"(sorted by impact at n={max_n}, top-{n_jaccard}, {config.district})",
        fontsize=13,
    )
    ax.tick_params(axis="y", labelsize=6)

    fig.tight_layout()

    path = output_dir / f"{base}_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Heatmap: {subfolder_stem(output_dir)}/{path.name}")


def subfolder_stem(p: Path) -> str:
    return p.name


def _save_report(
    df: pd.DataFrame,
    at_max_sorted: pd.DataFrame,
    n_values: list[int],
    n_jaccard: int,
    config,
    output_dir: Path,
    base: str,
    top_n: int = 10,
):
    """Human-readable report."""
    max_n = max(n_values) if n_values else df["n_clones"].max()
    min_n = min(n_values) if n_values else df["n_clones"].min()

    lines = [
        "=" * 90,
        "CLONE COUNT SWEEP — ALL QUESTIONS REPORT",
        "=" * 90,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Config: {_get_clean_name(config)}",
        f"Clone type: identical",
        f"n_clones values: {sorted(df['n_clones'].unique().tolist())}",
        f"Jaccard top-k (n): {n_jaccard}",
        f"Questions tested: {df['question_id'].nunique()}",
        "",
        f"TOP {top_n} WORST-CASE QUESTIONS (by Jaccard at n={max_n}):",
        "-" * 90,
        f"{'Rank':>4}  {'QID':>6}  {'Jac@' + str(min_n):>7}  {'Jac@' + str(max_n):>7}  "
        f"{'Spe@' + str(max_n):>7}  {'Ken@' + str(max_n):>7}  {'CombVar':>8}  Text",
        "-" * 90,
    ]

    for i, (_, row) in enumerate(at_max_sorted.head(top_n).iterrows()):
        q_id = int(row["question_id"])
        jac_at_min = df[(df["question_id"] == q_id) & (df["n_clones"] == min_n)]["jaccard_mean"].values
        jac_1 = jac_at_min[0] if len(jac_at_min) > 0 else np.nan
        text = str(row["question_text"])[:45]
        combined_var = row.get("combined_var", np.nan)
        lines.append(
            f"{i + 1:>4}  {q_id:>6}  {jac_1:>7.4f}  {row['jaccard_mean']:>7.4f}  "
            f"{row['spearman_mean']:>7.4f}  {row['kendall_mean']:>7.4f}  "
            f"{combined_var:>8.1f}  {text}"
        )

    # Overall statistics at max n_clones
    lines.extend([
        "",
        "=" * 90,
        f"STATISTICS AT n_clones={max_n}:",
        f"  Jaccard  — mean: {at_max_sorted['jaccard_mean'].mean():.4f}, "
        f"median: {at_max_sorted['jaccard_mean'].median():.4f}, "
        f"min: {at_max_sorted['jaccard_mean'].min():.4f}, "
        f"max: {at_max_sorted['jaccard_mean'].max():.4f}",
        f"  Spearman — mean: {at_max_sorted['spearman_mean'].mean():.4f}, "
        f"median: {at_max_sorted['spearman_mean'].median():.4f}",
        f"  Kendall  — mean: {at_max_sorted['kendall_mean'].mean():.4f}, "
        f"median: {at_max_sorted['kendall_mean'].median():.4f}",
        "=" * 90,
    ])

    path = output_dir / f"{base}_report.txt"
    path.write_text("\n".join(lines))
    print(f"  -> Report: {subfolder_stem(output_dir)}/{path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()
    config = load_config(Path(args.config))

    if args.n_values:
        n_values = sorted([int(v.strip()) for v in args.n_values.split(",")])
    else:
        n_values = DEFAULT_N_VALUES

    n_jaccard = _resolve_n(config, args.n)
    name = _get_clean_name(config)

    print(f"\n=== Clone Count Sweep — All Questions ({args.mode} mode) ===")
    print(f"  Config    : {name}")
    print(f"  n values  : {n_values}")
    print(f"  Top-k (n) : {n_jaccard}")

    if args.mode == "sweep":
        _run_sweep(args, config, n_values, n_jaccard)
    elif args.mode == "worker":
        _run_worker(args, config, n_values, n_jaccard)
    elif args.mode == "collect":
        _run_collect(args, config, n_values, n_jaccard)


if __name__ == "__main__":
    main()
