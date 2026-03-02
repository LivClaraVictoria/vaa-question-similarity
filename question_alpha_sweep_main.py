"""
Per-question alpha sweep: for every question in the dataset, clone it N times
with easy paraphrases (in-memory), then run a full alpha sweep comparing
base vs cloned CRW recommendations.

Produces a 2D grid: question_id × alpha, with CRW correction metrics at each point.

Supports four modes:
    - prepare: Generate all needed paraphrases (run once before SLURM workers).
    - sweep (default): Run all questions × all alphas sequentially.
    - worker:  Compute a single question (by --task-id). For SLURM job arrays.
    - collect: Read per-question worker CSVs, aggregate, and plot.

Usage:
    # Generate paraphrases (interactive, requires OPENAI_API_KEY):
    python -m question_alpha_sweep_main \\
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py \\
        --mode prepare

    # Sequential (all questions):
    python -m question_alpha_sweep_main \\
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py

    # Worker (one question, for SLURM array):
    python -m question_alpha_sweep_main --mode worker --task-id 3 \\
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py \\
        --sweep-dir /path/to/sweep_dir

    # Collect (aggregate workers + plot):
    python -m question_alpha_sweep_main --mode collect \\
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py \\
        --sweep-dir /path/to/sweep_dir

Outputs (saved to experiment_results/question_alpha_sweep_results/):
    - question_alpha_sweep_*.csv              — one row per (question, alpha)
    - question_alpha_sweep_*_heatmap.png      — questions × alpha heatmap
    - question_alpha_sweep_*_avg_curve.png    — average CRW metrics vs alpha
    - question_alpha_sweep_*_per_question.png — per-question max improvement
    - question_alpha_sweep_*_optimal_alpha_hist.png — optimal alpha distribution
    - question_alpha_sweep_*_report.txt       — summary report
"""

import argparse
import json
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

from alpha_sweep_main import (
    DEFAULT_ALPHAS,
    _compute_alpha,
    _get_clean_name,
    _get_or_compute_recs,
    _resolve_n,
    _setup_side,
)
from clone_pipeline.applicator import apply_specs
from clone_pipeline.paraphrase_generator import ensure_paraphrases
from clone_pipeline.spec import CloneSpec
from cross_run_analysis.analyzer import CrossRunAnalyzer
from main import load_config
from vqs.data_loader import load_dataset
from vqs.similarity_metrics import get_calculator

DEFAULT_N_CLONES = 5
DEFAULT_ALPHA_REFERENCE = 0.3
RESULTS_DIR = Path("experiment_results/question_alpha_sweep_results")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Per-question alpha sweep with easy paraphrase clones"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Base pipeline config (e.g. configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["prepare", "sweep", "worker", "collect"],
        default="sweep",
        help="Execution mode",
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="Question index for worker mode (typically SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--sweep-dir", type=str, default=None,
        help="Directory for per-question worker CSVs",
    )
    parser.add_argument(
        "--alphas", type=str, default=None,
        help="Comma-separated alpha values (default: standard 21-value range)",
    )
    parser.add_argument(
        "--n-clones", type=int, default=DEFAULT_N_CLONES,
        help=f"Number of easy_paraphrase clones per question (default: {DEFAULT_N_CLONES})",
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


def _load_paraphrases_readonly(config) -> dict:
    """Load paraphrase cache (read-only). Fails if cache doesn't exist."""
    cache_path = config.PARAPHRASES_DIR / f"paraphrases_{config.data_year}.json"
    if not cache_path.exists():
        print(
            f"ERROR: Paraphrase cache not found at {cache_path}.\n"
            f"Run with --mode prepare first to generate paraphrases.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(cache_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _setup_pipeline(config, n_clones: int):
    """Load dataset, compute base side, load paraphrases, get question IDs."""
    print("\n--- Setting up base pipeline ---")
    base_side = _setup_side(config)

    questions_df = base_side["dataset"]["questions"]
    question_ids = sorted(
        questions_df.loc[
            questions_df["ID_question"] < 9_000_000, "ID_question"
        ].tolist()
    )

    print(f"  Questions: {len(question_ids)}")

    # Load paraphrases (read-only)
    paraphrases = _load_paraphrases_readonly(config)

    # Verify all questions have enough paraphrases
    missing = []
    for q_id in question_ids:
        q_id_str = str(q_id)
        existing = paraphrases.get(q_id_str, {}).get("easy_paraphrase", [])
        if len(existing) < n_clones:
            missing.append((q_id, len(existing)))
    if missing:
        print(
            f"ERROR: {len(missing)} questions lack enough easy_paraphrases "
            f"(need {n_clones} each).\n"
            f"Run with --mode prepare first.\n"
            f"Examples: {missing[:5]}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Pre-compute question metadata
    text_col = _get_question_text_col(questions_df)
    question_texts = {}
    for q_id in question_ids:
        row = questions_df.loc[questions_df["ID_question"] == q_id, text_col]
        question_texts[q_id] = row.iloc[0] if not row.empty else ""

    return {
        "base_side": base_side,
        "question_ids": question_ids,
        "question_texts": question_texts,
        "paraphrases": paraphrases,
    }


# ---------------------------------------------------------------------------
# Per-question alpha sweep
# ---------------------------------------------------------------------------


def _compute_question_sweep(
    q_id: int,
    config,
    pipeline: dict,
    alphas: list[float],
    n_clones: int,
    n_jaccard: int,
) -> list[dict]:
    """Run full alpha sweep for one question cloned n_clones times."""
    base_side = pipeline["base_side"]
    dataset = base_side["dataset"]
    paraphrases = pipeline["paraphrases"]
    q_text = pipeline["question_texts"][q_id]

    # Clone in-memory
    spec = CloneSpec(
        source_q_id=q_id, clone_type="easy_paraphrase", n_clones=n_clones
    )
    cloned_data = apply_specs(
        specs=[spec],
        dataframes={
            "questions": dataset["questions"],
            "voters": dataset["voters"],
            "candidates": dataset["candidates"],
        },
        paraphrases=paraphrases,
    )

    # Shallow-copy config with unique clone_id for cache safety
    # (deepcopy fails because config contains module references)
    cloned_config = SimpleNamespace(**vars(config))
    cloned_config.clone_id = f"qa_sweep_ep{n_clones}_q{q_id}"

    calculator = get_calculator(cloned_config)
    cloned_dist = calculator.calculate_distance(cloned_data, cloned_config)

    # Set up cloned side (baseline recs + rec engine)
    cloned_side = _setup_side(
        cloned_config, dataset=cloned_data, dist_df=cloned_dist
    )

    # Build pipeline dict compatible with _compute_alpha
    sweep_pipeline = {
        "dist_df_a": base_side["dist_df"],
        "dist_df_b": cloned_side["dist_df"],
        "rec_engine_a": base_side["rec_engine"],
        "rec_engine_b": cloned_side["rec_engine"],
        "baseline_a": base_side["baseline"],
        "baseline_b": cloned_side["baseline"],
    }

    analyzer = CrossRunAnalyzer.from_n(n_jaccard)
    rows = []

    for i, alpha in enumerate(alphas):
        row, std_metrics = _compute_alpha(
            alpha, config, cloned_config, sweep_pipeline, analyzer, n_jaccard
        )

        rows.append({
            "question_id": q_id,
            "question_text": q_text,
            "alpha": alpha,
            "base_jaccard_mean": std_metrics["base_jaccard_mean"],
            "base_spearman_mean": std_metrics["base_spearman_mean"],
            "base_kendall_mean": std_metrics["base_kendall_mean"],
            **{k: v for k, v in row.items() if k != "alpha"},
        })

        print(
            f"    alpha={alpha:.2f}: "
            f"CRW Jaccard={row['crw_jaccard_mean']:.4f}, "
            f"Spearman={row['crw_spearman_mean']:.4f}, "
            f"Kendall={row['crw_kendall_mean']:.4f}"
        )

    return rows


# ---------------------------------------------------------------------------
# Mode: prepare (generate paraphrases)
# ---------------------------------------------------------------------------


def _run_prepare(config, n_clones: int):
    """Generate all needed easy_paraphrase paraphrases for every question."""
    print("\n--- Loading dataset for paraphrase generation ---")
    dataset = load_dataset(config)
    questions_df = dataset["questions"]

    question_ids = sorted(
        questions_df.loc[
            questions_df["ID_question"] < 9_000_000, "ID_question"
        ].tolist()
    )

    print(f"  Questions: {len(question_ids)}")
    print(f"  Clones per question: {n_clones} (easy_paraphrase)")
    print(f"  Total paraphrases needed: {len(question_ids) * n_clones}")

    specs = [
        CloneSpec(source_q_id=q_id, clone_type="easy_paraphrase", n_clones=n_clones)
        for q_id in question_ids
    ]

    paraphrases = ensure_paraphrases(
        specs=specs,
        questions_df=questions_df,
        data_year=config.data_year,
        paraphrase_dir=config.PARAPHRASES_DIR,
    )

    # Verify
    ready = sum(
        1 for q_id in question_ids
        if len(paraphrases.get(str(q_id), {}).get("easy_paraphrase", [])) >= n_clones
    )
    print(f"\n  Ready: {ready}/{len(question_ids)} questions have >= {n_clones} easy paraphrases")

    if ready < len(question_ids):
        print("WARNING: Some questions still lack paraphrases!", file=sys.stderr)
    else:
        print("All paraphrases ready. You can now run sweep/worker mode.")

    print("\n=== Prepare Complete ===")


# ---------------------------------------------------------------------------
# Mode: sweep (sequential)
# ---------------------------------------------------------------------------


def _run_sweep(args, config, alphas: list[float], n_clones: int, n_jaccard: int):
    pipeline = _setup_pipeline(config, n_clones)
    question_ids = pipeline["question_ids"]

    all_rows = []
    for i, q_id in enumerate(question_ids):
        print(f"\n--- Question {q_id} ({i + 1}/{len(question_ids)}) ---")
        rows = _compute_question_sweep(
            q_id, config, pipeline, alphas, n_clones, n_jaccard
        )
        all_rows.extend(rows)

    sweep_df = pd.DataFrame(all_rows)

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_collect_outputs(sweep_df, config, alphas, n_clones, n_jaccard, output_dir)

    print("\n=== Question Alpha Sweep Complete ===")


# ---------------------------------------------------------------------------
# Mode: worker (single question for SLURM array)
# ---------------------------------------------------------------------------


def _run_worker(args, config, alphas: list[float], n_clones: int, n_jaccard: int):
    task_id = args.task_id
    if task_id is None:
        print("ERROR: --task-id is required in worker mode", file=sys.stderr)
        sys.exit(1)

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "workers"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _setup_pipeline(config, n_clones)
    question_ids = pipeline["question_ids"]

    if task_id < 0 or task_id >= len(question_ids):
        print(
            f"ERROR: --task-id {task_id} out of range [0, {len(question_ids) - 1}]",
            file=sys.stderr,
        )
        sys.exit(1)

    q_id = question_ids[task_id]
    out_path = sweep_dir / f"qa_sweep_worker_{task_id:03d}_q{q_id}.csv"

    # Skip if already computed
    if out_path.exists():
        print(f"  [SKIP] Worker CSV already exists: {out_path}")
        return

    print(f"\n=== Worker: question {q_id} (task {task_id}/{len(question_ids) - 1}) ===")

    rows = _compute_question_sweep(
        q_id, config, pipeline, alphas, n_clones, n_jaccard
    )

    worker_df = pd.DataFrame(rows)
    worker_df.to_csv(out_path, index=False)

    print(f"\n  -> Worker CSV: {out_path}")
    print(f"\n=== Worker Complete ===")


# ---------------------------------------------------------------------------
# Mode: collect (aggregate + plot)
# ---------------------------------------------------------------------------


def _run_collect(args, config, alphas: list[float], n_clones: int, n_jaccard: int):
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "workers"

    worker_files = sorted(sweep_dir.glob("qa_sweep_worker_*.csv"))
    if not worker_files:
        print(f"ERROR: No worker CSVs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Collect: reading {len(worker_files)} worker CSVs from {sweep_dir} ===")

    dfs = [pd.read_csv(f) for f in worker_files]
    combined = pd.concat(dfs, ignore_index=True)

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_collect_outputs(combined, config, alphas, n_clones, n_jaccard, output_dir)

    print("\n=== Collect Complete ===")


# ---------------------------------------------------------------------------
# Collect output generation
# ---------------------------------------------------------------------------


def _save_collect_outputs(
    df: pd.DataFrame,
    config,
    alphas: list[float],
    n_clones: int,
    n_jaccard: int,
    output_dir: Path,
):
    """Save CSV, plots, and report."""
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")

    subfolder_name = f"question_alpha_sweep_{name}_ep{n_clones}"
    subfolder = output_dir / subfolder_name
    subfolder.mkdir(parents=True, exist_ok=True)

    base = f"question_alpha_sweep_{name}_ep{n_clones}_{timestamp}"

    # --- CSV ---
    csv_path = subfolder / f"{base}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> CSV: {subfolder_name}/{csv_path.name}")

    # --- Compute per-question summary ---
    summary = _compute_per_question_summary(df)

    sns.set_theme(style="whitegrid")

    _plot_heatmap(df, config, n_clones, n_jaccard, subfolder, base)
    _plot_avg_curve(df, config, n_clones, n_jaccard, subfolder, base)
    _plot_per_question(summary, config, n_clones, n_jaccard, subfolder, base)
    _plot_optimal_alpha_hist(summary, config, n_clones, subfolder, base)
    _save_report(df, summary, config, n_clones, n_jaccard, subfolder, base)


def _compute_per_question_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-question aggregation: optimal alpha, max CRW Jaccard, baseline distortion."""
    rows = []
    for q_id, grp in df.groupby("question_id"):
        # Baseline distortion (alpha-independent — take from first row)
        base_jac = grp["base_jaccard_mean"].iloc[0]

        # Best CRW correction
        best_idx = grp["crw_jaccard_mean"].idxmax()
        best_row = grp.loc[best_idx]

        rows.append({
            "question_id": q_id,
            "question_text": grp["question_text"].iloc[0],
            "base_jaccard_mean": base_jac,
            "base_spearman_mean": grp["base_spearman_mean"].iloc[0],
            "base_kendall_mean": grp["base_kendall_mean"].iloc[0],
            "optimal_alpha": best_row["alpha"],
            "max_crw_jaccard": best_row["crw_jaccard_mean"],
            "max_crw_spearman": best_row["crw_spearman_mean"],
            "max_crw_kendall": best_row["crw_kendall_mean"],
            "improvement": best_row["crw_jaccard_mean"] - base_jac,
        })

    summary = pd.DataFrame(rows).sort_values("improvement", ascending=False)
    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_heatmap(
    df: pd.DataFrame,
    config,
    n_clones: int,
    n_jaccard: int,
    output_dir: Path,
    base: str,
):
    """Heatmap: questions (y, sorted by max CRW Jaccard) × alpha (x)."""
    # Sort questions by max CRW Jaccard (best correction at top)
    q_order = (
        df.groupby("question_id")["crw_jaccard_mean"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )

    pivot = df.pivot_table(
        index="question_id", columns="alpha", values="crw_jaccard_mean"
    )
    pivot = pivot.loc[q_order]

    # Y-axis labels
    y_labels = []
    for q_id in q_order:
        q_rows = df[df["question_id"] == q_id]
        q_text = str(q_rows["question_text"].iloc[0])[:35]
        y_labels.append(f"Q{int(q_id)}  {q_text}")

    fig, ax = plt.subplots(figsize=(16, max(10, len(q_order) * 0.28)))

    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn",
        vmin=pivot.values.min() * 0.95,
        vmax=min(1.0, pivot.values.max() * 1.02),
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "CRW Jaccard (mean)", "shrink": 0.6},
        xticklabels=True, yticklabels=y_labels,
        annot=False,
    )

    # Mark reference alpha
    alpha_cols = sorted(pivot.columns.tolist())
    if DEFAULT_ALPHA_REFERENCE in alpha_cols:
        ref_idx = alpha_cols.index(DEFAULT_ALPHA_REFERENCE)
        ax.axvline(ref_idx + 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.7)

    ax.set_xlabel("Alpha (α)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(
        f"Per-Question CRW Correction: Jaccard vs (Question, Alpha)\n"
        f"({n_clones}× easy_paraphrase, top-{n_jaccard}, {config.district})",
        fontsize=13,
    )
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="x", labelsize=8, rotation=45)

    fig.tight_layout()

    path = output_dir / f"{base}_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Heatmap: {output_dir.name}/{path.name}")


def _plot_avg_curve(
    df: pd.DataFrame,
    config,
    n_clones: int,
    n_jaccard: int,
    output_dir: Path,
    base: str,
):
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

    fig, ax = plt.subplots(figsize=(10, 6))

    # CRW curves
    ax.plot(
        alpha_agg["alpha"], alpha_agg["crw_jaccard_mean"],
        color=colors["jaccard"], linewidth=2, label="CRW Jaccard (mean across questions)",
    )
    ax.fill_between(
        alpha_agg["alpha"],
        alpha_agg["crw_jaccard_q25"],
        alpha_agg["crw_jaccard_q75"],
        color=colors["jaccard"], alpha=0.15, label="Jaccard 25th–75th percentile",
    )
    ax.plot(
        alpha_agg["alpha"], alpha_agg["crw_spearman_mean"],
        color=colors["spearman"], linewidth=2, label="CRW Spearman (mean)",
    )
    ax.plot(
        alpha_agg["alpha"], alpha_agg["crw_kendall_mean"],
        color=colors["kendall"], linewidth=2, label="CRW Kendall (mean)",
    )

    # Baseline dashed lines
    ax.axhline(
        alpha_agg["base_jaccard_mean"].iloc[0],
        color=colors["jaccard"], linestyle="--", alpha=0.45, label="Jaccard – no CRW",
    )
    ax.axhline(
        alpha_agg["base_spearman_mean"].iloc[0],
        color=colors["spearman"], linestyle="--", alpha=0.45, label="Spearman – no CRW",
    )
    ax.axhline(
        alpha_agg["base_kendall_mean"].iloc[0],
        color=colors["kendall"], linestyle="--", alpha=0.45, label="Kendall – no CRW",
    )

    # Reference alpha
    ax.axvline(
        DEFAULT_ALPHA_REFERENCE, color="grey", linestyle="--", alpha=0.5,
        label=f"Reference α={DEFAULT_ALPHA_REFERENCE}",
    )

    ax.set_xlabel("Alpha (α)", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Average CRW Correction vs Alpha (across all questions)\n"
        f"({n_clones}× easy_paraphrase, top-{n_jaccard}, {config.district})",
        fontsize=13,
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = output_dir / f"{base}_avg_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Avg curve: {output_dir.name}/{path.name}")


def _plot_per_question(
    summary: pd.DataFrame,
    config,
    n_clones: int,
    n_jaccard: int,
    output_dir: Path,
    base: str,
    top_n: int = 30,
):
    """Bar chart: max CRW Jaccard improvement per question."""
    plot_df = summary.head(top_n).iloc[::-1]  # reverse for horizontal bars

    fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.35)))

    labels = [
        f"Q{int(row['question_id'])}  {str(row['question_text'])[:40]}"
        for _, row in plot_df.iterrows()
    ]

    bars = ax.barh(
        range(len(plot_df)),
        plot_df["improvement"],
        color="#2196F3", alpha=0.8,
    )

    # Annotate with optimal alpha
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            row["improvement"] + 0.002,
            i,
            f"α={row['optimal_alpha']:.1f}",
            va="center", fontsize=7, color="#666",
        )

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("CRW Jaccard Improvement over Baseline (pp)", fontsize=11)
    ax.set_title(
        f"Per-Question CRW Correction (Top {top_n})\n"
        f"({n_clones}× easy_paraphrase, top-{n_jaccard}, {config.district})",
        fontsize=13,
    )
    fig.tight_layout()

    path = output_dir / f"{base}_per_question.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Per-question: {output_dir.name}/{path.name}")


def _plot_optimal_alpha_hist(
    summary: pd.DataFrame,
    config,
    n_clones: int,
    output_dir: Path,
    base: str,
):
    """Histogram of per-question optimal alphas."""
    fig, ax = plt.subplots(figsize=(8, 5))

    alphas = summary["optimal_alpha"]
    bins = np.arange(0, alphas.max() + 0.15, 0.1)

    ax.hist(alphas, bins=bins, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(
        DEFAULT_ALPHA_REFERENCE, color="red", linestyle="--", linewidth=1.5,
        label=f"Reference α={DEFAULT_ALPHA_REFERENCE}",
    )
    ax.axvline(
        alphas.median(), color="orange", linestyle="--", linewidth=1.5,
        label=f"Median optimal α={alphas.median():.2f}",
    )

    ax.set_xlabel("Optimal Alpha", fontsize=11)
    ax.set_ylabel("Number of Questions", fontsize=11)
    ax.set_title(
        f"Distribution of Per-Question Optimal Alpha\n"
        f"({n_clones}× easy_paraphrase, {config.district})",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = output_dir / f"{base}_optimal_alpha_hist.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Optimal alpha hist: {output_dir.name}/{path.name}")


def _save_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    config,
    n_clones: int,
    n_jaccard: int,
    output_dir: Path,
    base: str,
    top_n: int = 10,
):
    """Human-readable report."""
    lines = [
        "=" * 90,
        "PER-QUESTION ALPHA SWEEP REPORT",
        "=" * 90,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Config: {_get_clean_name(config)}",
        f"Clone type: easy_paraphrase × {n_clones}",
        f"Alpha range: {sorted(df['alpha'].unique().tolist())}",
        f"Jaccard top-k (n): {n_jaccard}",
        f"Questions tested: {df['question_id'].nunique()}",
        "",
        f"TOP {top_n} MOST CORRECTABLE QUESTIONS (by CRW Jaccard improvement):",
        "-" * 90,
        f"{'Rank':>4}  {'QID':>6}  {'BaseJac':>8}  {'BestCRW':>8}  {'Improv':>7}  "
        f"{'OptAlpha':>8}  Text",
        "-" * 90,
    ]

    for i, (_, row) in enumerate(summary.head(top_n).iterrows()):
        text = str(row["question_text"])[:45]
        lines.append(
            f"{i + 1:>4}  {int(row['question_id']):>6}  "
            f"{row['base_jaccard_mean']:>8.4f}  {row['max_crw_jaccard']:>8.4f}  "
            f"{row['improvement']:>+7.4f}  {row['optimal_alpha']:>8.2f}  {text}"
        )

    lines.extend([
        "",
        f"BOTTOM {top_n} LEAST CORRECTABLE QUESTIONS:",
        "-" * 90,
    ])

    for i, (_, row) in enumerate(summary.tail(top_n).iloc[::-1].iterrows()):
        text = str(row["question_text"])[:45]
        lines.append(
            f"{i + 1:>4}  {int(row['question_id']):>6}  "
            f"{row['base_jaccard_mean']:>8.4f}  {row['max_crw_jaccard']:>8.4f}  "
            f"{row['improvement']:>+7.4f}  {row['optimal_alpha']:>8.2f}  {text}"
        )

    lines.extend([
        "",
        "=" * 90,
        "OVERALL STATISTICS:",
        f"  Baseline Jaccard — mean: {summary['base_jaccard_mean'].mean():.4f}, "
        f"median: {summary['base_jaccard_mean'].median():.4f}",
        f"  Best CRW Jaccard — mean: {summary['max_crw_jaccard'].mean():.4f}, "
        f"median: {summary['max_crw_jaccard'].median():.4f}",
        f"  Improvement — mean: {summary['improvement'].mean():.4f}, "
        f"median: {summary['improvement'].median():.4f}, "
        f"min: {summary['improvement'].min():.4f}, "
        f"max: {summary['improvement'].max():.4f}",
        f"  Optimal alpha — mean: {summary['optimal_alpha'].mean():.2f}, "
        f"median: {summary['optimal_alpha'].median():.2f}, "
        f"mode: {summary['optimal_alpha'].mode().iloc[0]:.2f}",
        f"  Questions with improvement > 0: "
        f"{(summary['improvement'] > 0).sum()}/{len(summary)}",
        "=" * 90,
    ])

    path = output_dir / f"{base}_report.txt"
    path.write_text("\n".join(lines))
    print(f"  -> Report: {output_dir.name}/{path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()
    config = load_config(Path(args.config))

    if args.alphas:
        alphas = sorted([float(a.strip()) for a in args.alphas.split(",")])
    else:
        alphas = DEFAULT_ALPHAS

    n_clones = args.n_clones
    n_jaccard = _resolve_n(config, args.n)
    name = _get_clean_name(config)

    print(f"\n=== Question Alpha Sweep ({args.mode} mode) ===")
    print(f"  Config    : {name}")
    print(f"  Clones    : {n_clones} × easy_paraphrase")
    print(f"  Alphas    : {len(alphas)} values")
    print(f"  Top-k (n) : {n_jaccard}")

    if args.mode == "prepare":
        _run_prepare(config, n_clones)
    elif args.mode == "sweep":
        _run_sweep(args, config, alphas, n_clones, n_jaccard)
    elif args.mode == "worker":
        _run_worker(args, config, alphas, n_clones, n_jaccard)
    elif args.mode == "collect":
        _run_collect(args, config, alphas, n_clones, n_jaccard)


if __name__ == "__main__":
    main()
