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
        --config configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py \\
        --mode prepare

    # Sequential (all questions):
    python -m question_alpha_sweep_main \\
        --config configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py

    # Worker (one question, for SLURM array):
    python -m question_alpha_sweep_main --mode worker --task-id 3 \\
        --config configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py \\
        --sweep-dir /path/to/sweep_dir

    # Collect (aggregate workers + plot):
    python -m question_alpha_sweep_main --mode collect \\
        --config configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py \\
        --sweep-dir /path/to/sweep_dir

Outputs (saved to experiment_results/exp1/question_alpha_sweep/):
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

from experiments._common import (
    _get_clean_name,
    _resolve_n,
    _get_question_text_col,
    DEFAULT_ALPHAS,
    FLIP_TYPES,
    PERFECT_MIX_COMPONENTS,
)
from experiments.perfect_clones.model_selection import _setup_side
from clone_pipeline.applicator import apply_specs
from clone_pipeline.paraphrase_generator import ensure_paraphrases
from clone_pipeline.spec import CloneSpec
from cross_run_analysis.analyzer import CrossRunAnalyzer
from vqs.config_utils import load_config
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine
from vqs.similarity_metrics import get_calculator

DEFAULT_N_CLONES = 5
DEFAULT_ALPHA_REFERENCE = 0.4
RESULTS_DIR = Path("experiment_results/exp1/question_alpha_sweep")
ALL_CLONE_TYPES = ["easy_paraphrase", "hard_paraphrase", "negation_easy", "negation_hard", "perfect_mix"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Per-question alpha sweep with easy paraphrase clones"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Base pipeline config (e.g. configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py)",
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
        "--clone-type", type=str, default="easy_paraphrase",
        help="Clone type to use (default: easy_paraphrase). "
             "Use 'all' for prepare mode to generate paraphrases for all types.",
    )
    parser.add_argument(
        "--n-clones", type=int, default=DEFAULT_N_CLONES,
        help=f"Number of clones per question (default: {DEFAULT_N_CLONES})",
    )
    parser.add_argument(
        "-n", type=int, default=None,
        help="Override top-k for Jaccard (default: derived from config)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _setup_pipeline(config, n_clones: int, clone_type: str = "easy_paraphrase"):
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
    print(f"  Clone type: {clone_type}")

    # Validate perfect_mix clone count
    if clone_type == "perfect_mix":
        n_components = len(PERFECT_MIX_COMPONENTS)
        if n_clones % n_components != 0:
            raise ValueError(
                f"n_clones={n_clones} is not divisible by {n_components} "
                f"(PERFECT_MIX_COMPONENTS={PERFECT_MIX_COMPONENTS}). "
                f"For perfect_mix, n_clones must be a multiple of {n_components} "
                f"so each component type gets an equal number of clones."
            )

    # Load paraphrases (read-only) — needed for all non-identical types
    paraphrases = None
    if clone_type != "identical":
        paraphrases = _load_paraphrases_readonly(config)

        # Determine which paraphrase types we need
        if clone_type == "perfect_mix":
            needed_types = PERFECT_MIX_COMPONENTS
        else:
            needed_types = [clone_type]

        # Verify all questions have enough paraphrases
        missing = []
        for q_id in question_ids:
            q_id_str = str(q_id)
            for pt in needed_types:
                existing = paraphrases.get(q_id_str, {}).get(pt, [])
                n_needed = n_clones // len(PERFECT_MIX_COMPONENTS) if clone_type == "perfect_mix" else n_clones
                if len(existing) < n_needed:
                    missing.append((q_id, pt, len(existing), n_needed))
        if missing:
            print(
                f"ERROR: {len(missing)} question/type combos lack enough paraphrases "
                f"(need {missing[0][3]} each).\n"
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
    clone_type: str = "easy_paraphrase",
) -> list[dict]:
    """Run full alpha sweep for one question cloned n_clones times.

    All computations are done in-memory — no recommendation or analysis
    cache files are written to disk.  This avoids the combinatorial cache
    explosion that previously generated ~787 MB per (question, alpha, clone_type).
    """
    base_side = pipeline["base_side"]
    dataset = base_side["dataset"]
    paraphrases = pipeline["paraphrases"]
    q_text = pipeline["question_texts"][q_id]

    # Build clone specs
    if clone_type == "perfect_mix":
        specs = [
            CloneSpec(
                source_q_id=q_id, clone_type=ct,
                n_clones=n_clones // len(PERFECT_MIX_COMPONENTS),
                flip_answers=(ct in FLIP_TYPES),
            )
            for ct in PERFECT_MIX_COMPONENTS
        ]
    else:
        flip = clone_type in FLIP_TYPES
        specs = [CloneSpec(
            source_q_id=q_id, clone_type=clone_type,
            n_clones=n_clones, flip_answers=flip,
        )]

    cloned_data = apply_specs(
        specs=specs,
        dataframes={
            "questions": dataset["questions"],
            "voters": dataset["voters"],
            "candidates": dataset["candidates"],
        },
        paraphrases=paraphrases,
    )

    # Compute distances for cloned data (in-memory, no cache)
    cloned_config = SimpleNamespace(**vars(config))
    cloned_config.clone_id = f"qa_sweep_{clone_type}_n{n_clones}_q{q_id}"

    calculator = get_calculator(cloned_config)
    cloned_dist = calculator.calculate_distance(cloned_data, cloned_config)

    # Build cloned-side rec engine and baseline (once per question, reused across alphas)
    cloned_rec_engine = RecommendationEngine(config=cloned_config, data_map=cloned_data)
    cloned_baseline = cloned_rec_engine.run_baseline()

    # Base side: combine baseline with CRW for each alpha (in-memory)
    base_rec_engine = base_side["rec_engine"]
    base_baseline = base_side["baseline"]
    base_dist = base_side["dist_df"]

    analyzer = CrossRunAnalyzer.from_n(n_jaccard)
    rows = []

    for i, alpha in enumerate(alphas):
        # --- Base side: compute CRW recs for this alpha ---
        config.alpha = alpha
        base_reweighter = CloneRobustReweighter(config)
        base_weights = base_reweighter.reweight(base_dist)
        base_crw = base_rec_engine.run_crw(base_weights)

        base_match_cols = [c for c in base_crw.columns if "match" in c or "Dist" in c]
        base_combined = base_baseline.join(base_crw[base_match_cols].add_prefix("CRW_"))

        # --- Cloned side: compute CRW recs for this alpha ---
        cloned_config.alpha = alpha
        cloned_reweighter = CloneRobustReweighter(cloned_config)
        cloned_weights = cloned_reweighter.reweight(cloned_dist)
        cloned_crw = cloned_rec_engine.run_crw(cloned_weights)

        cloned_match_cols = [c for c in cloned_crw.columns if "match" in c or "Dist" in c]
        cloned_combined = cloned_baseline.join(cloned_crw[cloned_match_cols].add_prefix("CRW_"))

        # --- Cross-run analysis (in-memory, no cache) ---
        results = analyzer.analyze_from_dfs(base_combined, cloned_combined)

        crw_jac = results["crw_jaccard"]
        row = {
            "alpha": alpha,
            "crw_jaccard_mean": crw_jac.mean(),
            "crw_jaccard_median": crw_jac.median(),
            "crw_jaccard_p10": crw_jac.quantile(0.1),
            "crw_spearman_mean": results["crw_spearman"].mean(),
            "crw_kendall_mean": results["crw_kendall"].mean(),
        }
        std_metrics = {
            "base_jaccard_mean": results["base_jaccard"].mean(),
            "base_spearman_mean": results["base_spearman"].mean(),
            "base_kendall_mean": results["base_kendall"].mean(),
        }

        rows.append({
            "question_id": q_id,
            "question_text": q_text,
            "clone_type": clone_type,
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


def _run_prepare(config, n_clones: int, clone_type: str = "easy_paraphrase"):
    """Generate all needed paraphrases for every question."""
    print("\n--- Loading dataset for paraphrase generation ---")
    dataset = load_dataset(config)
    questions_df = dataset["questions"]

    question_ids = sorted(
        questions_df.loc[
            questions_df["ID_question"] < 9_000_000, "ID_question"
        ].tolist()
    )

    # Determine which paraphrase types to generate
    if clone_type == "all":
        prep_types = list(set(
            ct for ct in ALL_CLONE_TYPES if ct != "identical" and ct != "perfect_mix"
        ) | set(PERFECT_MIX_COMPONENTS))
    elif clone_type == "perfect_mix":
        prep_types = PERFECT_MIX_COMPONENTS
    elif clone_type == "identical":
        print("  No paraphrases needed for identical clones.")
        print("\n=== Prepare Complete ===")
        return
    else:
        prep_types = [clone_type]

    print(f"  Questions: {len(question_ids)}")
    print(f"  Paraphrase types: {prep_types}")
    print(f"  Clones per question per type: {n_clones}")

    specs = [
        CloneSpec(source_q_id=q_id, clone_type=ct, n_clones=n_clones)
        for q_id in question_ids
        for ct in prep_types
    ]

    paraphrases = ensure_paraphrases(
        specs=specs,
        questions_df=questions_df,
        data_year=config.data_year,
        paraphrase_dir=config.PARAPHRASES_DIR,
    )

    # Verify
    ready = 0
    total = len(question_ids) * len(prep_types)
    for q_id in question_ids:
        for ct in prep_types:
            existing = paraphrases.get(str(q_id), {}).get(ct, [])
            if len(existing) >= n_clones:
                ready += 1
    print(f"\n  Ready: {ready}/{total} question/type combos have >= {n_clones} paraphrases")

    if ready < total:
        print("WARNING: Some questions still lack paraphrases!", file=sys.stderr)
    else:
        print("All paraphrases ready. You can now run sweep/worker mode.")

    print("\n=== Prepare Complete ===")


# ---------------------------------------------------------------------------
# Mode: sweep (sequential)
# ---------------------------------------------------------------------------


def _run_sweep(args, config, alphas: list[float], n_clones: int, n_jaccard: int,
               clone_type: str = "easy_paraphrase"):
    pipeline = _setup_pipeline(config, n_clones, clone_type=clone_type)
    question_ids = pipeline["question_ids"]

    all_rows = []
    for i, q_id in enumerate(question_ids):
        print(f"\n--- Question {q_id} ({i + 1}/{len(question_ids)}) ---")
        rows = _compute_question_sweep(
            q_id, config, pipeline, alphas, n_clones, n_jaccard,
            clone_type=clone_type,
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


def _run_worker(args, config, alphas: list[float], n_clones: int, n_jaccard: int,
                clone_type: str = "easy_paraphrase"):
    task_id = args.task_id
    if task_id is None:
        print("ERROR: --task-id is required in worker mode", file=sys.stderr)
        sys.exit(1)

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "workers"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _setup_pipeline(config, n_clones, clone_type=clone_type)
    question_ids = pipeline["question_ids"]

    if task_id < 0 or task_id >= len(question_ids):
        print(
            f"ERROR: --task-id {task_id} out of range [0, {len(question_ids) - 1}]",
            file=sys.stderr,
        )
        sys.exit(1)

    q_id = question_ids[task_id]
    out_path = sweep_dir / f"qa_sweep_worker_{task_id:03d}_q{q_id}_{clone_type}.csv"

    # Skip if already computed
    if out_path.exists():
        print(f"  [SKIP] Worker CSV already exists: {out_path}")
        return

    print(f"\n=== Worker: question {q_id} (task {task_id}/{len(question_ids) - 1}), clone_type={clone_type} ===")

    rows = _compute_question_sweep(
        q_id, config, pipeline, alphas, n_clones, n_jaccard,
        clone_type=clone_type,
    )

    worker_df = pd.DataFrame(rows)
    worker_df.to_csv(out_path, index=False)

    print(f"\n  -> Worker CSV: {out_path}")
    print(f"\n=== Worker Complete ===")


# ---------------------------------------------------------------------------
# Mode: collect (aggregate + plot)
# ---------------------------------------------------------------------------


def _get_base_min_distance(config) -> float | None:
    """Load the base (non-cloned) distance matrix and return the minimum distance.

    This represents the tightest pair of original questions — the threshold
    above which CRW starts treating naturally similar questions as clones.
    Returns None if the distance cache is not available.
    """
    try:
        calculator = get_calculator(config)
        dataset = load_dataset(config)
        dist_df = calculator.calculate_distance(dataset, config)
        # Filter to original questions only (no clones)
        orig = dist_df[(dist_df["ID1"] < 9_000_000) & (dist_df["ID2"] < 9_000_000)]
        min_dist = orig["Distance"].min()
        print(f"  Base min non-clone distance: {min_dist:.4f}")
        return float(min_dist)
    except Exception as e:
        print(f"  Warning: could not load base distances for min-distance line: {e}")
        return None


def _run_collect(args, config, alphas: list[float], n_clones: int, n_jaccard: int):
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "workers"

    worker_files = sorted(sweep_dir.glob("qa_sweep_worker_*.csv"))
    if not worker_files:
        print(f"ERROR: No worker CSVs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Collect: reading {len(worker_files)} worker CSVs from {sweep_dir} ===")
    print(f"  (covers multiple clone types if present in worker CSVs)")

    dfs = [pd.read_csv(f) for f in worker_files]
    combined = pd.concat(dfs, ignore_index=True)

    # Try to compute min non-clone distance for annotation
    min_dist = _get_base_min_distance(config)

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_collect_outputs(combined, config, alphas, n_clones, n_jaccard, output_dir,
                          min_nonclone_dist=min_dist)

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
    min_nonclone_dist: float | None = None,
):
    """Save CSV, plots, and report."""
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # Determine clone types present in data
    has_clone_types = "clone_type" in df.columns
    if has_clone_types:
        clone_types_in_data = sorted(df["clone_type"].unique())
        ct_suffix = f"_{len(clone_types_in_data)}ct" if len(clone_types_in_data) > 1 else f"_{clone_types_in_data[0]}"
    else:
        ct_suffix = "_ep"

    subfolder_name = f"question_alpha_sweep_{name}{ct_suffix}_n{n_clones}"
    subfolder = output_dir / subfolder_name
    subfolder.mkdir(parents=True, exist_ok=True)

    base = f"question_alpha_sweep_{name}{ct_suffix}_n{n_clones}_{timestamp}"

    # --- CSV ---
    # Add min non-clone distance as a column (constant across rows, useful for thesis figures)
    if min_nonclone_dist is not None:
        df = df.copy()
        df["min_nonclone_dist"] = min_nonclone_dist
    csv_path = subfolder / f"{base}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> CSV: {subfolder_name}/{csv_path.name}")

    # --- Compute per-question summary ---
    summary = _compute_per_question_summary(df)

    sns.set_theme(style="whitegrid")

    _plot_heatmap(df, config, n_clones, n_jaccard, subfolder, base,
                  min_nonclone_dist=min_nonclone_dist)
    _plot_avg_curve(df, config, n_clones, n_jaccard, subfolder, base,
                    min_nonclone_dist=min_nonclone_dist)
    _plot_per_question(summary, config, n_clones, n_jaccard, subfolder, base)
    _plot_optimal_alpha_hist(summary, config, n_clones, subfolder, base)
    _save_report(df, summary, config, n_clones, n_jaccard, subfolder, base)


def _compute_per_question_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-question (per clone_type) aggregation: optimal alpha, max CRW Jaccard, baseline distortion."""
    has_clone_types = "clone_type" in df.columns
    group_cols = ["question_id", "clone_type"] if has_clone_types else ["question_id"]

    rows = []
    for key, grp in df.groupby(group_cols):
        if has_clone_types:
            q_id, clone_type = key
        else:
            q_id = key
            clone_type = None

        # Baseline distortion (alpha-independent — take from first row)
        base_jac = grp["base_jaccard_mean"].iloc[0]

        # Best CRW correction
        best_idx = grp["crw_jaccard_mean"].idxmax()
        best_row = grp.loc[best_idx]

        row = {
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
        }
        if has_clone_types:
            row["clone_type"] = clone_type
        rows.append(row)

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
    min_nonclone_dist: float | None = None,
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

    # Mark min non-clone distance (alpha above which OG questions start being grouped)
    if min_nonclone_dist is not None:
        # Find position on the alpha axis (interpolate between discrete alpha columns)
        for k, a in enumerate(alpha_cols):
            if a >= min_nonclone_dist:
                # Interpolate position between k-1 and k
                if k > 0:
                    frac = (min_nonclone_dist - alpha_cols[k - 1]) / (a - alpha_cols[k - 1])
                    pos = (k - 1) + frac + 0.5
                else:
                    pos = 0.5
                ax.axvline(pos, color="red", linewidth=1.5, linestyle=":", alpha=0.8)
                ax.text(pos + 0.2, len(q_order) * 0.02,
                        f"min OG dist={min_nonclone_dist:.3f}",
                        color="red", fontsize=8, rotation=90, va="bottom")
                break

    ax.set_xlabel("Alpha (α)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(
        f"Per-Question CRW Correction: Jaccard vs (Question, Alpha)\n"
        f"({n_clones}× clones, top-{n_jaccard}, {config.district})",
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
    min_nonclone_dist: float | None = None,
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

    # Min non-clone distance line
    if min_nonclone_dist is not None:
        ax.axvline(
            min_nonclone_dist, color="red", linewidth=1.5, linestyle=":", alpha=0.8,
            label=f"Min OG distance ({min_nonclone_dist:.3f})",
        )

    ax.set_xlabel("Alpha (α)", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Average CRW Correction vs Alpha (across all questions)\n"
        f"({n_clones}× clones, top-{n_jaccard}, {config.district})",
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


def main(argv=None):
    args = _parse_args(argv)
    config = load_config(Path(args.config))

    if args.alphas:
        alphas = sorted([float(a.strip()) for a in args.alphas.split(",")])
    else:
        alphas = DEFAULT_ALPHAS

    clone_type = args.clone_type
    n_clones = args.n_clones
    n_jaccard = _resolve_n(config, args.n)
    name = _get_clean_name(config)

    print(f"\n=== Question Alpha Sweep ({args.mode} mode) ===")
    print(f"  Config    : {name}")
    print(f"  Clone type: {clone_type}")
    print(f"  Clones    : {n_clones}")
    print(f"  Alphas    : {len(alphas)} values")
    print(f"  Top-k (n) : {n_jaccard}")

    if args.mode == "prepare":
        _run_prepare(config, n_clones, clone_type=clone_type)
    elif args.mode == "sweep":
        _run_sweep(args, config, alphas, n_clones, n_jaccard, clone_type=clone_type)
    elif args.mode == "worker":
        _run_worker(args, config, alphas, n_clones, n_jaccard, clone_type=clone_type)
    elif args.mode == "collect":
        _run_collect(args, config, alphas, n_clones, n_jaccard)


if __name__ == "__main__":
    main()
