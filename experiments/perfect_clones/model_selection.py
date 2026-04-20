"""
Alpha sweep: evaluates CRW robustness by running the CRW+recommendation pipeline
across a range of alpha values for two configs and comparing recommendations.

Supports three modes:
    - sweep (default): Run all alphas sequentially in one process.
    - worker:  Compute a single alpha (by --task-id index). For SLURM job arrays.
    - collect: Read per-alpha worker CSVs from --sweep-dir, aggregate, and plot.

Usage:
    # Sequential (original behavior):
    python -m alpha_sweep_main \\
        --config_a configs/base_pipeline/pipeline_e5_ZH.py \\
        --config_b configs/experiments/perfect_clones_model_selection/identical_highcandvar_n10_e5_ZH.py

    # Worker (one alpha, for SLURM array):
    python -m alpha_sweep_main --mode worker --task-id 3 \\
        --config_a ... --config_b ... --sweep-dir /path/to/sweep_dir

    # Collect (aggregate workers + plot):
    python -m alpha_sweep_main --mode collect \\
        --config_a ... --config_b ... --sweep-dir /path/to/sweep_dir

Outputs (saved to experiment_results/exp1/model_alpha_sweep/):
    - alpha_sweep_<a>_vs_<b>_<timestamp>_<hash>.csv          — sweep data
    - alpha_sweep_<a>_vs_<b>_<timestamp>_<hash>_metrics.png  — CRW vs STANDARD metrics
    - alpha_sweep_<a>_vs_<b>_<timestamp>_<hash>_jaccard_dist.png — Jaccard distribution
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs import base_constants as default_config
from cross_run_analysis.analyzer import CrossRunAnalyzer
from experiments._common import _get_clean_name, _resolve_n, DEFAULT_ALPHAS, DEFAULT_ALPHA_REFERENCE
from vqs.config_utils import load_config
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine
from vqs.result_management import ResultManager
from vqs.similarity_metrics import get_calculator

ANALYSIS_CACHE_DIR = default_config.CACHE_DIR / "alpha_sweep_analysis"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Alpha sweep for CRW robustness evaluation"
    )
    parser.add_argument(
        "--config_a", type=str, required=True, help="Config for reference run (e.g. baseline)"
    )
    parser.add_argument(
        "--config_b", type=str, required=True, help="Config for comparison run (e.g. cloned)"
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated alpha values (default: 0.0 to 3.0 in 0.1 steps)",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=None,
        help="Override top-k for Jaccard (default: derived from config)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sweep", "worker", "collect"],
        default="sweep",
        help="Execution mode: sweep (sequential), worker (single alpha), collect (aggregate + plot)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Alpha index for worker mode (typically SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Directory for per-alpha worker CSVs (worker writes here, collect reads from here)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for results (default: ALPHA_SWEEP_RESULTS_DIR)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




def _get_sweep_hash(config_a, config_b, alphas: list[float], n: int) -> str:
    """Deterministic 12-char hash over sweep parameters."""
    payload = {
        "config_a": Path(config_a.__file__).stem,
        "config_b": Path(config_b.__file__).stem,
        "alphas": sorted(alphas),
        "n_jaccard": n,
    }
    for param in default_config.COMPARATOR_HASH_PARAMS:
        payload[f"a_{param}"] = getattr(config_a, param, None)
        payload[f"b_{param}"] = getattr(config_b, param, None)
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _get_analysis_cache_hash(config_a, config_b, alpha: float, n: int) -> str:
    """Deterministic hash for a single cross-run analysis at a given alpha."""
    payload = {
        "config_a": Path(config_a.__file__).stem,
        "config_b": Path(config_b.__file__).stem,
        "alpha": alpha,
        "n_jaccard": n,
    }
    for param in default_config.COMPARATOR_HASH_PARAMS:
        payload[f"a_{param}"] = getattr(config_a, param, None)
        payload[f"b_{param}"] = getattr(config_b, param, None)
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Cross-run analysis caching
# ---------------------------------------------------------------------------


def _get_or_compute_analysis(
    analyzer: CrossRunAnalyzer,
    rec_df_a: pd.DataFrame,
    rec_df_b: pd.DataFrame,
    config_a,
    config_b,
    alpha: float,
    n: int,
) -> pd.DataFrame:
    """Run cross-run analysis with file-based caching."""
    cache_dir = ANALYSIS_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    h = _get_analysis_cache_hash(config_a, config_b, alpha, n)
    cache_path = cache_dir / f"analysis_a{alpha}_{h}.parquet"

    if cache_path.exists():
        print(f"  [CACHE HIT] Cross-run analysis: {cache_path.name}")
        return pd.read_parquet(cache_path)

    results = analyzer.analyze_from_dfs(rec_df_a, rec_df_b)
    results.to_parquet(cache_path, index=False)
    print(f"  [CACHE SAVE] Cross-run analysis: {cache_path.name}")
    return results


# ---------------------------------------------------------------------------
# Per-alpha recommendation computation (with caching, baseline-separated)
# ---------------------------------------------------------------------------


def _get_or_compute_recs(
    config,
    rec_engine: RecommendationEngine,
    dist_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the combined (baseline + CRW) recommendation DataFrame for the current
    config.alpha. Uses the same cache format as RecommendationEngine.evaluate_pipeline,
    so results are mutually compatible.

    Separating baseline from CRW means we only recompute the expensive
    baseline once per config rather than once per alpha value.
    """
    # Build prefix exactly like evaluate_pipeline
    prefix = (
        f"recs_{config.data_year}_{config.dist}"
        f"_a{config.alpha}_subset={config.subset_n}"
    )
    if config.district != "all":
        prefix += f"_{config.district}"
    if rec_engine.n_recs is not None:
        prefix += f"_top{rec_engine.n_recs}"

    rm = ResultManager(
        config=config,
        dir=config.RECOMMENDATION_CACHE_DIR,
        params_list=rec_engine.important_params_list,
        prefix=prefix,
    )

    cached = rm.load()
    if cached is not None:
        return cached

    # Compute CRW weights for this alpha
    reweighter = CloneRobustReweighter(config)
    weights_df = reweighter.reweight(dist_df)

    # Compute CRW recommendations
    crw_df = rec_engine.run_crw(weights_df)

    # Combine (mirrors evaluate_pipeline logic exactly)
    match_cols = [c for c in crw_df.columns if "match" in c or "Dist" in c]
    crw_prefixed = crw_df[match_cols].add_prefix("CRW_")
    combined = baseline_df.join(crw_prefixed)

    rm.save(data=combined)
    return combined


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_outputs(
    sweep_df: pd.DataFrame,
    std_metrics: dict,
    config_a,
    config_b,
    alphas: list[float],
    n: int,
    output_dir: Path,
) -> None:
    name_a = _get_clean_name(config_a)
    name_b = _get_clean_name(config_b)
    h = _get_sweep_hash(config_a, config_b, alphas, n)
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # Per-experiment subfolder (matches CrossRunSaver pattern)
    subfolder_name = f"alpha_sweep_{name_a}_vs_{name_b}"
    subfolder = output_dir / subfolder_name
    subfolder.mkdir(parents=True, exist_ok=True)

    base = f"alpha_sweep_{name_a}_vs_{name_b}_{timestamp}_{h}"

    # Deduplication check
    existing = list(subfolder.glob(f"*{h}*_metrics.png"))
    if existing:
        print(f"[SKIP SAVE] Alpha sweep with hash {h} already exists: {existing[0].name}")
        return

    # --- CSV ---
    csv_path = subfolder / f"{base}.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"  -> CSV:     {subfolder_name}/{csv_path.name}")

    sns.set_theme(style="whitegrid")
    colors = {"jaccard": "#2196F3", "spearman": "#4CAF50", "kendall": "#FF9800"}
    alpha_ref = DEFAULT_ALPHA_REFERENCE

    # --- Graph 1: Main Metrics ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        sweep_df["alpha"], sweep_df["crw_jaccard_mean"],
        color=colors["jaccard"], label="CRW Jaccard (mean)",
    )
    ax.plot(
        sweep_df["alpha"], sweep_df["crw_spearman_mean"],
        color=colors["spearman"], label="CRW Spearman (mean)",
    )
    ax.plot(
        sweep_df["alpha"], sweep_df["crw_kendall_mean"],
        color=colors["kendall"], label="CRW Kendall (mean)",
    )

    ax.axhline(
        std_metrics["base_jaccard_mean"],
        color=colors["jaccard"], linestyle="--", alpha=0.45,
        label="Jaccard – no CRW",
    )
    ax.axhline(
        std_metrics["base_spearman_mean"],
        color=colors["spearman"], linestyle="--", alpha=0.45,
        label="Spearman – no CRW",
    )
    ax.axhline(
        std_metrics["base_kendall_mean"],
        color=colors["kendall"], linestyle="--", alpha=0.45,
        label="Kendall – no CRW",
    )
    ax.axvline(
        alpha_ref, color="grey", linestyle="--", alpha=0.5,
        label=f"Default α={alpha_ref}",
    )

    ax.set_xlabel("Alpha (α)")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"CRW Metrics vs Alpha\n{name_a}  vs  {name_b}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    metrics_path = subfolder / f"{base}_metrics.png"
    fig.savefig(metrics_path, dpi=300)
    plt.close(fig)
    print(f"  -> Metrics: {subfolder_name}/{metrics_path.name}")

    # --- Graph 2: Jaccard Distribution ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        sweep_df["alpha"], sweep_df["crw_jaccard_mean"],
        color="#1565C0", label="CRW Jaccard (mean)",
    )
    ax.plot(
        sweep_df["alpha"], sweep_df["crw_jaccard_median"],
        color="#2196F3", linestyle="-.", label="CRW Jaccard (median)",
    )
    ax.plot(
        sweep_df["alpha"], sweep_df["crw_jaccard_p10"],
        color="#90CAF9", linestyle=":", label="CRW Jaccard (10th pctile)",
    )
    ax.axhline(
        std_metrics["base_jaccard_mean"],
        color="#1565C0", linestyle="--", alpha=0.45,
        label="Jaccard – no CRW",
    )
    ax.axvline(
        alpha_ref, color="grey", linestyle="--", alpha=0.5,
        label=f"Default α={alpha_ref}",
    )

    ax.set_xlabel("Alpha (α)")
    ax.set_ylabel("Jaccard Similarity")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"CRW Jaccard Distribution vs Alpha\n{name_a}  vs  {name_b}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    jaccard_path = subfolder / f"{base}_jaccard_dist.png"
    fig.savefig(jaccard_path, dpi=300)
    plt.close(fig)
    print(f"  -> Jaccard: {subfolder_name}/{jaccard_path.name}")


# ---------------------------------------------------------------------------
# Shared setup: load data, distances, baselines
# ---------------------------------------------------------------------------


def _setup_side(config, dataset=None, dist_df=None):
    """Set up one side of the alpha sweep comparison.

    Optionally accepts pre-loaded dataset and/or distance DataFrame
    for in-memory cloning workflows (e.g. question_alpha_sweep_main).
    """
    if dataset is None:
        dataset = load_dataset(config)

    if dist_df is None:
        calculator = get_calculator(config)
        dist_df = calculator.calculate_distance(dataset, config)

    rec_engine = RecommendationEngine(config=config, data_map=dataset)
    baseline = rec_engine.run_baseline()

    return {
        "dataset": dataset,
        "dist_df": dist_df,
        "rec_engine": rec_engine,
        "baseline": baseline,
    }


def _setup_pipeline(config_a, config_b):
    """Load datasets, compute distances, build rec engines + baselines. Returns all shared state."""
    print("\n--- Setting up config A ---")
    side_a = _setup_side(config_a)

    print("\n--- Setting up config B ---")
    side_b = _setup_side(config_b)

    return {
        "dist_df_a": side_a["dist_df"], "dist_df_b": side_b["dist_df"],
        "rec_engine_a": side_a["rec_engine"], "rec_engine_b": side_b["rec_engine"],
        "baseline_a": side_a["baseline"], "baseline_b": side_b["baseline"],
    }


# ---------------------------------------------------------------------------
# Per-alpha computation (shared by sweep and worker modes)
# ---------------------------------------------------------------------------


def _compute_alpha(
    alpha: float,
    config_a,
    config_b,
    pipeline: dict,
    analyzer: CrossRunAnalyzer,
    n: int,
) -> tuple[dict, dict | None]:
    """
    Compute metrics for a single alpha value. Returns (row_dict, std_metrics_or_None).
    std_metrics is returned on the first call for baseline (alpha-invariant) metrics.
    """
    config_a.alpha = alpha
    config_b.alpha = alpha

    rec_df_a = _get_or_compute_recs(
        config_a, pipeline["rec_engine_a"], pipeline["dist_df_a"], pipeline["baseline_a"]
    )
    rec_df_b = _get_or_compute_recs(
        config_b, pipeline["rec_engine_b"], pipeline["dist_df_b"], pipeline["baseline_b"]
    )

    sys.stdout.flush()

    results = _get_or_compute_analysis(
        analyzer, rec_df_a, rec_df_b, config_a, config_b, alpha, n
    )

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

    return row, std_metrics


# ---------------------------------------------------------------------------
# Mode: sweep (original sequential behavior)
# ---------------------------------------------------------------------------


def _run_sweep(args, config_a, config_b, alphas: list[float], n: int):
    pipeline = _setup_pipeline(config_a, config_b)
    analyzer = CrossRunAnalyzer.from_n(n)

    sweep_rows = []
    std_metrics: dict | None = None

    for i, alpha in enumerate(alphas):
        print(f"\n--- Alpha {alpha:.2f}  ({i + 1}/{len(alphas)}) ---")

        row, base_metrics = _compute_alpha(alpha, config_a, config_b, pipeline, analyzer, n)
        sweep_rows.append(row)

        if std_metrics is None:
            std_metrics = base_metrics
            print(
                f"  STANDARD (no CRW): Jaccard={std_metrics['base_jaccard_mean']:.4f}, "
                f"Spearman={std_metrics['base_spearman_mean']:.4f}, "
                f"Kendall={std_metrics['base_kendall_mean']:.4f}"
            )

        print(
            f"  CRW: Jaccard={row['crw_jaccard_mean']:.4f} "
            f"(med={row['crw_jaccard_median']:.4f}, p10={row['crw_jaccard_p10']:.4f}), "
            f"Spearman={row['crw_spearman_mean']:.4f}, "
            f"Kendall={row['crw_kendall_mean']:.4f}"
        )

    sweep_df = pd.DataFrame(sweep_rows)

    output_dir = Path(args.output_dir) if args.output_dir else default_config.ALPHA_SWEEP_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Saving outputs ---")
    _save_outputs(sweep_df, std_metrics, config_a, config_b, alphas, n, output_dir)
    print("\n=== Alpha Sweep Complete ===")


# ---------------------------------------------------------------------------
# Mode: worker (single alpha for SLURM job array)
# ---------------------------------------------------------------------------


def _run_worker(args, config_a, config_b, alphas: list[float], n: int):
    task_id = args.task_id
    if task_id is None:
        print("ERROR: --task-id is required in worker mode", file=sys.stderr)
        sys.exit(1)
    if task_id < 0 or task_id >= len(alphas):
        print(
            f"ERROR: --task-id {task_id} out of range [0, {len(alphas) - 1}]",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
    else:
        base_dir = Path(args.output_dir) if args.output_dir else default_config.ALPHA_SWEEP_RESULTS_DIR
        name_a = _get_clean_name(config_a)
        name_b = _get_clean_name(config_b)
        subfolder_name = f"alpha_sweep_{name_a}_vs_{name_b}"
        sweep_dir = base_dir / subfolder_name / "workers"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    alpha = alphas[task_id]
    print(f"\n=== Worker: alpha={alpha:.2f} (task {task_id}/{len(alphas) - 1}) ===")

    pipeline = _setup_pipeline(config_a, config_b)
    analyzer = CrossRunAnalyzer.from_n(n)

    row, std_metrics = _compute_alpha(alpha, config_a, config_b, pipeline, analyzer, n)

    # Save per-alpha CSV (includes both CRW and baseline metrics)
    worker_row = {**row, **std_metrics}
    worker_df = pd.DataFrame([worker_row])
    out_path = sweep_dir / f"alpha_worker_{task_id:03d}_a{alpha}.csv"
    worker_df.to_csv(out_path, index=False)

    print(f"\n  -> Worker CSV: {out_path}")
    print(
        f"  CRW: Jaccard={row['crw_jaccard_mean']:.4f}, "
        f"Spearman={row['crw_spearman_mean']:.4f}, "
        f"Kendall={row['crw_kendall_mean']:.4f}"
    )
    print(f"\n=== Worker Complete ===")


# ---------------------------------------------------------------------------
# Mode: collect (aggregate worker CSVs + plot)
# ---------------------------------------------------------------------------


def _run_collect(args, config_a, config_b, alphas: list[float], n: int):
    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
    else:
        base_dir = Path(args.output_dir) if args.output_dir else default_config.ALPHA_SWEEP_RESULTS_DIR
        name_a = _get_clean_name(config_a)
        name_b = _get_clean_name(config_b)
        subfolder_name = f"alpha_sweep_{name_a}_vs_{name_b}"
        sweep_dir = base_dir / subfolder_name / "workers"

    worker_files = sorted(sweep_dir.glob("alpha_worker_*.csv"))
    if not worker_files:
        print(f"ERROR: No worker CSVs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Collect: reading {len(worker_files)} worker CSVs from {sweep_dir} ===")

    dfs = [pd.read_csv(f) for f in worker_files]
    combined = pd.concat(dfs, ignore_index=True).sort_values("alpha").reset_index(drop=True)

    # Extract sweep metrics (CRW columns only)
    sweep_cols = ["alpha", "crw_jaccard_mean", "crw_jaccard_median", "crw_jaccard_p10",
                  "crw_spearman_mean", "crw_kendall_mean"]
    sweep_df = combined[sweep_cols]

    # Baseline metrics are alpha-invariant; take from first row
    std_metrics = {
        "base_jaccard_mean": combined["base_jaccard_mean"].iloc[0],
        "base_spearman_mean": combined["base_spearman_mean"].iloc[0],
        "base_kendall_mean": combined["base_kendall_mean"].iloc[0],
    }

    print(f"  Alphas collected: {sorted(sweep_df['alpha'].tolist())}")
    print(
        f"  STANDARD (no CRW): Jaccard={std_metrics['base_jaccard_mean']:.4f}, "
        f"Spearman={std_metrics['base_spearman_mean']:.4f}, "
        f"Kendall={std_metrics['base_kendall_mean']:.4f}"
    )

    output_dir = Path(args.output_dir) if args.output_dir else default_config.ALPHA_SWEEP_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    collected_alphas = sorted(sweep_df["alpha"].tolist())

    print("\n--- Saving outputs ---")
    _save_outputs(sweep_df, std_metrics, config_a, config_b, collected_alphas, n, output_dir)
    print("\n=== Collect Complete ===")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = _parse_args(argv)

    # Parse alpha values
    if args.alphas:
        alphas = [float(a.strip()) for a in args.alphas.split(",")]
    else:
        alphas = DEFAULT_ALPHAS

    # Load configs
    config_a = load_config(Path(args.config_a))
    config_b = load_config(Path(args.config_b))

    # Resolve n_jaccard
    n = _resolve_n(config_a, args.n)

    name_a = _get_clean_name(config_a)
    name_b = _get_clean_name(config_b)
    print(f"\n=== Alpha Sweep ({args.mode} mode) ===")
    print(f"  Config A : {name_a}")
    print(f"  Config B : {name_b}")
    print(f"  Alphas   : {alphas}")
    print(f"  Top-k (n): {n}")

    if args.mode == "sweep":
        _run_sweep(args, config_a, config_b, alphas, n)
    elif args.mode == "worker":
        _run_worker(args, config_a, config_b, alphas, n)
    elif args.mode == "collect":
        _run_collect(args, config_a, config_b, alphas, n)


if __name__ == "__main__":
    main()
