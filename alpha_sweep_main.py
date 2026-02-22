"""
Alpha sweep: evaluates CRW robustness by running the CRW+recommendation pipeline
across a range of alpha values for two configs and comparing recommendations.

Usage:
    python -m alpha_sweep_main \\
        --config_a configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --config_b configs/full_pipeline/cloned/identical_highcandvar_n10_e5_ZH.py \\
        [--alphas 0.1,0.2,...,1.5] \\
        [--n 36]

Outputs (saved to experiment_results/comparator_results/):
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
from main import load_config
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine
from vqs.result_management import ResultManager
from vqs.similarity_metrics import get_calculator

DEFAULT_ALPHAS = [round(0.1 * i, 1) for i in range(1, 16)]  # 0.1 to 1.5
DEFAULT_ALPHA_REFERENCE = 0.6


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
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
        help="Comma-separated alpha values (default: 0.1 to 1.5 in 0.1 steps)",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=None,
        help="Override top-k for Jaccard (default: derived from config)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_clean_name(config) -> str:
    """Readable run name (matches CrossRunSaver._clean_name logic)."""
    base = Path(config.__file__).stem
    overrides = getattr(config, "overrides", [])
    if overrides:
        suffix = "_".join(overrides).replace("~", "").replace("=", "")
        return f"{base}_{suffix}"
    return base


def _resolve_n(config, n_override: int | None) -> int:
    """Determine Jaccard top-k from CLI override or config (mirrors recommendation_saver._get_jaccard_n)."""
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


def _get_sweep_hash(config_a, config_b, alphas: list[float], n: int) -> str:
    """Deterministic 12-char hash over sweep parameters."""
    payload = {
        "config_a": Path(config_a.__file__).stem,
        "config_b": Path(config_b.__file__).stem,
        "alphas": sorted(alphas),
        "n_jaccard": n,
        "data_year_a": config_a.data_year,
        "dist_a": config_a.dist,
        "data_choice_a": config_a.data_choice,
        "clone_id_a": config_a.clone_id,
        "data_year_b": config_b.data_year,
        "dist_b": config_b.dist,
        "data_choice_b": config_b.data_choice,
        "clone_id_b": config_b.clone_id,
    }
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


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
    base = f"alpha_sweep_{name_a}_vs_{name_b}_{timestamp}_{h}"

    # Deduplication check
    existing = list(output_dir.glob(f"*{h}*_metrics.png"))
    if existing:
        print(f"[SKIP SAVE] Alpha sweep with hash {h} already exists: {existing[0].name}")
        return

    # --- CSV ---
    csv_path = output_dir / f"{base}.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"  -> CSV:     {csv_path.name}")

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
    metrics_path = output_dir / f"{base}_metrics.png"
    fig.savefig(metrics_path, dpi=300)
    plt.close(fig)
    print(f"  -> Metrics: {metrics_path.name}")

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
    jaccard_path = output_dir / f"{base}_jaccard_dist.png"
    fig.savefig(jaccard_path, dpi=300)
    plt.close(fig)
    print(f"  -> Jaccard: {jaccard_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()

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
    print(f"\n=== Alpha Sweep ===")
    print(f"  Config A : {name_a}")
    print(f"  Config B : {name_b}")
    print(f"  Alphas   : {alphas}")
    print(f"  Top-k (n): {n}")

    # ------------------------------------------------------------------
    # Step 1: Load datasets (once per config)
    # ------------------------------------------------------------------
    print("\n--- Loading datasets ---")
    dataset_a = load_dataset(config_a)
    dataset_b = load_dataset(config_b)

    # ------------------------------------------------------------------
    # Step 2: Compute distances (once per config; alpha not in dist hash)
    # ------------------------------------------------------------------
    print("\n--- Computing / loading distances ---")
    calculator_a = get_calculator(config_a)
    dist_df_a = calculator_a.calculate_distance(dataset_a, config_a)

    calculator_b = get_calculator(config_b)
    dist_df_b = calculator_b.calculate_distance(dataset_b, config_b)

    # ------------------------------------------------------------------
    # Step 3: Create recommendation engines and run baselines (once)
    # ------------------------------------------------------------------
    print("\n--- Computing baseline recommendations (alpha-independent) ---")
    rec_engine_a = RecommendationEngine(config=config_a, data_map=dataset_a)
    baseline_a = rec_engine_a.run_baseline()

    rec_engine_b = RecommendationEngine(config=config_b, data_map=dataset_b)
    baseline_b = rec_engine_b.run_baseline()

    # ------------------------------------------------------------------
    # Step 4: Create analyzer (reused across all alpha values)
    # ------------------------------------------------------------------
    analyzer = CrossRunAnalyzer.from_n(n)

    # ------------------------------------------------------------------
    # Step 5: Alpha sweep
    # ------------------------------------------------------------------
    sweep_rows = []
    std_metrics: dict | None = None

    for i, alpha in enumerate(alphas):
        print(f"\n--- Alpha {alpha:.2f}  ({i + 1}/{len(alphas)}) ---")

        # Override alpha on both configs (ResultManager reads this at construction)
        config_a.alpha = alpha
        config_b.alpha = alpha

        rec_df_a = _get_or_compute_recs(config_a, rec_engine_a, dist_df_a, baseline_a)
        rec_df_b = _get_or_compute_recs(config_b, rec_engine_b, dist_df_b, baseline_b)

        sys.stdout.flush()

        results = analyzer.analyze_from_dfs(rec_df_a, rec_df_b)

        crw_jac = results["crw_jaccard"]
        sweep_rows.append(
            {
                "alpha": alpha,
                "crw_jaccard_mean": crw_jac.mean(),
                "crw_jaccard_median": crw_jac.median(),
                "crw_jaccard_p10": crw_jac.quantile(0.1),
                "crw_spearman_mean": results["crw_spearman"].mean(),
                "crw_kendall_mean": results["crw_kendall"].mean(),
            }
        )

        # Capture STANDARD metrics from the first alpha (they're alpha-invariant)
        if std_metrics is None:
            std_metrics = {
                "base_jaccard_mean": results["base_jaccard"].mean(),
                "base_spearman_mean": results["base_spearman"].mean(),
                "base_kendall_mean": results["base_kendall"].mean(),
            }
            print(
                f"  STANDARD (no CRW): Jaccard={std_metrics['base_jaccard_mean']:.4f}, "
                f"Spearman={std_metrics['base_spearman_mean']:.4f}, "
                f"Kendall={std_metrics['base_kendall_mean']:.4f}"
            )

        row = sweep_rows[-1]
        print(
            f"  CRW: Jaccard={row['crw_jaccard_mean']:.4f} "
            f"(med={row['crw_jaccard_median']:.4f}, p10={row['crw_jaccard_p10']:.4f}), "
            f"Spearman={row['crw_spearman_mean']:.4f}, "
            f"Kendall={row['crw_kendall_mean']:.4f}"
        )

    sweep_df = pd.DataFrame(sweep_rows)

    # ------------------------------------------------------------------
    # Step 6: Save plots + CSV
    # ------------------------------------------------------------------
    output_dir = default_config.ALPHA_SWEEP_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Saving outputs ---")
    _save_outputs(sweep_df, std_metrics, config_a, config_b, alphas, n, output_dir)

    print("\n=== Alpha Sweep Complete ===")


if __name__ == "__main__":
    main()
