"""
Threshold alpha sweep: investigates CRW behavior when alpha crosses the
minimum non-clone distance boundary for identical clones.

Theoretical prediction: for identical clones (distance=0 to source), when
alpha < min_non_clone_distance, the CRW integral reduces to a constant
(alpha cancels in width * density = alpha * 1/alpha = 1), so ALL
sub-threshold alpha values produce identical weights.

This script:
1. Computes min_non_clone_distance from the cloned config's distance data
2. Auto-generates alpha values dense around the threshold (~10%-200%)
3. Runs the standard alpha sweep pipeline
4. Produces plots with the threshold clearly marked

Supports three modes: sweep, worker, collect (same as alpha_sweep_main.py).

Usage:
    # Sequential:
    python -m threshold_alpha_sweep_main \\
        --config_a configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --config_b configs/full_pipeline/cloned/identical_combinedvar_n10_e5_ZH.py

    # Worker (one alpha, for SLURM array):
    python -m threshold_alpha_sweep_main --mode worker --task-id 3 \\
        --config_a ... --config_b ... --sweep-dir /path/to/sweep_dir

    # Collect (aggregate workers + plot):
    python -m threshold_alpha_sweep_main --mode collect \\
        --config_a ... --config_b ... --sweep-dir /path/to/sweep_dir

Outputs (saved to experiment_results/threshold_alpha_sweep_results/):
    - threshold_sweep_<a>_vs_<b>_<timestamp>_<hash>.csv
    - threshold_sweep_<a>_vs_<b>_<timestamp>_<hash>_metrics.png
    - threshold_sweep_<a>_vs_<b>_<timestamp>_<hash>_jaccard_dist.png
    - threshold_sweep_<a>_vs_<b>_<timestamp>_<hash>_report.txt
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

from alpha_sweep_main import (
    _compute_alpha,
    _get_clean_name,
    _get_or_compute_analysis,
    _resolve_n,
    _setup_pipeline,
)
from configs import base_constants as default_config
from cross_run_analysis.analyzer import CrossRunAnalyzer
from main import load_config
from vqs.data_loader import load_dataset
from vqs.distance_utils import compute_min_non_clone_distance
from vqs.similarity_metrics import get_calculator


# ---------------------------------------------------------------------------
# Alpha generation
# ---------------------------------------------------------------------------


def generate_threshold_alphas(threshold: float) -> list[float]:
    """Generate alpha values dense around the threshold.

    Strategy:
    - Below threshold (plateau confirmation): threshold * {0.1, 0.25, 0.5, 0.75, 0.95}
    - At + just above threshold (transition onset): threshold * {1.0, 1.05, 1.1, 1.2}
    - Transition zone: threshold * {1.3, 1.4, ..., 2.0}
    - Total: ~20 alpha values, all within 10%–200% of threshold
    """
    alphas = set()

    # Below threshold (plateau region)
    for frac in [0.1, 0.25, 0.5, 0.75, 0.95]:
        alphas.add(round(threshold * frac, 6))

    # At and just above threshold (transition onset)
    for frac in [1.0, 1.05, 1.1, 1.2]:
        alphas.add(round(threshold * frac, 6))

    # Transition zone: 1.3x to 2.0x threshold
    for i in range(3, 11):  # 1.3, 1.4, ..., 2.0
        alphas.add(round(threshold * (1.0 + i / 10), 6))

    # Remove zero or near-zero values
    alphas = {a for a in alphas if a > 1e-6}

    return sorted(alphas)


def compute_threshold_and_alphas(config_b) -> tuple[dict, list[float]]:
    """Load distance data for config_b, compute threshold, generate alphas.

    This function is used by all modes and by the SLURM launcher to
    determine the number of alpha values (= SLURM array size).

    Returns:
        (threshold_info dict, sorted alpha list)
    """
    print("\n--- Computing threshold from config_b distances ---")
    dataset_b = load_dataset(config_b)
    calculator_b = get_calculator(config_b)
    dist_df_b = calculator_b.calculate_distance(dataset_b, config_b)

    threshold_info = compute_min_non_clone_distance(dist_df_b)
    threshold = threshold_info["threshold"]

    if threshold is None:
        print("ERROR: No positive distances found — cannot compute threshold.", file=sys.stderr)
        sys.exit(1)

    alphas = generate_threshold_alphas(threshold)

    print(f"  Threshold (min positive distance): {threshold:.6f}")
    print(f"  Min real-real distance:            {threshold_info['min_real_real']:.6f}")
    print(f"  Pair counts: {threshold_info['pair_counts']}")
    print(f"  Generated {len(alphas)} alpha values: {[f'{a:.4f}' for a in alphas]}")

    return threshold_info, alphas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Threshold alpha sweep for CRW behavior around min non-clone distance"
    )
    parser.add_argument(
        "--config_a", type=str, required=True, help="Config for reference run (e.g. baseline)"
    )
    parser.add_argument(
        "--config_b", type=str, required=True, help="Config for comparison run (e.g. cloned)"
    )
    parser.add_argument(
        "-n", "--n", type=int, default=None,
        help="Override top-k for Jaccard (default: derived from config)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["sweep", "worker", "collect"], default="sweep",
        help="Execution mode: sweep (sequential), worker (single alpha), collect (aggregate + plot)",
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="Alpha index for worker mode (typically SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--sweep-dir", type=str, default=None,
        help="Directory for per-alpha worker CSVs (worker writes here, collect reads from here)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_sweep_hash(config_a, config_b, alphas: list[float], n: int, threshold: float) -> str:
    """Deterministic 12-char hash over sweep parameters."""
    payload = {
        "config_a": Path(config_a.__file__).stem,
        "config_b": Path(config_b.__file__).stem,
        "alphas": sorted(alphas),
        "n_jaccard": n,
        "threshold": threshold,
    }
    for param in default_config.COMPARATOR_HASH_PARAMS:
        payload[f"a_{param}"] = getattr(config_a, param, None)
        payload[f"b_{param}"] = getattr(config_b, param, None)
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_threshold_outputs(
    sweep_df: pd.DataFrame,
    std_metrics: dict,
    threshold_info: dict,
    config_a,
    config_b,
    alphas: list[float],
    n: int,
    output_dir: Path,
) -> None:
    threshold = threshold_info["threshold"]
    name_a = _get_clean_name(config_a)
    name_b = _get_clean_name(config_b)
    h = _get_sweep_hash(config_a, config_b, alphas, n, threshold)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = f"threshold_sweep_{name_a}_vs_{name_b}_{timestamp}_{h}"

    # Deduplication check
    existing = list(output_dir.glob(f"*{h}*_metrics.png"))
    if existing:
        print(f"[SKIP SAVE] Threshold sweep with hash {h} already exists: {existing[0].name}")
        return

    # --- CSV ---
    csv_path = output_dir / f"{base}.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"  -> CSV:     {csv_path.name}")

    sns.set_theme(style="whitegrid")
    colors = {"jaccard": "#2196F3", "spearman": "#4CAF50", "kendall": "#FF9800"}

    # --- Graph 1: Main Metrics with Threshold ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        sweep_df["alpha"], sweep_df["crw_jaccard_mean"],
        color=colors["jaccard"], marker="o", markersize=4, label="CRW Jaccard (mean)",
    )
    ax.plot(
        sweep_df["alpha"], sweep_df["crw_spearman_mean"],
        color=colors["spearman"], marker="s", markersize=4, label="CRW Spearman (mean)",
    )
    ax.plot(
        sweep_df["alpha"], sweep_df["crw_kendall_mean"],
        color=colors["kendall"], marker="^", markersize=4, label="CRW Kendall (mean)",
    )

    ax.axhline(
        std_metrics["base_jaccard_mean"],
        color=colors["jaccard"], linestyle="--", alpha=0.45,
        label="Jaccard - no CRW",
    )
    ax.axhline(
        std_metrics["base_spearman_mean"],
        color=colors["spearman"], linestyle="--", alpha=0.45,
        label="Spearman - no CRW",
    )
    ax.axhline(
        std_metrics["base_kendall_mean"],
        color=colors["kendall"], linestyle="--", alpha=0.45,
        label="Kendall - no CRW",
    )

    # Threshold annotation
    ax.axvline(
        threshold, color="red", linestyle="--", alpha=0.7,
        label=f"d_min = {threshold:.4f}",
    )
    ax.axvspan(
        0, threshold, alpha=0.08, color="red",
        label="Plateau (alpha-independent)",
    )

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"CRW Metrics vs Alpha (Threshold Sweep)\n{name_a}  vs  {name_b}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    metrics_path = output_dir / f"{base}_metrics.png"
    fig.savefig(metrics_path, dpi=300)
    plt.close(fig)
    print(f"  -> Metrics: {metrics_path.name}")

    # --- Graph 2: Jaccard Distribution with Threshold ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        sweep_df["alpha"], sweep_df["crw_jaccard_mean"],
        color="#1565C0", marker="o", markersize=4, label="CRW Jaccard (mean)",
    )
    ax.plot(
        sweep_df["alpha"], sweep_df["crw_jaccard_median"],
        color="#2196F3", linestyle="-.", marker="s", markersize=4, label="CRW Jaccard (median)",
    )
    ax.plot(
        sweep_df["alpha"], sweep_df["crw_jaccard_p10"],
        color="#90CAF9", linestyle=":", marker="^", markersize=4, label="CRW Jaccard (10th pctile)",
    )
    ax.axhline(
        std_metrics["base_jaccard_mean"],
        color="#1565C0", linestyle="--", alpha=0.45,
        label="Jaccard - no CRW",
    )

    # Threshold annotation
    ax.axvline(
        threshold, color="red", linestyle="--", alpha=0.7,
        label=f"d_min = {threshold:.4f}",
    )
    ax.axvspan(
        0, threshold, alpha=0.08, color="red",
        label="Plateau (alpha-independent)",
    )

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Jaccard Similarity")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"CRW Jaccard Distribution vs Alpha (Threshold Sweep)\n{name_a}  vs  {name_b}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    jaccard_path = output_dir / f"{base}_jaccard_dist.png"
    fig.savefig(jaccard_path, dpi=300)
    plt.close(fig)
    print(f"  -> Jaccard: {jaccard_path.name}")

    # --- Report ---
    report_lines = [
        f"Threshold Alpha Sweep Report",
        f"{'=' * 50}",
        f"",
        f"Config A (baseline): {name_a}",
        f"Config B (cloned):   {name_b}",
        f"Top-k (n):           {n}",
        f"Timestamp:           {timestamp}",
        f"",
        f"--- Threshold Analysis ---",
        f"Min positive distance (threshold): {threshold:.6f}",
        f"Min real-real distance:            {threshold_info['min_real_real']}",
        f"Pair counts: {json.dumps(threshold_info['pair_counts'], indent=2)}",
        f"",
        f"--- Alpha Values ({len(alphas)}) ---",
        f"Below threshold: {[f'{a:.4f}' for a in alphas if a < threshold]}",
        f"At/above threshold: {[f'{a:.4f}' for a in alphas if a >= threshold]}",
        f"",
        f"--- Baseline (no CRW) ---",
        f"Jaccard mean:  {std_metrics['base_jaccard_mean']:.4f}",
        f"Spearman mean: {std_metrics['base_spearman_mean']:.4f}",
        f"Kendall mean:  {std_metrics['base_kendall_mean']:.4f}",
        f"",
        f"--- Plateau Verification ---",
    ]

    below = sweep_df[sweep_df["is_below_threshold"]]
    if len(below) > 1:
        jac_range = below["crw_jaccard_mean"].max() - below["crw_jaccard_mean"].min()
        spr_range = below["crw_spearman_mean"].max() - below["crw_spearman_mean"].min()
        ken_range = below["crw_kendall_mean"].max() - below["crw_kendall_mean"].min()
        plateau_ok = jac_range < 1e-10 and spr_range < 1e-10 and ken_range < 1e-10
        report_lines.append(
            f"Sub-threshold alpha count: {len(below)}"
        )
        report_lines.append(
            f"Jaccard range:  {jac_range:.2e}  {'FLAT' if jac_range < 1e-10 else 'NOT FLAT'}"
        )
        report_lines.append(
            f"Spearman range: {spr_range:.2e}  {'FLAT' if spr_range < 1e-10 else 'NOT FLAT'}"
        )
        report_lines.append(
            f"Kendall range:  {ken_range:.2e}  {'FLAT' if ken_range < 1e-10 else 'NOT FLAT'}"
        )
        report_lines.append(
            f"Plateau verified: {'YES' if plateau_ok else 'NO'}"
        )
    else:
        report_lines.append("Not enough sub-threshold values to verify plateau.")

    report_lines.extend([
        f"",
        f"--- Per-Alpha Results ---",
        f"{'Alpha':>10}  {'Below?':>6}  {'Jaccard':>8}  {'Spearman':>9}  {'Kendall':>8}",
        f"{'-' * 50}",
    ])
    for _, row in sweep_df.iterrows():
        flag = "*" if row["is_below_threshold"] else " "
        report_lines.append(
            f"{row['alpha']:>10.6f}  {flag:>6}  "
            f"{row['crw_jaccard_mean']:>8.4f}  "
            f"{row['crw_spearman_mean']:>9.4f}  "
            f"{row['crw_kendall_mean']:>8.4f}"
        )

    report_path = output_dir / f"{base}_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"  -> Report:  {report_path.name}")


# ---------------------------------------------------------------------------
# Mode: sweep (sequential)
# ---------------------------------------------------------------------------


def _run_sweep(args, config_a, config_b, n: int):
    pipeline = _setup_pipeline(config_a, config_b)

    threshold_info = compute_min_non_clone_distance(pipeline["dist_df_b"])
    threshold = threshold_info["threshold"]
    if threshold is None:
        print("ERROR: No positive distances found.", file=sys.stderr)
        sys.exit(1)

    alphas = generate_threshold_alphas(threshold)

    print(f"\n--- Threshold: {threshold:.6f} ---")
    print(f"--- {len(alphas)} alpha values ---")

    analyzer = CrossRunAnalyzer.from_n(n)
    sweep_rows = []
    std_metrics: dict | None = None

    for i, alpha in enumerate(alphas):
        print(f"\n--- Alpha {alpha:.6f}  ({i + 1}/{len(alphas)}) ---")

        row, base_metrics = _compute_alpha(alpha, config_a, config_b, pipeline, analyzer, n)
        row["is_below_threshold"] = alpha < threshold
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
            f"  {'[below threshold]' if row['is_below_threshold'] else ''}"
        )

    sweep_df = pd.DataFrame(sweep_rows)

    output_dir = default_config.THRESHOLD_SWEEP_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Saving outputs ---")
    _save_threshold_outputs(
        sweep_df, std_metrics, threshold_info, config_a, config_b, alphas, n, output_dir
    )
    print("\n=== Threshold Alpha Sweep Complete ===")


# ---------------------------------------------------------------------------
# Mode: worker (single alpha for SLURM job array)
# ---------------------------------------------------------------------------


def _run_worker(args, config_a, config_b, n: int):
    task_id = args.task_id
    if task_id is None:
        print("ERROR: --task-id is required in worker mode", file=sys.stderr)
        sys.exit(1)

    # Compute threshold and alphas (deterministic, from cached distances)
    threshold_info, alphas = compute_threshold_and_alphas(config_b)

    if task_id < 0 or task_id >= len(alphas):
        print(
            f"ERROR: --task-id {task_id} out of range [0, {len(alphas) - 1}]",
            file=sys.stderr,
        )
        sys.exit(1)

    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else default_config.THRESHOLD_SWEEP_RESULTS_DIR / "workers"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)

    alpha = alphas[task_id]
    print(f"\n=== Worker: alpha={alpha:.6f} (task {task_id}/{len(alphas) - 1}) ===")

    pipeline = _setup_pipeline(config_a, config_b)
    analyzer = CrossRunAnalyzer.from_n(n)

    row, std_metrics = _compute_alpha(alpha, config_a, config_b, pipeline, analyzer, n)
    row["is_below_threshold"] = alpha < threshold_info["threshold"]

    # Save per-alpha CSV (includes both CRW and baseline metrics)
    worker_row = {**row, **std_metrics, "threshold": threshold_info["threshold"]}
    worker_df = pd.DataFrame([worker_row])
    out_path = sweep_dir / f"threshold_worker_{task_id:03d}_a{alpha}.csv"
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


def _run_collect(args, config_a, config_b, n: int):
    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else default_config.THRESHOLD_SWEEP_RESULTS_DIR / "workers"
    )

    worker_files = sorted(sweep_dir.glob("threshold_worker_*.csv"))
    if not worker_files:
        print(f"ERROR: No worker CSVs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Collect: reading {len(worker_files)} worker CSVs from {sweep_dir} ===")

    dfs = [pd.read_csv(f) for f in worker_files]
    combined = pd.concat(dfs, ignore_index=True).sort_values("alpha").reset_index(drop=True)

    # Extract sweep metrics
    sweep_cols = [
        "alpha", "crw_jaccard_mean", "crw_jaccard_median", "crw_jaccard_p10",
        "crw_spearman_mean", "crw_kendall_mean", "is_below_threshold",
    ]
    sweep_df = combined[sweep_cols]

    # Baseline metrics (alpha-invariant)
    std_metrics = {
        "base_jaccard_mean": combined["base_jaccard_mean"].iloc[0],
        "base_spearman_mean": combined["base_spearman_mean"].iloc[0],
        "base_kendall_mean": combined["base_kendall_mean"].iloc[0],
    }

    # Reconstruct threshold info from workers + re-compute from distances
    threshold_info, alphas = compute_threshold_and_alphas(config_b)
    collected_alphas = sorted(sweep_df["alpha"].tolist())

    print(f"  Alphas collected: {[f'{a:.4f}' for a in collected_alphas]}")
    print(
        f"  STANDARD (no CRW): Jaccard={std_metrics['base_jaccard_mean']:.4f}, "
        f"Spearman={std_metrics['base_spearman_mean']:.4f}, "
        f"Kendall={std_metrics['base_kendall_mean']:.4f}"
    )

    output_dir = default_config.THRESHOLD_SWEEP_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Saving outputs ---")
    _save_threshold_outputs(
        sweep_df, std_metrics, threshold_info, config_a, config_b, collected_alphas, n, output_dir
    )
    print("\n=== Collect Complete ===")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()

    config_a = load_config(Path(args.config_a))
    config_b = load_config(Path(args.config_b))

    n = _resolve_n(config_a, args.n)

    name_a = _get_clean_name(config_a)
    name_b = _get_clean_name(config_b)
    print(f"\n=== Threshold Alpha Sweep ({args.mode} mode) ===")
    print(f"  Config A : {name_a}")
    print(f"  Config B : {name_b}")
    print(f"  Top-k (n): {n}")

    if args.mode == "sweep":
        _run_sweep(args, config_a, config_b, n)
    elif args.mode == "worker":
        _run_worker(args, config_a, config_b, n)
    elif args.mode == "collect":
        _run_collect(args, config_a, config_b, n)


if __name__ == "__main__":
    main()
