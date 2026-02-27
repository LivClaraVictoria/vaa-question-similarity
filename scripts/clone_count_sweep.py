"""
Clone count sweep: measures how recommendation metrics (Jaccard, Spearman, Kendall)
change as more identical clones of a specific question are added to the dataset.

Baseline-only comparison: no CRW involved. Measures raw distortion from cloning.

Usage:
    python scripts/clone_count_sweep.py \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --question-id 32214

    # Custom n values:
    python scripts/clone_count_sweep.py \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --question-id 32214 --n-values 1,2,3,5,10,20,50

Outputs (saved to experiment_results/clone_count_sweep_results/):
    - clone_count_sweep_*.csv            -- one row per n_clones
    - clone_count_sweep_*_metrics.png    -- Jaccard/Spearman/Kendall vs n_clones
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

from clone_pipeline.applicator import apply_specs
from clone_pipeline.spec import CloneSpec
from cross_run_analysis.analyzer import CrossRunAnalyzer
from main import load_config
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine

DEFAULT_N_VALUES = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
RESULTS_DIR = Path("experiment_results/clone_count_sweep_results")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Clone count sweep: measure recommendation distortion vs number of clones"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Base pipeline config (e.g. configs/full_pipeline/base_data/pipeline_e5_ZH.py)",
    )
    parser.add_argument(
        "--question-id", type=int, required=True,
        help="Question ID to clone (e.g. 32214)",
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


def _get_sweep_hash(config, question_id: int, n_values: list[int], n_jaccard: int) -> str:
    payload = {
        "config": Path(config.__file__).stem,
        "question_id": question_id,
        "n_values": sorted(n_values),
        "n_jaccard": n_jaccard,
        "data_year": config.data_year,
        "district": config.district,
        "rec_dist_method": config.rec_dist_method,
    }
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _setup_pipeline(config, question_id: int):
    """Load dataset and compute base baseline recommendations."""
    print("\n--- Loading dataset ---")
    dataset = load_dataset(config)

    questions_df = dataset["questions"]
    if question_id not in questions_df["ID_question"].values:
        print(f"ERROR: Question {question_id} not found in dataset", file=sys.stderr)
        sys.exit(1)

    print("\n--- Computing base baseline recommendations ---")
    rec_engine = RecommendationEngine(config=config, data_map=dataset)
    base_recs = rec_engine.run_baseline()

    base_rankings = CrossRunAnalyzer._extract_rankings(base_recs)

    return {
        "dataset": dataset,
        "base_rankings": base_rankings,
    }


# ---------------------------------------------------------------------------
# Per-n comparison (baseline only)
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


def _compute_for_n(
    n_clones: int,
    question_id: int,
    config,
    pipeline: dict,
    n_jaccard: int,
) -> dict:
    """Clone question n_clones times, compute cloned baseline recs, compare with base."""
    dataset = pipeline["dataset"]

    spec = CloneSpec(source_q_id=question_id, clone_type="identical", n_clones=n_clones)
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

    return {
        "n_clones": n_clones,
        "jaccard_mean": per_voter["jaccard"].mean(),
        "jaccard_median": per_voter["jaccard"].median(),
        "jaccard_p10": per_voter["jaccard"].quantile(0.1),
        "spearman_mean": per_voter["spearman"].mean(),
        "kendall_mean": per_voter["kendall"].mean(),
        "n_changed_mean": per_voter["n_changed"].mean(),
        "avg_pos_moved_mean": per_voter["avg_pos_moved"].mean(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_outputs(
    sweep_df: pd.DataFrame,
    config,
    question_id: int,
    n_values: list[int],
    n_jaccard: int,
    output_dir: Path,
) -> None:
    name = _get_clean_name(config)
    h = _get_sweep_hash(config, question_id, n_values, n_jaccard)
    timestamp = datetime.now().strftime("%m%d_%H%M")

    subfolder_name = f"clone_count_sweep_{name}_q{question_id}"
    subfolder = output_dir / subfolder_name
    subfolder.mkdir(parents=True, exist_ok=True)

    base = f"clone_count_sweep_q{question_id}_{name}_{timestamp}_{h}"

    # Deduplication check
    existing = list(subfolder.glob(f"*{h}*_metrics.png"))
    if existing:
        print(f"[SKIP SAVE] Clone count sweep with hash {h} already exists: {existing[0].name}")
        return

    csv_path = subfolder / f"{base}.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"  -> CSV: {subfolder_name}/{csv_path.name}")

    _plot_metrics(sweep_df, config, question_id, n_jaccard, subfolder, base)


def _plot_metrics(
    sweep_df: pd.DataFrame,
    config,
    question_id: int,
    n_jaccard: int,
    output_dir: Path,
    base: str,
):
    sns.set_theme(style="whitegrid")
    colors = {"jaccard": "#2196F3", "spearman": "#4CAF50", "kendall": "#FF9800"}

    fig, ax = plt.subplots(figsize=(10, 6))

    x = sweep_df["n_clones"]

    ax.plot(
        x, sweep_df["jaccard_mean"], color=colors["jaccard"],
        marker="o", linewidth=2, label="Jaccard (mean)",
    )
    ax.plot(
        x, sweep_df["spearman_mean"], color=colors["spearman"],
        marker="s", linewidth=2, label="Spearman (mean)",
    )
    ax.plot(
        x, sweep_df["kendall_mean"], color=colors["kendall"],
        marker="^", linewidth=2, label="Kendall (mean)",
    )

    ax.plot(
        x, sweep_df["jaccard_median"], color=colors["jaccard"],
        marker="o", linewidth=1.5, linestyle="-.", alpha=0.6, label="Jaccard (median)",
    )
    ax.plot(
        x, sweep_df["jaccard_p10"], color=colors["jaccard"],
        marker="o", linewidth=1.5, linestyle=":", alpha=0.5, label="Jaccard (10th pctile)",
    )

    ax.axhline(1.0, color="grey", linestyle=":", alpha=0.3)

    ax.set_xlabel("Number of Clones Added", fontsize=12)
    ax.set_ylabel("Metric Value (higher = less distortion)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_title(
        f"Recommendation Distortion vs Number of Clones\n"
        f"Question {question_id} (identical clones), "
        f"top-{n_jaccard}, {config.district}",
        fontsize=13,
    )
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    path = output_dir / f"{base}_metrics.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Plot: {output_dir.name}/{path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()
    config = load_config(Path(args.config))

    question_id = args.question_id

    if args.n_values:
        n_values = sorted([int(v.strip()) for v in args.n_values.split(",")])
    else:
        n_values = DEFAULT_N_VALUES

    n_jaccard = _resolve_n(config, args.n)
    name = _get_clean_name(config)

    print(f"\n=== Clone Count Sweep ===")
    print(f"  Config    : {name}")
    print(f"  Question  : {question_id}")
    print(f"  n values  : {n_values}")
    print(f"  Top-k (n) : {n_jaccard}")

    pipeline = _setup_pipeline(config, question_id)

    rows = []
    for i, n_clones in enumerate(n_values):
        print(f"\n--- n_clones={n_clones} ({i + 1}/{len(n_values)}) ---")
        row = _compute_for_n(n_clones, question_id, config, pipeline, n_jaccard)
        rows.append(row)
        print(
            f"  Jaccard={row['jaccard_mean']:.4f}, "
            f"Spearman={row['spearman_mean']:.4f}, "
            f"Kendall={row['kendall_mean']:.4f}"
        )

    sweep_df = pd.DataFrame(rows)

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_outputs(sweep_df, config, question_id, n_values, n_jaccard, output_dir)

    print("\n=== Clone Count Sweep Complete ===")


if __name__ == "__main__":
    main()
