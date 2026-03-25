"""
Mini vs Maxi party impact analysis: tests whether CRW can detect and correct
for real question additions (not synthetic clones) to the expert-curated
30-question mini SmartVote questionnaire.

Phase 1 (sweep/worker/collect): For each of 45 full-only questions, add it to
  the mini questionnaire and measure party visibility delta + redundancy scores.

Phase 2 (phase2): For top-K questions benefiting a target party, add them all
  simultaneously and test CRW correction with multiple distance metrics
  (E5-INSTRUCT, ANSWER-CORRELATION, QWEN3) at multiple alphas.

Usage:
    # Phase 1 — Sequential:
    python -m mini_maxi_party_impact_main \\
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py

    # Phase 1 — Worker (one question, for SLURM array):
    python -m mini_maxi_party_impact_main --mode worker --task-id 3 \\
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py \\
        --sweep-dir /path/to/sweep_dir

    # Phase 1 — Collect (aggregate workers + plot):
    python -m mini_maxi_party_impact_main --mode collect \\
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py \\
        --sweep-dir /path/to/sweep_dir

    # Phase 2 — CRW correction for top-K (by party delta):
    python -m mini_maxi_party_impact_main --mode phase2 \\
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py \\
        --target-party Centre --top-k 5

    # Phase 2 — Corr-weighted selection (delta * max_abs_r):
    python -m mini_maxi_party_impact_main --mode phase2 --selection-mode corr_weighted \\
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py \\
        --target-party Centre --top-k 5

    # Compile — aggregate all 6 per-party Phase 2 results:
    python -m mini_maxi_party_impact_main --mode compile \\
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py

    # Compile — corr-weighted results:
    python -m mini_maxi_party_impact_main --mode compile --selection-mode corr_weighted \\
        --config configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py
"""

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from configs import base_constants as default_config
from experiments._common import _get_clean_name, _get_question_text_col, _resolve_n
from vqs.config_utils import load_config
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine

sys.path.insert(0, str(Path(__file__).resolve().parent / "dependencies" / "rsfp"))
from dependencies import SVDataFrame
from vqs.party_visibility import MAJOR_PARTIES, PARTY2COLOR, _build_candidate_party_map, compute_party_visibility

RESULTS_DIR = default_config.RESULTS_DIR / "party_impact" / "mini_maxi"

# Phase 2 metric configs: (display_name, config_path)
PHASE2_CONFIGS = [
    ("E5-INSTRUCT α=0.3", "configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py"),
    ("E5-INSTRUCT α=0.4", "configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a04.py"),
    ("ANSWER-CORR α=0.3", "configs/full_pipeline/base_data/pipeline_answer_corr_ZH_a03.py"),
    ("ANSWER-CORR α=0.4", "configs/full_pipeline/base_data/pipeline_answer_corr_ZH_a04.py"),
    ("QWEN3 α=0.6", "configs/full_pipeline/base_data/pipeline_qwen3_ZH.py"),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Mini vs Maxi party impact: measure effect of adding "
        "full-only questions to the mini questionnaire"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Base pipeline config (e.g. configs/full_pipeline/base_data/pipeline_e5_instruct_ZH_a03.py)",
    )
    parser.add_argument(
        "--mode", type=str,
        choices=["sweep", "worker", "collect", "phase2", "compile"],
        default="sweep",
        help="Execution mode",
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="Full-only question index for worker mode (typically SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--sweep-dir", type=str, default=None,
        help="Directory for per-question worker CSVs",
    )
    parser.add_argument(
        "-n", type=int, default=None,
        help="Override top-k for party visibility (default: seats per canton)",
    )
    parser.add_argument(
        "--phase1-csv", type=str, default=None,
        help="Path to Phase 1 CSV for Phase 2 (auto-detected if omitted)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top questions to analyse in Phase 2",
    )
    parser.add_argument(
        "--target-party", type=str, default=None,
        help="Target party for Phase 2 (select questions that benefit this party most)",
    )
    parser.add_argument(
        "--selection-mode", type=str,
        choices=["delta", "corr_weighted"],
        default="delta",
        help="Phase 2 question selection: 'delta' = rank by party delta only, "
        "'corr_weighted' = rank by delta * max_abs_r (favours CRW-detectable questions)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Mini dataset operations
# ---------------------------------------------------------------------------


def filter_to_mini(dataset: dict):
    """Filter dataset to mini (rapide=1) questions.

    Returns (mini_dataset, mini_ids, full_only_ids).
    """
    df_questions = dataset["questions"]
    all_real_ids = set(
        df_questions.loc[df_questions["ID_question"] < 9_000_000, "ID_question"]
    )
    mini_ids = set(
        df_questions.loc[df_questions["rapide"] == 1, "ID_question"]
    )
    full_only_ids = sorted(all_real_ids - mini_ids)

    df_q = df_questions[df_questions["ID_question"].isin(mini_ids)].copy()

    keep_answer = {f"answer_{qid}" for qid in mini_ids}
    keep_weight = {f"weight_{qid}" for qid in mini_ids}

    df_v = dataset["voters"].copy()
    drop_v = [
        c for c in df_v.columns
        if (c.startswith("answer_") and c not in keep_answer)
        or (c.startswith("weight_") and c not in keep_weight)
    ]
    df_v = df_v.drop(columns=drop_v)

    df_c = dataset["candidates"].copy()
    drop_c = [
        c for c in df_c.columns
        if c.startswith("answer_") and c not in keep_answer
    ]
    df_c = df_c.drop(columns=drop_c)

    df_v = SVDataFrame(df_v, term=2023)
    df_c = SVDataFrame(df_c, term=2023)

    mini_dataset = {"questions": df_q, "voters": df_v, "candidates": df_c}
    return mini_dataset, mini_ids, full_only_ids


def add_question_to_mini(mini_dataset: dict, full_dataset: dict, q_id: int) -> dict:
    """Add a single full-only question back to the mini dataset.

    Returns a new dataset dict (does not mutate mini_dataset).
    """
    ans_col = f"answer_{q_id}"
    wt_col = f"weight_{q_id}"

    # Questions
    q_row = full_dataset["questions"][
        full_dataset["questions"]["ID_question"] == q_id
    ]
    df_q = pd.concat([mini_dataset["questions"], q_row], ignore_index=True)

    # Voters
    df_v = mini_dataset["voters"].copy()
    if ans_col in full_dataset["voters"].columns:
        df_v[ans_col] = full_dataset["voters"][ans_col].values
    if wt_col in full_dataset["voters"].columns:
        df_v[wt_col] = full_dataset["voters"][wt_col].values
    else:
        df_v[wt_col] = 1.0
    df_v = SVDataFrame(df_v, term=2023)

    # Candidates
    df_c = mini_dataset["candidates"].copy()
    if ans_col in full_dataset["candidates"].columns:
        df_c[ans_col] = full_dataset["candidates"][ans_col].values
    df_c = SVDataFrame(df_c, term=2023)

    return {"questions": df_q, "voters": df_v, "candidates": df_c}


def add_questions_to_mini(
    mini_dataset: dict, full_dataset: dict, q_ids: list[int]
) -> dict:
    """Add multiple full-only questions to the mini dataset at once."""
    result = mini_dataset
    for q_id in q_ids:
        result = add_question_to_mini(result, full_dataset, q_id)
    return result


# ---------------------------------------------------------------------------
# Redundancy scoring
# ---------------------------------------------------------------------------


def compute_redundancy_scores(
    full_voters_df: pd.DataFrame,
    mini_ids: set[int],
    full_only_ids: list[int],
) -> dict[int, dict]:
    """Compute answer-correlation redundancy for each full-only question vs mini set.

    Returns dict: q_id -> {mean_abs_r, max_abs_r, n_high_corr}.
    """
    mini_cols = [f"answer_{qid}" for qid in sorted(mini_ids)]
    # Filter to columns that exist
    mini_cols = [c for c in mini_cols if c in full_voters_df.columns]

    scores = {}
    for q_id in full_only_ids:
        q_col = f"answer_{q_id}"
        if q_col not in full_voters_df.columns:
            scores[q_id] = {"mean_abs_r": 0.0, "max_abs_r": 0.0, "n_high_corr": 0}
            continue

        q_vals = full_voters_df[q_col]
        abs_correlations = []

        for mc in mini_cols:
            m_vals = full_voters_df[mc]
            # Drop rows where either is NaN
            mask = q_vals.notna() & m_vals.notna()
            if mask.sum() < 10:
                continue
            r, _ = pearsonr(q_vals[mask], m_vals[mask])
            abs_correlations.append(abs(r))

        if abs_correlations:
            scores[q_id] = {
                "mean_abs_r": float(np.mean(abs_correlations)),
                "max_abs_r": float(np.max(abs_correlations)),
                "n_high_corr": int(sum(1 for r in abs_correlations if r > 0.3)),
            }
        else:
            scores[q_id] = {"mean_abs_r": 0.0, "max_abs_r": 0.0, "n_high_corr": 0}

    return scores


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _setup_pipeline(config):
    """Load dataset, filter to mini, compute baseline, build maps."""
    print("\n--- Loading full dataset ---")
    full_dataset = load_dataset(config)

    print("\n--- Filtering to mini questionnaire ---")
    mini_dataset, mini_ids, full_only_ids = filter_to_mini(full_dataset)
    print(f"  Mini questions: {len(mini_ids)}, Full-only: {len(full_only_ids)}")

    print("\n--- Computing mini baseline recommendations ---")
    rec_engine = RecommendationEngine(config=config, data_map=mini_dataset)
    mini_base_recs = rec_engine.run_baseline()

    candidate_party_map = _build_candidate_party_map(full_dataset["candidates"])

    print("\n--- Computing redundancy scores ---")
    redundancy = compute_redundancy_scores(
        full_dataset["voters"], mini_ids, full_only_ids
    )

    questions_df = full_dataset["questions"]
    question_ids = full_only_ids

    print(f"\n  Candidates: {len(candidate_party_map)}")
    party_dist = Counter(candidate_party_map.values())
    for p in MAJOR_PARTIES:
        print(f"    {p}: {party_dist.get(p, 0)} candidates")

    return {
        "full_dataset": full_dataset,
        "mini_dataset": mini_dataset,
        "mini_ids": mini_ids,
        "full_only_ids": full_only_ids,
        "mini_base_recs": mini_base_recs,
        "candidate_party_map": candidate_party_map,
        "redundancy": redundancy,
        "question_ids": question_ids,
        "questions_df": questions_df,
    }


# ---------------------------------------------------------------------------
# Per-question Phase 1 computation
# ---------------------------------------------------------------------------


def _compute_question_addition_impact(
    q_id: int,
    config,
    pipeline: dict,
    n: int,
) -> dict:
    """Add question to mini, compute baseline recs, measure party visibility delta."""
    candidate_party_map = pipeline["candidate_party_map"]
    questions_df = pipeline["questions_df"]
    text_col = _get_question_text_col(questions_df)

    # 1. Mini baseline party visibility
    base_visibility = compute_party_visibility(
        pipeline["mini_base_recs"], candidate_party_map, n
    )

    # 2. Add question to mini
    augmented = add_question_to_mini(
        pipeline["mini_dataset"], pipeline["full_dataset"], q_id
    )

    # 3. Compute augmented baseline recs
    aug_engine = RecommendationEngine(config=config, data_map=augmented)
    aug_recs = aug_engine.run_baseline()
    sys.stdout.flush()

    # 4. Augmented party visibility
    aug_visibility = compute_party_visibility(aug_recs, candidate_party_map, n)

    # 5. Build result row
    q_text = questions_df.loc[
        questions_df["ID_question"] == q_id, text_col
    ].iloc[0]
    q_category = questions_df.loc[
        questions_df["ID_question"] == q_id, "_category"
    ].iloc[0]

    redundancy = pipeline["redundancy"].get(q_id, {})

    row = {
        "question_id": q_id,
        "question_text": q_text,
        "category": q_category,
        "mean_abs_r": redundancy.get("mean_abs_r", 0.0),
        "max_abs_r": redundancy.get("max_abs_r", 0.0),
        "n_high_corr": redundancy.get("n_high_corr", 0),
    }

    # Party deltas
    all_parties = sorted(
        set(list(base_visibility.keys()) + list(aug_visibility.keys()))
    )
    for party in all_parties:
        base_v = base_visibility.get(party, 0.0)
        aug_v = aug_visibility.get(party, 0.0)
        row[f"base_{party}"] = base_v
        row[f"added_{party}"] = aug_v
        row[f"delta_{party}"] = aug_v - base_v

    # Max absolute delta
    major_deltas = {p: abs(row.get(f"delta_{p}", 0.0)) for p in MAJOR_PARTIES}
    row["max_abs_delta"] = max(major_deltas.values()) if major_deltas else 0.0
    row["max_delta_party"] = (
        max(major_deltas, key=major_deltas.get) if major_deltas else ""
    )

    # Max positive delta
    positive_deltas = {p: row.get(f"delta_{p}", 0.0) for p in MAJOR_PARTIES}
    row["max_positive_delta"] = max(positive_deltas.values())
    row["max_positive_party"] = max(positive_deltas, key=positive_deltas.get)

    return row


# ---------------------------------------------------------------------------
# Mode: sweep (sequential)
# ---------------------------------------------------------------------------


def _run_sweep(args, config, n: int):
    pipeline = _setup_pipeline(config)
    full_only_ids = pipeline["full_only_ids"]

    rows = []
    for i, q_id in enumerate(full_only_ids):
        print(f"\n--- Question {q_id} ({i + 1}/{len(full_only_ids)}) ---")
        row = _compute_question_addition_impact(q_id, config, pipeline, n)
        rows.append(row)
        print(
            f"  Max delta: {row['max_abs_delta']:.4f} "
            f"({row['max_delta_party']}), "
            f"mean_abs_r: {row['mean_abs_r']:.3f}"
        )

    sweep_df = pd.DataFrame(rows)

    name = _get_clean_name(config)
    output_dir = RESULTS_DIR / "phase1" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_phase1_outputs(sweep_df, config, n, output_dir)

    print("\n=== Mini-Maxi Phase 1 Sweep Complete ===")


# ---------------------------------------------------------------------------
# Mode: worker (single question for SLURM array)
# ---------------------------------------------------------------------------


def _run_worker(args, config, n: int):
    task_id = args.task_id
    if task_id is None:
        print("ERROR: --task-id is required in worker mode", file=sys.stderr)
        sys.exit(1)

    name = _get_clean_name(config)
    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else RESULTS_DIR / "phase1" / "workers" / name
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _setup_pipeline(config)
    full_only_ids = pipeline["full_only_ids"]

    if task_id < 0 or task_id >= len(full_only_ids):
        print(
            f"ERROR: --task-id {task_id} out of range [0, {len(full_only_ids) - 1}]",
            file=sys.stderr,
        )
        sys.exit(1)

    q_id = full_only_ids[task_id]
    out_path = sweep_dir / f"mini_maxi_worker_{task_id:03d}_q{q_id}.csv"

    if out_path.exists():
        print(f"  [SKIP] Worker CSV already exists: {out_path}")
        return

    print(
        f"\n=== Worker: question {q_id} (task {task_id}/{len(full_only_ids) - 1}) ==="
    )

    row = _compute_question_addition_impact(q_id, config, pipeline, n)

    worker_df = pd.DataFrame([row])
    worker_df.to_csv(out_path, index=False)

    print(f"\n  -> Worker CSV: {out_path}")
    print(
        f"  Max delta: {row['max_abs_delta']:.4f} ({row['max_delta_party']}), "
        f"mean_abs_r: {row['mean_abs_r']:.3f}"
    )
    print("\n=== Worker Complete ===")


# ---------------------------------------------------------------------------
# Mode: collect (aggregate + plot)
# ---------------------------------------------------------------------------


def _run_collect(args, config, n: int):
    name = _get_clean_name(config)
    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else RESULTS_DIR / "phase1" / "workers" / name
    )

    worker_files = sorted(sweep_dir.glob("mini_maxi_worker_*.csv"))
    if not worker_files:
        print(f"ERROR: No worker CSVs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\n=== Collect: reading {len(worker_files)} worker CSVs from {sweep_dir} ==="
    )

    dfs = [pd.read_csv(f) for f in worker_files]
    combined = pd.concat(dfs, ignore_index=True)

    output_dir = RESULTS_DIR / "phase1" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_phase1_outputs(combined, config, n, output_dir)

    print("\n=== Collect Complete ===")


# ---------------------------------------------------------------------------
# Phase 1 output generation
# ---------------------------------------------------------------------------


def _save_phase1_outputs(
    df: pd.DataFrame,
    config,
    n: int,
    output_dir: Path,
):
    """Sort, save CSV, generate plots + report."""
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = f"mini_maxi_{name}_{timestamp}"

    df = df.sort_values("max_abs_delta", ascending=False).reset_index(drop=True)

    csv_path = output_dir / f"{base}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> CSV: {csv_path.name}")

    sns.set_theme(style="whitegrid")
    _plot_heatmap(df, output_dir, base)
    _plot_per_party(df, output_dir, base)
    _plot_impact_vs_redundancy(df, output_dir, base)
    _plot_redundancy_distribution(df, output_dir, base)
    _save_phase1_report(df, output_dir, base, n)


# ---------------------------------------------------------------------------
# Phase 1 plots
# ---------------------------------------------------------------------------


def _plot_heatmap(df: pd.DataFrame, output_dir: Path, base: str):
    """Heatmap: rows = questions (sorted by max impact), cols = major parties."""
    delta_cols = [f"delta_{p}" for p in MAJOR_PARTIES if f"delta_{p}" in df.columns]
    col_labels = [p for p in MAJOR_PARTIES if f"delta_{p}" in df.columns]

    if not delta_cols:
        print("  [SKIP] No party delta columns found for heatmap")
        return

    matrix = df[delta_cols].values

    row_labels = [
        f"Q{int(r['question_id'])}  {str(r['question_text'])[:45]}"
        for _, r in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.28)))

    sns.heatmap(
        matrix * 100,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Δ Party Visibility (pp)"},
    )
    ax.set_title(
        "Party Visibility Change from Adding Question to Mini\n"
        "Red = party gains visibility, Blue = party loses visibility"
    )
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=10)
    fig.tight_layout()

    path = output_dir / f"{base}_heatmap.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Heatmap: {path.name}")


def _plot_per_party(df: pd.DataFrame, output_dir: Path, base: str):
    """2×3 grid: for each major party, top-5 questions that benefit it most."""
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))

    for ax, party in zip(axes.flat, MAJOR_PARTIES):
        delta_col = f"delta_{party}"
        if delta_col not in df.columns:
            ax.set_visible(False)
            continue

        top5 = df.nlargest(5, delta_col)
        labels = [
            f"Q{int(r['question_id'])}  {str(r['question_text'])[:30]}"
            for _, r in top5.iterrows()
        ]
        values = top5[delta_col].values * 100

        color = PARTY2COLOR.get(party, "#888888")
        ax.barh(range(len(labels)), values, color=color, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Δ Visibility (pp)")
        ax.set_title(f"{party}", fontsize=13, fontweight="bold", color=color)
        ax.axvline(0, color="black", linewidth=0.5)

    for i in range(len(MAJOR_PARTIES), n_rows * n_cols):
        axes.flat[i].set_visible(False)

    fig.suptitle(
        "Top Questions Benefiting Each Party (added to mini)",
        fontsize=14,
    )
    fig.tight_layout()

    path = output_dir / f"{base}_per_party.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Per-party ranking: {path.name}")


def _plot_impact_vs_redundancy(df: pd.DataFrame, output_dir: Path, base: str):
    """2×3 scatter: impact (Δ party visibility) vs redundancy (mean_abs_r)."""
    if "mean_abs_r" not in df.columns:
        print("  [SKIP] No mean_abs_r column for scatter plot")
        return

    categories = df["category"].unique() if "category" in df.columns else []
    # Build a color map for categories
    cat_colors = {}
    palette = sns.color_palette("husl", n_colors=max(len(categories), 1))
    for i, cat in enumerate(sorted(categories)):
        cat_colors[cat] = palette[i]

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))

    for ax, party in zip(axes.flat, MAJOR_PARTIES):
        delta_col = f"delta_{party}"
        if delta_col not in df.columns:
            ax.set_visible(False)
            continue

        x = df["mean_abs_r"].values
        y = df[delta_col].values * 100

        if "category" in df.columns:
            for cat in sorted(categories):
                mask = df["category"] == cat
                ax.scatter(
                    x[mask], y[mask],
                    c=[cat_colors[cat]], label=cat,
                    alpha=0.7, s=40, edgecolors="white", linewidth=0.5,
                )
        else:
            ax.scatter(x, y, alpha=0.7, s=40)

        # Pearson r annotation
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() > 2:
            r_val, p_val = pearsonr(x[valid], y[valid])
            ax.annotate(
                f"r={r_val:.2f} (p={p_val:.3f})",
                xy=(0.02, 0.98), xycoords="axes fraction",
                fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Mean |Pearson r| with mini questions")
        ax.set_ylabel("Δ Visibility (pp)")

        color = PARTY2COLOR.get(party, "#888888")
        ax.set_title(f"{party}", fontsize=13, fontweight="bold", color=color)

    for i in range(len(MAJOR_PARTIES), n_rows * n_cols):
        axes.flat[i].set_visible(False)

    # Shared legend
    if categories is not None and len(categories) > 0:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=min(len(categories), 5),
            fontsize=7,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(
        "Party Impact vs Redundancy with Mini Questions\n"
        "(each point = one full-only question added to mini)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    path = output_dir / f"{base}_impact_vs_redundancy.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Impact vs redundancy: {path.name}")


def _plot_redundancy_distribution(df: pd.DataFrame, output_dir: Path, base: str):
    """Bar chart of mean_abs_r per question, sorted descending, colored by topic."""
    if "mean_abs_r" not in df.columns:
        print("  [SKIP] No mean_abs_r for redundancy distribution")
        return

    sorted_df = df.sort_values("mean_abs_r", ascending=False).reset_index(drop=True)

    categories = sorted_df["category"].unique() if "category" in sorted_df.columns else []
    cat_colors = {}
    palette = sns.color_palette("husl", n_colors=max(len(categories), 1))
    for i, cat in enumerate(sorted(categories)):
        cat_colors[cat] = palette[i]

    fig, ax = plt.subplots(figsize=(16, 6))

    labels = [
        f"Q{int(r['question_id'])}"
        for _, r in sorted_df.iterrows()
    ]
    values = sorted_df["mean_abs_r"].values
    colors = [
        cat_colors.get(r.get("category", ""), "#888888")
        for _, r in sorted_df.iterrows()
    ]

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Mean |Pearson r| with mini questions")
    ax.set_title("Redundancy of Full-Only Questions with Mini Questionnaire")
    ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="r=0.3 threshold")
    ax.legend()

    # Category legend
    if len(categories) > 0:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cat_colors[c], label=c) for c in sorted(categories)
        ]
        ax.legend(
            handles=legend_elements, loc="upper right",
            fontsize=6, ncol=2,
        )

    fig.tight_layout()

    path = output_dir / f"{base}_redundancy_dist.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Redundancy distribution: {path.name}")


def _save_phase1_report(
    df: pd.DataFrame, output_dir: Path, base: str, n: int, top_n: int = 10
):
    """Save human-readable summary of Phase 1 results."""
    lines = [
        "=" * 80,
        "MINI VS MAXI PARTY IMPACT — PHASE 1 REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Experiment: Add each full-only question to mini, measure party delta",
        f"Top-k (n): {n}",
        f"Questions tested: {len(df)}",
        f"Major parties: {', '.join(MAJOR_PARTIES)}",
        "",
        f"TOP {top_n} QUESTIONS BY MAX PARTY IMPACT (abs Δ visibility):",
        "-" * 80,
    ]

    for i, (_, row) in enumerate(df.head(top_n).iterrows()):
        lines.append(
            f"  {i + 1}. Q{int(row['question_id'])} — "
            f"max |Δ| = {row['max_abs_delta'] * 100:.2f}pp "
            f"({row['max_delta_party']})"
        )
        lines.append(f'     "{str(row["question_text"])[:70]}"')
        lines.append(
            f"     Category: {row.get('category', 'N/A')}, "
            f"mean_abs_r: {row.get('mean_abs_r', 0):.3f}, "
            f"max_abs_r: {row.get('max_abs_r', 0):.3f}, "
            f"n_high_corr: {row.get('n_high_corr', 0)}"
        )
        for p in MAJOR_PARTIES:
            d = row.get(f"delta_{p}", 0)
            if abs(d) > 0.0001:
                lines.append(f"       {p:>8s}: {d * 100:+.2f}pp")
        lines.append("")

    # Per-party summary
    lines.extend(["=" * 80, "PER-PARTY BEST QUESTION:", "-" * 80])
    for p in MAJOR_PARTIES:
        delta_col = f"delta_{p}"
        if delta_col in df.columns:
            best_idx = df[delta_col].idxmax()
            best = df.loc[best_idx]
            lines.append(
                f"  {p:>8s}: Q{int(best['question_id'])} "
                f"({best[delta_col] * 100:+.2f}pp) — "
                f'"{str(best["question_text"])[:50]}" '
                f"[mean_abs_r={best.get('mean_abs_r', 0):.3f}]"
            )

    # Redundancy summary
    lines.extend([
        "",
        "=" * 80,
        "REDUNDANCY SUMMARY:",
        "-" * 80,
        f"  Mean redundancy (mean_abs_r): {df['mean_abs_r'].mean():.3f}",
        f"  Max redundancy: {df['mean_abs_r'].max():.3f} "
        f"(Q{int(df.loc[df['mean_abs_r'].idxmax(), 'question_id'])})",
        f"  Questions with n_high_corr > 0: "
        f"{(df['n_high_corr'] > 0).sum()}/{len(df)}",
    ])

    if "category" in df.columns:
        lines.append("")
        lines.append("  Mean redundancy by category:")
        cat_means = df.groupby("category")["mean_abs_r"].mean().sort_values(ascending=False)
        for cat, val in cat_means.items():
            n_qs = (df["category"] == cat).sum()
            lines.append(f"    {cat}: {val:.3f} ({n_qs} questions)")

    lines.append("=" * 80)

    path = output_dir / f"{base}_report.txt"
    path.write_text("\n".join(lines))
    print(f"  -> Report: {path.name}")


# ---------------------------------------------------------------------------
# Mode: phase2 (CRW correction with multiple distance metrics)
# ---------------------------------------------------------------------------


def _run_phase2(args, config, n: int):
    from vqs.clone_robust_weighting import CloneRobustReweighter
    from vqs.similarity_metrics import get_calculator

    # 1. Load Phase 1 results
    if args.phase1_csv:
        phase1_path = Path(args.phase1_csv)
    else:
        csvs = sorted(
            f for f in (RESULTS_DIR / "phase1").glob("**/mini_maxi_*.csv")
            if "worker" not in f.name
        )
        if not csvs:
            print("ERROR: No Phase 1 CSV found. Run Phase 1 first.", file=sys.stderr)
            sys.exit(1)
        phase1_path = csvs[-1]

    print(f"\n=== Phase 2: Loading Phase 1 results from {phase1_path.name} ===")
    phase1_df = pd.read_csv(phase1_path)

    top_k = args.top_k
    target_party = args.target_party
    selection_mode = getattr(args, "selection_mode", "delta")

    if target_party:
        delta_col = f"delta_{target_party}"
        if delta_col not in phase1_df.columns:
            print(
                f"ERROR: Party '{target_party}' not found in Phase 1 CSV.",
                file=sys.stderr,
            )
            sys.exit(1)

        if selection_mode == "corr_weighted":
            # Only consider questions that benefit the target party
            positive = phase1_df[phase1_df[delta_col] > 0].copy()
            positive["_combined_score"] = positive[delta_col] * positive["max_abs_r"]
            top_questions = positive.nlargest(top_k, "_combined_score")
            print(
                f"  Top-{top_k} questions by delta*max_abs_r "
                f"(corr-weighted, {target_party}):"
            )
        else:
            top_questions = phase1_df.nlargest(top_k, delta_col)
            print(f"  Top-{top_k} questions benefiting {target_party}:")
    else:
        top_questions = phase1_df.nlargest(top_k, "max_abs_delta")
        print(f"  Top-{top_k} questions by max |Δ|:")

    for _, row in top_questions.iterrows():
        delta_str = (
            f"Δ({target_party})={row.get(f'delta_{target_party}', 0) * 100:+.2f}pp"
            if target_party
            else f"|Δ|={row['max_abs_delta'] * 100:.2f}pp ({row['max_delta_party']})"
        )
        extra = ""
        if selection_mode == "corr_weighted" and "_combined_score" in row.index:
            extra = f", score={row['_combined_score']:.4f}"
        print(
            f"    Q{int(row['question_id'])}: {delta_str}, "
            f"mean_abs_r={row.get('mean_abs_r', 0):.3f}, "
            f"max_abs_r={row.get('max_abs_r', 0):.3f}{extra}"
        )

    q_ids = [int(row["question_id"]) for _, row in top_questions.iterrows()]

    # 2. Load full dataset, build mini + additions
    print("\n--- Loading dataset and building mini + additions ---")
    full_dataset = load_dataset(config)
    mini_dataset, mini_ids, full_only_ids = filter_to_mini(full_dataset)

    augmented_dataset = add_questions_to_mini(mini_dataset, full_dataset, q_ids)
    print(
        f"  Mini: {len(mini_ids)} questions, "
        f"Augmented: {len(mini_ids) + len(q_ids)} questions"
    )

    candidate_party_map = _build_candidate_party_map(full_dataset["candidates"])

    # 3. Compute baselines (shared across all metrics)
    print("\n--- Computing mini baseline recommendations ---")
    mini_engine = RecommendationEngine(config=config, data_map=mini_dataset)
    mini_base_recs = mini_engine.run_baseline()
    mini_base_vis = compute_party_visibility(mini_base_recs, candidate_party_map, n)

    print("\n--- Computing augmented baseline recommendations ---")
    aug_engine = RecommendationEngine(config=config, data_map=augmented_dataset)
    aug_base_recs = aug_engine.run_baseline()
    aug_base_vis = compute_party_visibility(aug_base_recs, candidate_party_map, n)

    print("\n  Mini baseline visibility:")
    for p in MAJOR_PARTIES:
        print(f"    {p}: {mini_base_vis.get(p, 0) * 100:.2f}%")
    print("  Augmented baseline visibility:")
    for p in MAJOR_PARTIES:
        d = (aug_base_vis.get(p, 0) - mini_base_vis.get(p, 0)) * 100
        print(f"    {p}: {aug_base_vis.get(p, 0) * 100:.2f}% (Δ{d:+.2f}pp)")

    # 4. For each (metric, alpha): compute CRW
    q_tag = "_".join(str(q) for q in sorted(q_ids))
    metric_results = {}

    for metric_label, config_path in PHASE2_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Metric: {metric_label}")
        print(f"{'='*60}")

        metric_config = load_config(Path(config_path))

        calculator = get_calculator(metric_config)

        # Mini CRW
        print(f"  Computing mini distances + CRW ({metric_label})...")
        mini_dist_df = calculator.calculate_distance(mini_dataset, metric_config)
        reweighter_mini = CloneRobustReweighter(metric_config)
        mini_weights = reweighter_mini.reweight(mini_dist_df)
        mini_crw_recs = mini_engine.run_crw(mini_weights)
        mini_crw_vis = compute_party_visibility(mini_crw_recs, candidate_party_map, n)

        # Augmented CRW
        print(f"  Computing augmented distances + CRW ({metric_label})...")
        saved_clone_id = metric_config.clone_id
        metric_config.clone_id = f"mini_maxi_aug_{q_tag}"
        aug_dist_df = calculator.calculate_distance(augmented_dataset, metric_config)
        reweighter_aug = CloneRobustReweighter(metric_config)
        aug_weights = reweighter_aug.reweight(aug_dist_df)
        aug_crw_recs = aug_engine.run_crw(aug_weights)
        aug_crw_vis = compute_party_visibility(aug_crw_recs, candidate_party_map, n)
        metric_config.clone_id = saved_clone_id

        # Extract CRW weights for added questions
        weight_lookup = aug_weights.set_index("ID_question")["Weight"].to_dict()
        added_q_weights = {q: weight_lookup.get(q, None) for q in q_ids}
        mini_q_weights = {
            q: weight_lookup.get(q, None) for q in sorted(mini_ids)
        }
        avg_mini_weight = np.mean(
            [w for w in mini_q_weights.values() if w is not None]
        )

        metric_results[metric_label] = {
            "mini_crw_vis": mini_crw_vis,
            "aug_crw_vis": aug_crw_vis,
            "added_q_weights": added_q_weights,
            "avg_mini_weight": avg_mini_weight,
        }

        print(f"  Augmented CRW visibility ({metric_label}):")
        for p in MAJOR_PARTIES:
            d = (aug_crw_vis.get(p, 0) - mini_base_vis.get(p, 0)) * 100
            print(f"    {p}: {aug_crw_vis.get(p, 0) * 100:.2f}% (Δ{d:+.2f}pp)")
        print(f"  Added question weights: {added_q_weights}")
        print(f"  Average mini question weight: {avg_mini_weight:.4f}")

    sys.stdout.flush()

    # 5. Build results
    results = {
        "q_ids": q_ids,
        "top_questions": top_questions,
        "visibility": {
            "mini_baseline": mini_base_vis,
            "aug_baseline": aug_base_vis,
        },
        "metric_results": metric_results,
    }

    name = _get_clean_name(config)
    party_subdir = target_party if target_party else "no_target"
    if selection_mode == "corr_weighted":
        output_dir = RESULTS_DIR / "phase2_corr_weighted" / name / party_subdir
    else:
        output_dir = RESULTS_DIR / "phase2" / name / party_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_phase2_outputs(results, config, n, output_dir, target_party, selection_mode)

    print("\n=== Phase 2 Complete ===")


# ---------------------------------------------------------------------------
# Phase 2 output generation
# ---------------------------------------------------------------------------


def _save_phase2_outputs(
    results: dict,
    config,
    n: int,
    output_dir: Path,
    target_party: str | None = None,
    selection_mode: str = "delta",
):
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    party_tag = f"_{target_party}" if target_party else ""
    sel_tag = "_corrwt" if selection_mode == "corr_weighted" else ""
    base = f"mini_maxi_phase2_{name}_{timestamp}{party_tag}{sel_tag}"

    # Save CSV
    vis = results["visibility"]
    metric_results = results["metric_results"]

    rows = []
    for p in MAJOR_PARTIES:
        mb = vis["mini_baseline"].get(p, 0)
        ab = vis["aug_baseline"].get(p, 0)
        da = ab - mb
        row = {
            "party": p,
            "mini_baseline": mb,
            "aug_baseline": ab,
            "delta_attack": da,
        }
        for ml, mr in metric_results.items():
            safe_label = ml.replace(" ", "_").replace("=", "")
            mc = mr["mini_crw_vis"].get(p, 0)
            ac = mr["aug_crw_vis"].get(p, 0)
            dc = ac - mc
            row[f"mini_crw_{safe_label}"] = mc
            row[f"aug_crw_{safe_label}"] = ac
            row[f"crw_drift_{safe_label}"] = mc - mb
            row[f"delta_crw_{safe_label}"] = dc
            row[f"reduction_{safe_label}"] = (
                (1 - abs(dc) / abs(da)) * 100 if abs(da) > 1e-9 else 0
            )
        rows.append(row)

    csv_df = pd.DataFrame(rows)
    csv_path = output_dir / f"{base}.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\n  -> Phase 2 CSV: {csv_path.name}")

    sns.set_theme(style="whitegrid")
    _plot_phase2_grouped_bars(results, output_dir, base, target_party)
    _plot_phase2_deltas(results, output_dir, base, target_party)
    _plot_crw_weight_comparison(results, output_dir, base)
    _save_phase2_report(results, output_dir, base, n, target_party)


# ---------------------------------------------------------------------------
# Phase 2 plots
# ---------------------------------------------------------------------------


def _plot_phase2_grouped_bars(
    results: dict, output_dir: Path, base: str, target_party: str | None = None
):
    """Grouped bar chart: party visibility under all conditions."""
    vis = results["visibility"]
    metric_results = results["metric_results"]

    conditions = [
        ("Mini baseline", vis["mini_baseline"], "#4CAF50"),
        ("Mini + additions baseline", vis["aug_baseline"], "#F44336"),
    ]

    # Add CRW conditions with distinct colors
    crw_colors = ["#2196F3", "#64B5F6", "#FF9800", "#FFB74D", "#9C27B0"]
    for i, (ml, mr) in enumerate(metric_results.items()):
        color = crw_colors[i % len(crw_colors)]
        conditions.append((f"+ CRW ({ml})", mr["aug_crw_vis"], color))

    n_conditions = len(conditions)
    x = np.arange(len(MAJOR_PARTIES))
    width = 0.8 / n_conditions

    fig, ax = plt.subplots(figsize=(16, 7))

    for i, (label, vis_dict, color) in enumerate(conditions):
        offset = (i - n_conditions / 2 + 0.5) * width
        vals = [vis_dict.get(p, 0) * 100 for p in MAJOR_PARTIES]
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(MAJOR_PARTIES, fontsize=12)
    ax.set_ylabel("Party Visibility (%)", fontsize=12)

    n_q = len(results["q_ids"])
    target_str = f" targeting {target_party}" if target_party else ""
    ax.set_title(
        f"Party Visibility: Mini vs Mini+Additions with CRW Correction{target_str}\n"
        f"({n_q} questions added simultaneously)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path = output_dir / f"{base}_grouped_bars.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Grouped bars: {path.name}")


def _plot_phase2_deltas(
    results: dict, output_dir: Path, base: str, target_party: str | None = None
):
    """Bar chart: Δ from mini baseline, with CRW reduction % annotations."""
    vis = results["visibility"]
    metric_results = results["metric_results"]
    orig = vis["mini_baseline"]

    conditions = [("Addition (no CRW)", vis["aug_baseline"], "#F44336")]

    crw_colors = ["#2196F3", "#64B5F6", "#FF9800", "#FFB74D", "#9C27B0"]
    for i, (ml, mr) in enumerate(metric_results.items()):
        color = crw_colors[i % len(crw_colors)]
        conditions.append((f"+ CRW ({ml})", mr["aug_crw_vis"], color))

    n_conditions = len(conditions)
    x = np.arange(len(MAJOR_PARTIES))
    width = 0.8 / n_conditions

    fig, ax = plt.subplots(figsize=(16, 7))

    for i, (label, vis_dict, color) in enumerate(conditions):
        offset = (i - n_conditions / 2 + 0.5) * width
        vals = [(vis_dict.get(p, 0) - orig.get(p, 0)) * 100 for p in MAJOR_PARTIES]
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MAJOR_PARTIES, fontsize=12)
    ax.set_ylabel("Δ Party Visibility (pp)", fontsize=12)

    # Annotate reduction % for target party
    focus = target_party if target_party else max(
        MAJOR_PARTIES,
        key=lambda p: abs(vis["aug_baseline"].get(p, 0) - orig.get(p, 0)),
    )
    if focus in MAJOR_PARTIES:
        p_idx = MAJOR_PARTIES.index(focus)
        d_attack = (vis["aug_baseline"].get(focus, 0) - orig.get(focus, 0)) * 100

        if abs(d_attack) > 0.001:
            for i, (ml, mr) in enumerate(metric_results.items()):
                d_crw = (mr["aug_crw_vis"].get(focus, 0) - mr["mini_crw_vis"].get(focus, 0)) * 100
                red_pct = (1 - abs(d_crw) / abs(d_attack)) * 100
                bar_i = i + 1  # offset by 1 since attack bar is at index 0
                offset = (bar_i - n_conditions / 2 + 0.5) * width
                bar_x = p_idx + offset
                ax.annotate(
                    f"{red_pct:.0f}%",
                    (bar_x, d_crw),
                    fontsize=7, ha="center",
                    va="bottom" if d_crw >= 0 else "top",
                    color=crw_colors[i % len(crw_colors)],
                    fontweight="bold",
                )

    n_q = len(results["q_ids"])
    target_str = f" targeting {target_party}" if target_party else ""
    ax.set_title(
        f"Change from Mini Baseline{target_str}\n"
        f"({n_q} questions added, % = CRW reduction for {focus})",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    path = output_dir / f"{base}_deltas.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Delta chart: {path.name}")


def _plot_crw_weight_comparison(results: dict, output_dir: Path, base: str):
    """Show CRW weights for added questions under each metric."""
    metric_results = results["metric_results"]
    q_ids = results["q_ids"]

    n_metrics = len(metric_results)
    n_questions = len(q_ids)

    x = np.arange(n_questions + 1)  # +1 for avg mini weight
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(14, 6))

    crw_colors = ["#2196F3", "#64B5F6", "#FF9800", "#FFB74D", "#9C27B0"]

    for i, (ml, mr) in enumerate(metric_results.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        vals = [mr["added_q_weights"].get(q, 0) or 0 for q in q_ids]
        vals.append(mr["avg_mini_weight"])
        color = crw_colors[i % len(crw_colors)]
        ax.bar(x + offset, vals, width, label=ml, color=color, alpha=0.85)

    labels = [f"Q{q}" for q in q_ids] + ["Avg mini\n(reference)"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("CRW Weight")
    ax.set_title(
        "CRW Weights for Added Questions by Distance Metric\n"
        "(lower weight = metric detects redundancy)"
    )
    ax.legend(fontsize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="weight=1.0")

    fig.tight_layout()
    path = output_dir / f"{base}_crw_weights.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> CRW weight comparison: {path.name}")


def _save_phase2_report(
    results: dict,
    output_dir: Path,
    base: str,
    n: int,
    target_party: str | None = None,
):
    """Save human-readable Phase 2 report."""
    vis = results["visibility"]
    metric_results = results["metric_results"]
    top_questions = results["top_questions"]
    q_ids = results["q_ids"]
    orig = vis["mini_baseline"]

    lines = [
        "=" * 80,
        "MINI VS MAXI PARTY IMPACT — PHASE 2 REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Target party: {target_party or 'N/A'}",
        f"Questions added: {len(q_ids)}",
        f"Top-k (n): {n}",
        "",
        "SELECTED QUESTIONS:",
        "-" * 80,
    ]

    for _, row in top_questions.iterrows():
        lines.append(
            f"  Q{int(row['question_id'])}: "
            f'"{str(row.get("question_text", ""))[:60]}"'
        )
        lines.append(
            f"    Category: {row.get('category', 'N/A')}, "
            f"mean_abs_r: {row.get('mean_abs_r', 0):.3f}, "
            f"max_abs_r: {row.get('max_abs_r', 0):.3f}"
        )
        if target_party:
            lines.append(
                f"    Δ({target_party}): "
                f"{row.get(f'delta_{target_party}', 0) * 100:+.2f}pp"
            )
    lines.append("")

    # Visibility table
    lines.extend([
        "PARTY VISIBILITY TABLE (%):",
        "-" * 80,
        f"{'Party':>10s} | {'Mini base':>10s} | {'+ Additions':>12s} | "
        + " | ".join(f"{ml:>16s}" for ml in metric_results.keys()),
        "-" * 80,
    ])

    for p in MAJOR_PARTIES:
        parts = [
            f"{p:>10s}",
            f"{orig.get(p, 0) * 100:>10.2f}",
            f"{vis['aug_baseline'].get(p, 0) * 100:>12.2f}",
        ]
        for ml, mr in metric_results.items():
            parts.append(f"{mr['aug_crw_vis'].get(p, 0) * 100:>16.2f}")
        lines.append(" | ".join(parts))

    lines.append("")

    # Reduction table
    focus = target_party or max(
        MAJOR_PARTIES,
        key=lambda p: abs(vis["aug_baseline"].get(p, 0) - orig.get(p, 0)),
    )
    d_attack = (vis["aug_baseline"].get(focus, 0) - orig.get(focus, 0)) * 100

    lines.extend([
        f"CRW REDUCTION FOR {focus}:",
        "-" * 80,
        f"  Attack delta (no CRW): {d_attack:+.2f}pp",
    ])

    for ml, mr in metric_results.items():
        d_crw = (mr["aug_crw_vis"].get(focus, 0) - mr["mini_crw_vis"].get(focus, 0)) * 100
        red = (1 - abs(d_crw) / abs(d_attack)) * 100 if abs(d_attack) > 0.001 else 0
        lines.append(f"  {ml}: {d_crw:+.2f}pp ({red:.0f}% reduced)")

    lines.append("")

    # CRW weights
    lines.extend([
        "CRW WEIGHTS FOR ADDED QUESTIONS:",
        "-" * 80,
    ])

    header_parts = [f"{'Question':>10s}"]
    for ml in metric_results.keys():
        header_parts.append(f"{ml:>16s}")
    lines.append(" | ".join(header_parts))

    for q in q_ids:
        parts = [f"Q{q:>9d}"]
        for ml, mr in metric_results.items():
            w = mr["added_q_weights"].get(q, None)
            parts.append(f"{w:>16.4f}" if w is not None else f"{'N/A':>16s}")
        lines.append(" | ".join(parts))

    # Avg mini weight
    parts = [f"{'Avg mini':>10s}"]
    for ml, mr in metric_results.items():
        parts.append(f"{mr['avg_mini_weight']:>16.4f}")
    lines.append(" | ".join(parts))

    lines.append("=" * 80)

    path = output_dir / f"{base}_report.txt"
    path.write_text("\n".join(lines))
    print(f"  -> Report: {path.name}")


# ---------------------------------------------------------------------------
# Compile: aggregate all 6 party Phase 2 results
# ---------------------------------------------------------------------------


def _run_compile(args, config):
    """Load all per-party Phase 2 CSVs and produce aggregated outputs."""
    selection_mode = getattr(args, "selection_mode", "delta")
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    sel_tag = "_corrwt" if selection_mode == "corr_weighted" else ""
    base = f"mini_maxi_compiled_{name}_{timestamp}{sel_tag}"

    if selection_mode == "corr_weighted":
        phase2_dir = RESULTS_DIR / "phase2_corr_weighted" / name
    else:
        phase2_dir = RESULTS_DIR / "phase2" / name

    # Find Phase 2 CSVs (one per party subdirectory)
    dfs = {}
    for party in MAJOR_PARTIES:
        party_dir = phase2_dir / party
        if not party_dir.exists():
            print(f"  WARNING: No Phase 2 directory for {party}, skipping")
            continue
        matches = sorted(party_dir.glob("mini_maxi_phase2_*.csv"))
        if not matches:
            print(f"  WARNING: No Phase 2 CSV for {party}, skipping")
            continue
        dfs[party] = pd.read_csv(matches[-1])
        print(f"  {party}: {matches[-1].name}")

    if len(dfs) < 2:
        print("ERROR: Need at least 2 party CSVs to compile.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Loaded {len(dfs)} party results, compiling...")

    compiled_dir = phase2_dir / "compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)

    # Identify metric columns from the first CSV
    sample_df = next(iter(dfs.values()))
    metric_labels = []
    for col in sample_df.columns:
        if col.startswith("aug_crw_"):
            label = col[len("aug_crw_"):]
            metric_labels.append(label)

    # Save compiled CSV with drift and corrected reduction columns
    compiled_rows = []
    for target in MAJOR_PARTIES:
        if target not in dfs:
            continue
        df = dfs[target]
        row = df[df["party"] == target].iloc[0]
        o = row["mini_baseline"]
        a = row["aug_baseline"]
        da = a - o
        r = {"party": target, "mini_baseline": o, "aug_baseline": a, "delta_attack": da}
        for ml in metric_labels:
            mc = row[f"mini_crw_{ml}"]
            ac = row[f"aug_crw_{ml}"]
            dc = ac - mc
            r[f"crw_drift_{ml}"] = mc - o
            r[f"delta_crw_{ml}"] = dc
            r[f"reduction_{ml}"] = (
                (1 - abs(dc) / abs(da)) * 100 if abs(da) > 1e-9 else 0
            )
        compiled_rows.append(r)
    compiled_csv = pd.DataFrame(compiled_rows)
    csv_path = compiled_dir / f"{base}.csv"
    compiled_csv.to_csv(csv_path, index=False)
    print(f"  -> Compiled CSV: {csv_path.name}")

    sns.set_theme(style="whitegrid")
    _compile_report(dfs, metric_labels, compiled_dir, base)
    _compile_heatmap(dfs, metric_labels, compiled_dir, base)
    _compile_bar_chart(dfs, metric_labels, compiled_dir, base)
    _compile_crw_comparison(dfs, metric_labels, compiled_dir, base)

    print(f"\n=== Compilation complete ===")


def _compile_report(
    dfs: dict, metric_labels: list, output_dir: Path, base: str,
):
    """Summary table: one row per target party showing addition gain and CRW reduction per metric."""
    lines = [
        "=" * 100,
        "MINI VS MAXI PARTY IMPACT — COMPILED REPORT (ALL PARTIES)",
        "=" * 100,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Parties compiled: {', '.join(dfs.keys())}",
        f"Metrics: {', '.join(ml.replace('_', ' ') for ml in metric_labels)}",
        "",
    ]

    # Per-metric summary tables
    for ml in metric_labels:
        display = ml.replace("_", " ").replace("α", "α=")
        lines.append("=" * 100)
        lines.append(f"METRIC: {display}")
        lines.append("=" * 100)
        lines.append(
            f"{'Target':>8s}  {'Mini base':>10s}  {'+ Additions':>12s}  "
            f"{'+ CRW':>10s}  {'Δ Attack':>9s}  {'Δ CRW':>9s}  {'Reduced':>8s}"
        )
        lines.append("-" * 80)

        gains, reductions = [], []
        for target in MAJOR_PARTIES:
            if target not in dfs:
                continue
            df = dfs[target]
            row = df[df["party"] == target].iloc[0]
            o = row["mini_baseline"] * 100
            oc = row[f"mini_crw_{ml}"] * 100
            a = row["aug_baseline"] * 100
            c = row[f"aug_crw_{ml}"] * 100
            da = a - o
            dc = c - oc
            red = (1 - abs(dc) / abs(da)) * 100 if abs(da) > 0.001 else 0
            gains.append(da)
            if abs(da) > 0.001:
                reductions.append(red)
            lines.append(
                f"{target:>8s}  {o:>9.2f}%  {a:>11.2f}%  "
                f"{c:>9.2f}%  {da:>+8.2f}  {dc:>+8.2f}  {red:>7.0f}%"
            )

        lines.append("-" * 80)
        lines.append(
            f"{'AVG':>8s}  {'':>10s}  {'':>12s}  "
            f"{'':>10s}  {np.mean(gains):>+8.2f}  {'':>9s}  "
            f"{np.mean(reductions):>7.0f}%"
        )
        lines.append("")

    # Cross-metric comparison: for each target party, show reduction % per metric
    lines.append("=" * 100)
    lines.append("CROSS-METRIC COMPARISON — CRW REDUCTION % PER METRIC")
    lines.append("=" * 100)

    header = f"{'Target':>8s}"
    for ml in metric_labels:
        display = ml.replace("_", " ")
        header += f"  {display:>20s}"
    lines.append(header)
    lines.append("-" * (10 + 22 * len(metric_labels)))

    for target in MAJOR_PARTIES:
        if target not in dfs:
            continue
        df = dfs[target]
        row = df[df["party"] == target].iloc[0]
        o = row["mini_baseline"] * 100
        a = row["aug_baseline"] * 100
        da = a - o
        row_str = f"{target:>8s}"
        for ml in metric_labels:
            oc = row[f"mini_crw_{ml}"] * 100
            c = row[f"aug_crw_{ml}"] * 100
            dc = c - oc
            red = (1 - abs(dc) / abs(da)) * 100 if abs(da) > 0.001 else 0
            row_str += f"  {red:>19.0f}%"
        lines.append(row_str)

    lines.append("")

    # Collateral damage table: when party X adds its top-5, how does it affect others?
    lines.append("=" * 100)
    lines.append("COLLATERAL DAMAGE — ADDITIONS BASELINE (Δpp for non-target parties)")
    lines.append("=" * 100)

    header = f"{'Adder':>8s}"
    for p in MAJOR_PARTIES:
        header += f"  {p:>8s}"
    lines.append(header)
    lines.append("-" * (10 + 10 * len(MAJOR_PARTIES)))

    for target in MAJOR_PARTIES:
        if target not in dfs:
            continue
        df = dfs[target]
        row_str = f"{target:>8s}"
        for p in MAJOR_PARTIES:
            row = df[df["party"] == p].iloc[0]
            delta = (row["aug_baseline"] - row["mini_baseline"]) * 100
            if p == target:
                row_str += f"  {'[' + f'{delta:+.2f}' + ']':>8s}"
            else:
                row_str += f"  {delta:>+7.2f} "
        lines.append(row_str)

    lines.append("")
    lines.append("=" * 100)

    path = output_dir / f"{base}_report.txt"
    path.write_text("\n".join(lines))
    print(f"  -> Compiled report: {path.name}")


def _compile_heatmap(
    dfs: dict, metric_labels: list, output_dir: Path, base: str,
):
    """Heatmap: rows = target (adder), columns = affected party, cells = Δpp.

    One heatmap for the baseline additions effect, plus one per metric showing
    the CRW-corrected effect.
    """
    targets = [p for p in MAJOR_PARTIES if p in dfs]

    configs = [("additions_baseline", "aug_baseline", "Additions (no CRW)")]
    for ml in metric_labels:
        display = ml.replace("_", " ")
        configs.append((ml, f"aug_crw_{ml}", f"CRW: {display}"))

    for suffix, col, title_label in configs:
        matrix = np.zeros((len(targets), len(MAJOR_PARTIES)))

        for i, target in enumerate(targets):
            df = dfs[target]
            for j, party in enumerate(MAJOR_PARTIES):
                row = df[df["party"] == party].iloc[0]
                matrix[i, j] = (row[col] - row["mini_baseline"]) * 100

        fig, ax = plt.subplots(figsize=(10, 7))

        vmax = max(abs(matrix.max()), abs(matrix.min()), 0.01)
        im = ax.imshow(
            matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
        )

        ax.set_xticks(range(len(MAJOR_PARTIES)))
        ax.set_xticklabels(MAJOR_PARTIES, fontsize=11)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets, fontsize=11)
        ax.set_xlabel("Affected Party", fontsize=12)
        ax.set_ylabel("Adder (Target Party)", fontsize=12)

        for i in range(len(targets)):
            for j in range(len(MAJOR_PARTIES)):
                val = matrix[i, j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                weight = "bold" if targets[i] == MAJOR_PARTIES[j] else "normal"
                ax.text(
                    j, i, f"{val:+.2f}",
                    ha="center", va="center",
                    fontsize=10, fontweight=weight, color=color,
                )

        # Draw box around diagonal cells
        for k, target in enumerate(targets):
            if target in MAJOR_PARTIES:
                j = MAJOR_PARTIES.index(target)
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, k - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=2,
                ))

        plt.colorbar(im, ax=ax, label="Δ Visibility (pp)", shrink=0.8)
        ax.set_title(
            f"Party Visibility Change — {title_label}",
            fontsize=13, fontweight="bold", pad=12,
        )
        fig.tight_layout()

        path = output_dir / f"{base}_heatmap_{suffix}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Heatmap ({title_label}): {path.name}")


def _compile_bar_chart(
    dfs: dict, metric_labels: list, output_dir: Path, base: str,
):
    """Bar chart: self-benefit (target party Δpp) from adding top-5 questions.

    Attack (no CRW) vs CRW-corrected, grouped by metric.
    """
    targets = [p for p in MAJOR_PARTIES if p in dfs]
    n_metrics = len(metric_labels)

    fig, ax = plt.subplots(figsize=(max(12, 2 * len(targets) * n_metrics), 7))

    x = np.arange(len(targets))
    total_bars = 1 + n_metrics  # 1 attack + N metric CRW bars
    width = 0.8 / total_bars

    # Attack bars (additions baseline)
    attack_gains = []
    for target in targets:
        df = dfs[target]
        row = df[df["party"] == target].iloc[0]
        attack_gains.append((row["aug_baseline"] - row["mini_baseline"]) * 100)

    ax.bar(
        x - (total_bars - 1) * width / 2,
        attack_gains, width,
        label="Additions (no CRW)", color="#F44336", alpha=0.85,
    )

    # CRW bars per metric
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    for mi, ml in enumerate(metric_labels):
        display = ml.replace("_", " ")
        crw_gains = []
        for target in targets:
            df = dfs[target]
            row = df[df["party"] == target].iloc[0]
            crw_gains.append((row[f"aug_crw_{ml}"] - row[f"mini_crw_{ml}"]) * 100)

        offset = x - (total_bars - 1) * width / 2 + (mi + 1) * width
        bars = ax.bar(
            offset, crw_gains, width,
            label=f"CRW: {display}",
            color=colors[mi % len(colors)], alpha=0.85,
        )

        # Annotate reduction %
        for i, (da, dc) in enumerate(zip(attack_gains, crw_gains)):
            if abs(da) > 0.01:
                red = (1 - abs(dc) / abs(da)) * 100
                ax.text(
                    offset[i], dc + 0.05 * np.sign(dc) if dc != 0 else 0.05,
                    f"{red:.0f}%",
                    ha="center", va="bottom" if dc >= 0 else "top",
                    fontsize=7, fontweight="bold",
                    color=colors[mi % len(colors)],
                )

    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=12)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(PARTY2COLOR.get(targets[i], "black"))
        label.set_fontweight("bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ Target Party Visibility (pp)", fontsize=12)
    ax.set_title(
        "Self-Benefit from Adding Top-5 Questions\n"
        "(each party adds its most beneficial full-only questions)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    path = output_dir / f"{base}_bar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Bar chart: {path.name}")


def _compile_crw_comparison(
    dfs: dict, metric_labels: list, output_dir: Path, base: str,
):
    """Grouped bar chart: CRW reduction % per metric per party.

    Shows which metric best corrects the addition effect for each party.
    """
    targets = [p for p in MAJOR_PARTIES if p in dfs]

    x = np.arange(len(targets))
    n_metrics = len(metric_labels)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(12, 2 * len(targets)), 7))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    for mi, ml in enumerate(metric_labels):
        display = ml.replace("_", " ")
        reductions = []
        for target in targets:
            df = dfs[target]
            row = df[df["party"] == target].iloc[0]
            o = row["mini_baseline"] * 100
            oc = row[f"mini_crw_{ml}"] * 100
            a = row["aug_baseline"] * 100
            c = row[f"aug_crw_{ml}"] * 100
            da = a - o
            dc = c - oc
            red = (1 - abs(dc) / abs(da)) * 100 if abs(da) > 0.001 else 0
            reductions.append(red)

        offset = x - (n_metrics - 1) * width / 2 + mi * width
        bars = ax.bar(
            offset, reductions, width,
            label=display,
            color=colors[mi % len(colors)], alpha=0.85,
        )

        for i, (bar, red) in enumerate(zip(bars, reductions)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                red + 1,
                f"{red:.0f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=12)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(PARTY2COLOR.get(targets[i], "black"))
        label.set_fontweight("bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("CRW Reduction (%)", fontsize=12)
    ax.set_title(
        "CRW Correction Effectiveness by Metric\n"
        "(% of addition effect reversed by CRW)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    path = output_dir / f"{base}_crw_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> CRW comparison: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()
    config = load_config(Path(args.config))
    n = _resolve_n(config, args.n)
    name = _get_clean_name(config)

    selection_mode = getattr(args, "selection_mode", "delta")
    sel_str = f", selection={selection_mode}" if selection_mode != "delta" else ""
    print(f"\n=== Mini vs Maxi Party Impact Analysis ({args.mode} mode{sel_str}) ===")
    print(f"  Config : {name}")
    print(f"  Top-k (n): {n}")
    print(f"  Major parties: {', '.join(MAJOR_PARTIES)}")

    if args.mode == "sweep":
        _run_sweep(args, config, n)
    elif args.mode == "worker":
        _run_worker(args, config, n)
    elif args.mode == "collect":
        _run_collect(args, config, n)
    elif args.mode == "phase2":
        _run_phase2(args, config, n)
    elif args.mode == "compile":
        _run_compile(args, config)


if __name__ == "__main__":
    main()
