"""
Party visibility impact sweep: identifies which questions, when cloned,
cause the largest shift in party visibility within voter recommendations.

Combines the party visibility computation from party_impact_main.py with
the enrichment pipeline from question_impact_main.py (variance, NaN rates,
answer correlations) and adds topic-party association analysis.

Supports three modes:
    - sweep (default): Run all questions sequentially in one process.
    - worker:  Compute a single question (by --task-id index). For SLURM job arrays.
    - collect: Read per-question worker CSVs from --sweep-dir, aggregate, enrich,
               and generate compiled + per-party outputs.

Usage:
    # Sequential sweep:
    python -m party_visibility_impact_main \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py

    # Worker (one question, for SLURM array):
    python -m party_visibility_impact_main --mode worker --task-id 3 \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --sweep-dir /path/to/sweep_dir

    # Collect (aggregate workers + all plots):
    python -m party_visibility_impact_main --mode collect \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --sweep-dir /path/to/sweep_dir

Outputs (saved to experiment_results/question_impact/party_visibility_impact/):
    - full_results_*.csv                  -- per-question metrics for all parties
    - compiled/                           -- cross-party analysis (heatmap, boxplot, etc.)
    - per_party/{SP,Green,...}/           -- standalone per-party analysis
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
from scipy.stats import kruskal, spearmanr

from clone_pipeline.applicator import apply_specs
from clone_pipeline.spec import CloneSpec
from configs import base_constants as default_config
from main import load_config
from party_impact_main import (
    MAJOR_PARTIES,
    N_CLONES,
    _build_candidate_party_map,
    compute_party_visibility,
)
from rsfp.constants import PARTY2COLOR, QUESTION_ID2CATEGORY
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine

RESULTS_DIR = default_config.RESULTS_DIR / "question_impact" / "party_visibility_impact"

# Topic color palette (14 topics)
TOPIC_COLORS = {
    "Welfare state & family": "#E53935",
    "Health": "#D81B60",
    "Education": "#8E24AA",
    "Immigration & integration": "#5E35B1",
    "Society & ethics": "#3949AB",
    "Finances & taxes": "#1E88E5",
    "Economy & labour": "#039BE5",
    "Energy & transport": "#00ACC1",
    "Nature conservation": "#00897B",
    "Democracy, Media & Digitalization": "#43A047",
    "Security & military": "#7CB342",
    "Foreign trade & foreign policy": "#C0CA33",
    "Values": "#FDD835",
    "Federal budget": "#FFB300",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Party visibility impact sweep: which questions shift party visibility most?"
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
        help="Override top-k for party visibility (default: derived from config)",
    )
    parser.add_argument(
        "--n-clones", type=int, default=None,
        help=f"Number of identical clones per question (default: {N_CLONES})",
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

    # Build candidate ID -> party mapping
    candidate_party_map = _build_candidate_party_map(dataset["candidates"])

    # Get sorted question IDs (excluding clones) for deterministic task-id mapping
    questions_df = dataset["questions"]
    question_ids = sorted(
        questions_df.loc[questions_df["ID_question"] < 9_000_000, "ID_question"].tolist()
    )

    print(f"\n  Questions: {len(question_ids)}")

    # Pre-compute NaN rates and variance
    df_voters = dataset["voters"]
    df_candidates = dataset["candidates"]
    nan_info = {}
    var_info = {}
    for q_id in question_ids:
        ans_col = f"answer_{q_id}"
        voter_nan = df_voters[ans_col].isna().mean() if ans_col in df_voters.columns else 1.0
        cand_nan = df_candidates[ans_col].isna().mean() if ans_col in df_candidates.columns else 1.0
        nan_info[q_id] = {"voter_nan_pct": voter_nan, "candidate_nan_pct": cand_nan}

        cand_var = df_candidates[ans_col].var() if ans_col in df_candidates.columns else np.nan
        voter_var = df_voters[ans_col].var() if ans_col in df_voters.columns else np.nan
        combined = cand_var * voter_var if pd.notna(cand_var) and pd.notna(voter_var) else np.nan
        var_info[q_id] = {
            "candidate_var": cand_var, "voter_var": voter_var, "combined_var": combined,
        }

    return {
        "dataset": dataset,
        "base_recs": base_recs,
        "candidate_party_map": candidate_party_map,
        "question_ids": question_ids,
        "questions_df": questions_df,
        "nan_info": nan_info,
        "var_info": var_info,
    }


# ---------------------------------------------------------------------------
# Per-question computation
# ---------------------------------------------------------------------------


def _compute_question_pvi(
    q_id: int,
    config,
    pipeline: dict,
    n: int,
    n_clones: int,
) -> dict:
    """Clone question n_clones times identical, compute party visibility delta."""
    dataset = pipeline["dataset"]
    candidate_party_map = pipeline["candidate_party_map"]
    questions_df = pipeline["questions_df"]
    text_col = _get_question_text_col(questions_df)

    # 1. Base party visibility
    base_visibility = compute_party_visibility(
        pipeline["base_recs"], candidate_party_map, n
    )

    # 2. Clone in-memory
    spec = CloneSpec(source_q_id=q_id, clone_type="identical", n_clones=n_clones)
    cloned_data = apply_specs(
        specs=[spec],
        dataframes={
            "questions": dataset["questions"],
            "voters": dataset["voters"],
            "candidates": dataset["candidates"],
        },
    )

    # 3. Cloned baseline recs
    cloned_engine = RecommendationEngine(config=config, data_map=cloned_data)
    cloned_recs = cloned_engine.run_baseline()
    sys.stdout.flush()

    # 4. Cloned party visibility
    cloned_visibility = compute_party_visibility(cloned_recs, candidate_party_map, n)

    # 5. Build result row
    q_row = questions_df.loc[questions_df["ID_question"] == q_id].iloc[0]
    q_text = q_row[text_col]
    category = q_row.get("_category", "")
    question_order = sorted(pipeline["question_ids"]).index(q_id) + 1

    row = {
        "question_id": q_id,
        "question_text": q_text,
        "question_order": question_order,
        "category": category,
    }

    # Store base, cloned, and delta for all parties
    all_parties = sorted(
        set(list(base_visibility.keys()) + list(cloned_visibility.keys()))
    )
    for party in all_parties:
        base_v = base_visibility.get(party, 0.0)
        cloned_v = cloned_visibility.get(party, 0.0)
        row[f"base_{party}"] = base_v
        row[f"cloned_{party}"] = cloned_v
        row[f"delta_{party}"] = cloned_v - base_v

    # Max absolute delta across major parties
    major_deltas = {p: abs(row.get(f"delta_{p}", 0.0)) for p in MAJOR_PARTIES}
    row["max_abs_delta"] = max(major_deltas.values()) if major_deltas else 0.0
    row["max_delta_party"] = (
        max(major_deltas, key=major_deltas.get) if major_deltas else ""
    )

    # Max positive delta (which party benefits most)
    positive_deltas = {p: row.get(f"delta_{p}", 0.0) for p in MAJOR_PARTIES}
    row["max_positive_delta"] = max(positive_deltas.values())
    row["max_positive_party"] = max(positive_deltas, key=positive_deltas.get)

    # Statistical features
    row.update(pipeline["nan_info"].get(q_id, {}))
    row.update(pipeline["var_info"].get(q_id, {}))

    return row


# ---------------------------------------------------------------------------
# Mode: sweep (sequential)
# ---------------------------------------------------------------------------


def _run_sweep(args, config, n: int, n_clones: int):
    pipeline = _setup_pipeline(config)
    question_ids = pipeline["question_ids"]

    rows = []
    for i, q_id in enumerate(question_ids):
        print(f"\n--- Question {q_id} ({i + 1}/{len(question_ids)}) ---")
        row = _compute_question_pvi(q_id, config, pipeline, n, n_clones)
        rows.append(row)
        print(
            f"  Max delta: {row['max_abs_delta']:.4f} "
            f"({row['max_delta_party']}, "
            f"best: {row['max_positive_party']} {row['max_positive_delta']:+.4f})"
        )

    sweep_df = pd.DataFrame(rows)

    # Load dataset for correlation analysis (already in pipeline)
    name = _get_clean_name(config)
    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_all_outputs(sweep_df, config, n, n_clones, output_dir, pipeline["dataset"])

    print("\n=== Party Visibility Impact Sweep Complete ===")


# ---------------------------------------------------------------------------
# Mode: worker (single question for SLURM array)
# ---------------------------------------------------------------------------


def _run_worker(args, config, n: int, n_clones: int):
    task_id = args.task_id
    if task_id is None:
        print("ERROR: --task-id is required in worker mode", file=sys.stderr)
        sys.exit(1)

    name = _get_clean_name(config)
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / name / "workers"
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
    out_path = sweep_dir / f"pvi_worker_{task_id:03d}_q{q_id}.csv"

    # Skip if already computed
    if out_path.exists():
        print(f"  [SKIP] Worker CSV already exists: {out_path}")
        return

    print(
        f"\n=== Worker: question {q_id} (task {task_id}/{len(question_ids) - 1}) ==="
    )

    row = _compute_question_pvi(q_id, config, pipeline, n, n_clones)

    worker_df = pd.DataFrame([row])
    worker_df.to_csv(out_path, index=False)

    print(f"\n  -> Worker CSV: {out_path}")
    print(f"  Max delta: {row['max_abs_delta']:.4f} ({row['max_delta_party']})")
    print("\n=== Worker Complete ===")


# ---------------------------------------------------------------------------
# Mode: collect (aggregate + enrich + plot)
# ---------------------------------------------------------------------------


def _run_collect(args, config, n: int, n_clones: int):
    name = _get_clean_name(config)
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / name / "workers"

    worker_files = sorted(sweep_dir.glob("pvi_worker_*.csv"))
    if not worker_files:
        print(f"ERROR: No worker CSVs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Collect: reading {len(worker_files)} worker CSVs from {sweep_dir} ===")

    dfs = [pd.read_csv(f) for f in worker_files]
    combined = pd.concat(dfs, ignore_index=True)

    # Load dataset for correlation analysis
    print("\n--- Loading dataset for correlation analysis ---")
    dataset = load_dataset(config)

    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_all_outputs(combined, config, n, n_clones, output_dir, dataset)

    print("\n=== Collect Complete ===")


# ---------------------------------------------------------------------------
# Answer correlation (reused from question_impact_main pattern)
# ---------------------------------------------------------------------------


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
# All output generation
# ---------------------------------------------------------------------------


def _save_all_outputs(
    df: pd.DataFrame,
    config,
    n: int,
    n_clones: int,
    output_dir: Path,
    dataset: dict,
):
    """Enrich with correlations, then generate compiled + per-party outputs."""
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = f"pvi_{name}_{timestamp}"

    # --- Answer correlation analysis ---
    print("\n--- Computing answer correlation analysis ---")
    df_voters = dataset["voters"]
    df_candidates = dataset["candidates"]
    question_ids = sorted(df["question_id"].unique().tolist())

    voter_corr, cand_corr, voter_avg_corr, cand_avg_corr = _compute_answer_correlations(
        df_voters, df_candidates, question_ids
    )

    df["voter_avg_abs_corr"] = df["question_id"].map(voter_avg_corr)
    df["candidate_avg_abs_corr"] = df["question_id"].map(cand_avg_corr)

    # Sort by max absolute delta
    df = df.sort_values("max_abs_delta", ascending=False).reset_index(drop=True)

    # --- Save full CSV ---
    csv_path = output_dir / f"{base}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> CSV: {csv_path.name}")

    # --- Compiled outputs ---
    sns.set_theme(style="whitegrid")
    compiled_dir = output_dir / "compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)

    _plot_topic_party_heatmap(df, compiled_dir, base, n_clones)
    _plot_per_party_ranking_grid(df, compiled_dir, base, n_clones)
    _plot_predictor_analysis(df, compiled_dir, base, n_clones, target_col="max_abs_delta", label="Max |Δ| Visibility")
    _plot_corr_matrices(voter_corr, cand_corr, question_ids, compiled_dir, base)
    _plot_topic_party_boxplot(df, compiled_dir, base, n_clones)
    _save_compiled_report(df, compiled_dir, base, n, n_clones)

    # --- Per-party outputs ---
    for party in MAJOR_PARTIES:
        delta_col = f"delta_{party}"
        if delta_col not in df.columns:
            continue

        party_dir = output_dir / "per_party" / party
        party_dir.mkdir(parents=True, exist_ok=True)

        _plot_party_ranking(df, party, party_dir, base, n_clones)
        _plot_party_topic_breakdown(df, party, party_dir, base, n_clones)
        _plot_predictor_analysis(
            df, party_dir, base, n_clones,
            target_col=delta_col,
            label=f"Δ Visibility ({party})",
            suffix=f"_{party}",
        )
        _save_party_report(df, party, party_dir, base, n, n_clones)


# ---------------------------------------------------------------------------
# Compiled plots
# ---------------------------------------------------------------------------


def _plot_topic_party_heatmap(df: pd.DataFrame, output_dir: Path, base: str, n_clones: int):
    """Heatmap: 14 topics x 6 major parties, cell = mean delta (pp)."""
    if "category" not in df.columns:
        print("  [SKIP] No category column for topic-party heatmap")
        return

    delta_cols = [f"delta_{p}" for p in MAJOR_PARTIES if f"delta_{p}" in df.columns]
    parties = [p for p in MAJOR_PARTIES if f"delta_{p}" in df.columns]

    # Group by category, compute mean delta per party
    topic_means = df.groupby("category")[delta_cols].mean()
    topic_counts = df.groupby("category").size()

    # Sort by total absolute impact
    topic_means["_total_abs"] = topic_means.abs().sum(axis=1)
    topic_means = topic_means.sort_values("_total_abs", ascending=False)
    topic_means = topic_means.drop(columns=["_total_abs"])

    # Row labels with question count
    row_labels = [f"{t} ({topic_counts.get(t, 0)})" for t in topic_means.index]

    fig, ax = plt.subplots(figsize=(10, max(6, len(topic_means) * 0.45)))

    sns.heatmap(
        topic_means.values * 100,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        xticklabels=parties,
        yticklabels=row_labels,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Mean Δ Party Visibility (pp)"},
    )
    ax.set_title(
        f"Mean Party Visibility Shift by Topic ({n_clones}x identical clones)\n"
        f"Red = party gains visibility, Blue = party loses visibility"
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=10)
    fig.tight_layout()

    path = output_dir / f"{base}_topic_party_heatmap.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Topic-party heatmap: {path.name}")


def _plot_per_party_ranking_grid(df: pd.DataFrame, output_dir: Path, base: str, n_clones: int):
    """2x3 grid: for each major party, top-10 questions that benefit it most."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    for ax, party in zip(axes.flat, MAJOR_PARTIES):
        delta_col = f"delta_{party}"
        if delta_col not in df.columns:
            ax.set_visible(False)
            continue

        top10 = df.nlargest(10, delta_col)
        labels = [
            f"Q{int(r['question_id'])}  {str(r['question_text'])[:30]}"
            for _, r in top10.iterrows()
        ]
        values = top10[delta_col].values * 100

        color = PARTY2COLOR.get(party, "#888888")
        ax.barh(range(len(labels)), values, color=color, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Δ Visibility (pp)")
        ax.set_title(f"{party}", fontsize=13, fontweight="bold", color=color)
        ax.axvline(0, color="black", linewidth=0.5)

    for i in range(len(MAJOR_PARTIES), 6):
        axes.flat[i].set_visible(False)

    fig.suptitle(
        f"Top-10 Questions Benefiting Each Party ({n_clones}x identical clones)",
        fontsize=14,
    )
    fig.tight_layout()

    path = output_dir / f"{base}_per_party_ranking_grid.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Per-party ranking grid: {path.name}")


def _plot_predictor_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    base: str,
    n_clones: int,
    target_col: str = "max_abs_delta",
    label: str = "Max |Δ| Visibility",
    suffix: str = "",
):
    """2x3 scatter: predictors vs a target delta column."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    scatter_configs = [
        ("combined_var", "Combined Variance", axes[0, 0]),
        ("voter_var", "Voter Variance", axes[0, 1]),
        ("candidate_var", "Candidate Variance", axes[0, 2]),
        ("voter_nan_pct", "Voter NaN %", axes[1, 0]),
        ("question_order", "Question Position", axes[1, 1]),
        ("voter_avg_abs_corr", "Question Redundancy\n(mean |Pearson r|)", axes[1, 2]),
    ]

    for col, col_label, ax in scatter_configs:
        if col not in df.columns or df[col].isna().all():
            ax.set_visible(False)
            continue

        x = df[col].values
        y = df[target_col].values * 100  # percentage points
        mask = ~(np.isnan(x) | np.isnan(y))

        ax.scatter(x[mask], y[mask], alpha=0.6, s=40, color="#1565C0")

        if mask.sum() >= 3:
            rho, p = spearmanr(x[mask], y[mask])
            ax.set_title(f"{col_label}\n(Spearman r={rho:.3f}, p={p:.3f})", fontsize=10)
        else:
            ax.set_title(col_label, fontsize=10)

        ax.set_xlabel(col_label)
        ax.set_ylabel(f"{label} (pp)")

        # Annotate top-3
        top3 = df.nlargest(3, target_col)
        for _, row in top3.iterrows():
            if pd.notna(row.get(col)):
                ax.annotate(
                    f"Q{int(row['question_id'])}",
                    (row[col], row[target_col] * 100),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points",
                )

    fig.suptitle(
        f"Predictors of Party Visibility Impact ({n_clones}x identical clones)\n"
        f"Y-axis: {label} (pp)",
        fontsize=12,
    )
    fig.tight_layout()

    path = output_dir / f"{base}_predictor_analysis{suffix}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Predictor analysis{suffix}: {path.name}")


def _plot_corr_matrices(
    voter_corr: pd.DataFrame,
    cand_corr: pd.DataFrame,
    question_ids: list[int],
    output_dir: Path,
    base: str,
):
    """Heatmaps of voter and candidate answer correlation matrices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

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


def _plot_topic_party_boxplot(df: pd.DataFrame, output_dir: Path, base: str, n_clones: int):
    """2x3 grid: boxplot of delta_{party} by topic for each party."""
    if "category" not in df.columns:
        print("  [SKIP] No category column for topic-party boxplot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    for ax, party in zip(axes.flat, MAJOR_PARTIES):
        delta_col = f"delta_{party}"
        if delta_col not in df.columns:
            ax.set_visible(False)
            continue

        # Sort topics by median delta for this party
        topic_order = (
            df.groupby("category")[delta_col]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

        plot_df = df[["category", delta_col]].copy()
        plot_df[delta_col] = plot_df[delta_col] * 100  # pp

        color = PARTY2COLOR.get(party, "#888888")

        sns.boxplot(
            data=plot_df, y="category", x=delta_col,
            order=topic_order, ax=ax,
            color=color, alpha=0.4, width=0.6,
            fliersize=0,
        )
        sns.stripplot(
            data=plot_df, y="category", x=delta_col,
            order=topic_order, ax=ax,
            color=color, alpha=0.7, size=5, jitter=0.15,
        )

        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Δ Visibility (pp)")
        ax.set_ylabel("")
        ax.set_title(f"{party}", fontsize=13, fontweight="bold", color=color)
        ax.tick_params(axis="y", labelsize=7)

    for i in range(len(MAJOR_PARTIES), 6):
        axes.flat[i].set_visible(False)

    fig.suptitle(
        f"Party Visibility Delta Distribution by Topic ({n_clones}x identical clones)",
        fontsize=14,
    )
    fig.tight_layout()

    path = output_dir / f"{base}_topic_party_boxplot.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Topic-party boxplot: {path.name}")


# ---------------------------------------------------------------------------
# Per-party plots
# ---------------------------------------------------------------------------


def _plot_party_ranking(
    df: pd.DataFrame,
    party: str,
    output_dir: Path,
    base: str,
    n_clones: int,
):
    """Full 75-question horizontal bar chart for a single party, colored by topic."""
    delta_col = f"delta_{party}"
    df_sorted = df.sort_values(delta_col, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, max(10, len(df_sorted) * 0.3)))

    labels = [
        f"Q{int(r['question_id'])}  {str(r['question_text'])[:50]}"
        for _, r in df_sorted.iterrows()
    ]
    values = df_sorted[delta_col].values * 100

    # Color by topic
    colors = [
        TOPIC_COLORS.get(str(r.get("category", "")), "#888888")
        for _, r in df_sorted.iterrows()
    ]

    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Δ Visibility (pp)")
    ax.axvline(0, color="black", linewidth=0.8)

    party_color = PARTY2COLOR.get(party, "#888888")
    ax.set_title(
        f"{party} — Question Ranking by Visibility Impact ({n_clones}x identical clones)\n"
        f"Bars colored by question topic",
        fontsize=12, color=party_color,
    )

    # Legend for topics
    from matplotlib.patches import Patch
    categories_present = df_sorted["category"].unique() if "category" in df_sorted.columns else []
    legend_handles = [
        Patch(facecolor=TOPIC_COLORS.get(c, "#888888"), label=c, alpha=0.8)
        for c in sorted(categories_present)
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles, loc="lower right", fontsize=6,
            title="Topic", title_fontsize=7, ncol=2,
        )

    fig.tight_layout()

    path = output_dir / f"{base}_ranking_{party}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> {party} ranking: {path.name}")


def _plot_party_topic_breakdown(
    df: pd.DataFrame,
    party: str,
    output_dir: Path,
    base: str,
    n_clones: int,
):
    """Bar chart of mean delta by topic for a specific party, with individual dots."""
    delta_col = f"delta_{party}"
    if "category" not in df.columns:
        return

    topic_stats = df.groupby("category")[delta_col].agg(["mean", "median", "count"])
    topic_stats = topic_stats.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(6, len(topic_stats) * 0.5)))

    party_color = PARTY2COLOR.get(party, "#888888")
    y_pos = range(len(topic_stats))

    # Mean bars
    ax.barh(
        y_pos, topic_stats["mean"].values * 100,
        color=party_color, alpha=0.4, height=0.6, label="Mean Δ",
    )

    # Individual question dots
    for i, topic in enumerate(topic_stats.index):
        topic_vals = df.loc[df["category"] == topic, delta_col].values * 100
        ax.scatter(
            topic_vals, [i] * len(topic_vals),
            color=party_color, alpha=0.7, s=30, zorder=5,
        )

    labels = [f"{t} ({int(topic_stats.loc[t, 'count'])})" for t in topic_stats.index]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ Visibility (pp)")
    ax.axvline(0, color="black", linewidth=0.8)

    # Kruskal-Wallis test
    groups = [
        group[delta_col].values
        for _, group in df.groupby("category")
        if len(group) >= 2
    ]
    if len(groups) >= 2:
        try:
            h_stat, p_val = kruskal(*groups)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            ax.set_title(
                f"{party} — Visibility Impact by Topic ({n_clones}x identical clones)\n"
                f"Kruskal-Wallis H={h_stat:.2f}, p={p_val:.4f} ({sig})",
                fontsize=12, color=party_color,
            )
        except ValueError:
            ax.set_title(
                f"{party} — Visibility Impact by Topic ({n_clones}x identical clones)",
                fontsize=12, color=party_color,
            )
    else:
        ax.set_title(
            f"{party} — Visibility Impact by Topic ({n_clones}x identical clones)",
            fontsize=12, color=party_color,
        )

    fig.tight_layout()

    path = output_dir / f"{base}_topic_breakdown_{party}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> {party} topic breakdown: {path.name}")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _save_compiled_report(
    df: pd.DataFrame,
    output_dir: Path,
    base: str,
    n: int,
    n_clones: int,
    top_n: int = 10,
):
    """Cross-party compiled report."""
    lines = [
        "=" * 80,
        "PARTY VISIBILITY IMPACT — COMPILED REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Clone type: identical, n_clones: {n_clones}",
        f"Top-k (n): {n}",
        f"Questions tested: {len(df)}",
        f"Major parties: {', '.join(MAJOR_PARTIES)}",
    ]

    # --- Top-10 by max party impact ---
    lines.extend([
        "",
        f"TOP {top_n} QUESTIONS BY MAX PARTY IMPACT (abs Δ visibility):",
        "-" * 80,
    ])

    for i, (_, row) in enumerate(df.head(top_n).iterrows()):
        lines.append(
            f"  {i + 1}. Q{int(row['question_id'])} — "
            f"max |Δ| = {row['max_abs_delta'] * 100:.2f}pp "
            f"({row['max_delta_party']})"
        )
        cat = row.get("category", "")
        lines.append(f'     [{cat}] "{str(row["question_text"])[:65]}"')
        for p in MAJOR_PARTIES:
            d = row.get(f"delta_{p}", 0)
            if abs(d) > 0.0001:
                lines.append(f"       {p:>8s}: {d * 100:+.2f}pp")
        lines.append("")

    # --- Per-party top-5 summary ---
    lines.extend(["=" * 80, "PER-PARTY TOP-5 BENEFITING QUESTIONS:", "-" * 80])
    for p in MAJOR_PARTIES:
        delta_col = f"delta_{p}"
        if delta_col not in df.columns:
            continue
        top5 = df.nlargest(5, delta_col)
        lines.append(f"\n  {p}:")
        for j, (_, row) in enumerate(top5.iterrows()):
            lines.append(
                f"    {j + 1}. Q{int(row['question_id'])} "
                f"({row[delta_col] * 100:+.2f}pp) [{row.get('category', '')}] "
                f'"{str(row["question_text"])[:45]}"'
            )

    # --- Topic-party association (Kruskal-Wallis) ---
    lines.extend(["", "=" * 80, "TOPIC-PARTY ASSOCIATION (Kruskal-Wallis):", "-" * 80])
    lines.append(
        f"  {'Party':>8s}  {'H-stat':>8s}  {'p-value':>8s}  {'Sig':>4s}  "
        f"{'Top Topic':>30s}  {'Bottom Topic':>30s}"
    )

    if "category" in df.columns:
        for p in MAJOR_PARTIES:
            delta_col = f"delta_{p}"
            if delta_col not in df.columns:
                continue

            groups = [
                group[delta_col].values
                for _, group in df.groupby("category")
                if len(group) >= 2
            ]

            topic_means = df.groupby("category")[delta_col].mean()
            top_topic = topic_means.idxmax()
            bottom_topic = topic_means.idxmin()

            if len(groups) >= 2:
                try:
                    h_stat, p_val = kruskal(*groups)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                    lines.append(
                        f"  {p:>8s}  {h_stat:>8.2f}  {p_val:>8.4f}  {sig:>4s}  "
                        f"{top_topic:>30s}  {bottom_topic:>30s}"
                    )
                except ValueError:
                    lines.append(f"  {p:>8s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>4s}")

    # --- Predictor correlations ---
    lines.extend(["", "=" * 80, "PREDICTOR CORRELATIONS (Spearman vs max_abs_delta):", "-" * 80])
    predictors = [
        "combined_var", "voter_var", "candidate_var",
        "voter_nan_pct", "question_order", "voter_avg_abs_corr",
    ]
    for pred in predictors:
        if pred not in df.columns or df[pred].isna().all():
            continue
        mask = ~(df[pred].isna() | df["max_abs_delta"].isna())
        if mask.sum() >= 3:
            rho, p = spearmanr(df.loc[mask, pred], df.loc[mask, "max_abs_delta"])
            lines.append(f"  {pred:<25s}  r={rho:+.3f}  p={p:.4f}")

    # --- Full statistics ---
    lines.extend([
        "",
        "=" * 80,
        "FULL STATISTICS:",
        f"  Mean max_abs_delta:   {df['max_abs_delta'].mean() * 100:.2f}pp",
        f"  Median max_abs_delta: {df['max_abs_delta'].median() * 100:.2f}pp",
        f"  Max max_abs_delta:    {df['max_abs_delta'].max() * 100:.2f}pp",
        f"  Min max_abs_delta:    {df['max_abs_delta'].min() * 100:.2f}pp",
        "=" * 80,
    ])

    path = output_dir / f"{base}_report.txt"
    path.write_text("\n".join(lines))
    print(f"  -> Compiled report: {path.name}")


def _save_party_report(
    df: pd.DataFrame,
    party: str,
    output_dir: Path,
    base: str,
    n: int,
    n_clones: int,
):
    """Full standalone per-party report."""
    delta_col = f"delta_{party}"
    df_sorted = df.sort_values(delta_col, ascending=False).reset_index(drop=True)

    lines = [
        "=" * 80,
        f"PARTY VISIBILITY IMPACT — {party.upper()} REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Clone type: identical, n_clones: {n_clones}",
        f"Top-k (n): {n}",
        f"Questions tested: {len(df)}",
    ]

    # --- Top-10 benefiting questions ---
    lines.extend([
        "",
        f"TOP 10 QUESTIONS BENEFITING {party.upper()}:",
        "-" * 80,
    ])
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows()):
        lines.append(
            f"  {i + 1}. Q{int(row['question_id'])} — "
            f"Δ = {row[delta_col] * 100:+.2f}pp"
        )
        cat = row.get("category", "")
        lines.append(f'     [{cat}] "{str(row["question_text"])[:65]}"')
        # Show other party deltas for context
        others = []
        for p in MAJOR_PARTIES:
            if p != party:
                d = row.get(f"delta_{p}", 0)
                if abs(d) > 0.0001:
                    others.append(f"{p}: {d * 100:+.2f}pp")
        if others:
            lines.append(f"     Other parties: {', '.join(others)}")
        lines.append("")

    # --- Bottom-5 (most harmful) ---
    lines.extend([
        f"BOTTOM 5 QUESTIONS (MOST HARMFUL TO {party.upper()}):",
        "-" * 80,
    ])
    bottom5 = df_sorted.tail(5).iloc[::-1]
    for i, (_, row) in enumerate(bottom5.iterrows()):
        lines.append(
            f"  {i + 1}. Q{int(row['question_id'])} — "
            f"Δ = {row[delta_col] * 100:+.2f}pp"
        )
        cat = row.get("category", "")
        lines.append(f'     [{cat}] "{str(row["question_text"])[:65]}"')
        lines.append("")

    # --- Topic breakdown ---
    if "category" in df.columns:
        topic_stats = df.groupby("category")[delta_col].agg(["mean", "median", "std", "count"])
        topic_stats = topic_stats.sort_values("mean", ascending=False)

        lines.extend([
            "=" * 80,
            f"TOPIC BREAKDOWN FOR {party.upper()}:",
            "-" * 80,
            f"  {'Topic':<40s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}  {'N':>4s}",
            "-" * 80,
        ])
        for topic, stats in topic_stats.iterrows():
            lines.append(
                f"  {topic:<40s}  {stats['mean'] * 100:>+7.2f}  "
                f"{stats['median'] * 100:>+7.2f}  "
                f"{stats['std'] * 100:>7.2f}  {int(stats['count']):>4d}"
            )

        # Kruskal-Wallis
        groups = [
            group[delta_col].values
            for _, group in df.groupby("category")
            if len(group) >= 2
        ]
        if len(groups) >= 2:
            try:
                h_stat, p_val = kruskal(*groups)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                lines.extend([
                    "",
                    f"  Kruskal-Wallis test: H={h_stat:.2f}, p={p_val:.4f} ({sig})",
                    f"  {'Significant' if p_val < 0.05 else 'Not significant'}: "
                    f"topic {'does' if p_val < 0.05 else 'does not'} significantly predict "
                    f"visibility delta for {party}.",
                ])
            except ValueError:
                pass

    # --- Predictor correlations ---
    lines.extend(["", "=" * 80, f"PREDICTOR CORRELATIONS (Spearman vs delta_{party}):", "-" * 80])
    predictors = [
        "combined_var", "voter_var", "candidate_var",
        "voter_nan_pct", "question_order", "voter_avg_abs_corr", "candidate_avg_abs_corr",
    ]
    for pred in predictors:
        if pred not in df.columns or df[pred].isna().all():
            continue
        mask = ~(df[pred].isna() | df[delta_col].isna())
        if mask.sum() >= 3:
            rho, p = spearmanr(df.loc[mask, pred], df.loc[mask, delta_col])
            lines.append(f"  {pred:<25s}  r={rho:+.3f}  p={p:.4f}")

    # --- Summary statistics ---
    lines.extend([
        "",
        "=" * 80,
        f"SUMMARY STATISTICS (delta_{party}):",
        f"  Mean:   {df[delta_col].mean() * 100:+.2f}pp",
        f"  Median: {df[delta_col].median() * 100:+.2f}pp",
        f"  Std:    {df[delta_col].std() * 100:.2f}pp",
        f"  Max:    {df[delta_col].max() * 100:+.2f}pp",
        f"  Min:    {df[delta_col].min() * 100:+.2f}pp",
        "=" * 80,
    ])

    path = output_dir / f"{base}_report_{party}.txt"
    path.write_text("\n".join(lines))
    print(f"  -> {party} report: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()
    config = load_config(Path(args.config))
    n = _resolve_n(config, args.n)
    n_clones = args.n_clones if args.n_clones is not None else N_CLONES
    name = _get_clean_name(config)

    print(f"\n=== Party Visibility Impact Sweep ({args.mode} mode) ===")
    print(f"  Config: {name}")
    print(f"  Top-k (n): {n}")
    print(f"  Clones per question: {n_clones}")

    if args.mode == "sweep":
        _run_sweep(args, config, n, n_clones)
    elif args.mode == "worker":
        _run_worker(args, config, n, n_clones)
    elif args.mode == "collect":
        _run_collect(args, config, n, n_clones)


if __name__ == "__main__":
    main()
