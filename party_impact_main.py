"""
Party impact analysis: identifies which questions, when cloned 4 times,
most strongly shift party visibility in voter recommendations.

Phase 1 (sweep/worker/collect): Sweep all questions with baseline-only
  comparisons to find which questions benefit which parties.

Phase 2 (phase2): For top-K questions from Phase 1, run full CRW pipeline
  with mixed paraphrase clones to demonstrate that CRW neutralizes the
  party visibility manipulation.

Usage:
    # Phase 1 — Sequential:
    python -m party_impact_main \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py

    # Phase 1 — Worker (one question, for SLURM array):
    python -m party_impact_main --mode worker --task-id 3 \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --sweep-dir /path/to/sweep_dir

    # Phase 1 — Collect (aggregate workers + plot):
    python -m party_impact_main --mode collect \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --sweep-dir /path/to/sweep_dir

    # Phase 2 — CRW correction for top-K:
    python -m party_impact_main --mode phase2 \\
        --config configs/full_pipeline/base_data/pipeline_e5_ZH.py \\
        --phase1-csv experiment_results/party_impact/party_impact_*.csv \\
        --top-k 5
"""

import argparse
import os
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

from clone_pipeline.applicator import apply_specs
from clone_pipeline.spec import CloneSpec
from configs import base_constants as default_config
from main import load_config
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine

sys.path.insert(0, str(Path(__file__).resolve().parent / "dependencies" / "rsfp"))
from rsfp.constants import BIG_PARTIES, PARTIES_LEFT_TO_RIGHT, PARTY2COLOR

N_CLONES = 4
RESULTS_DIR = default_config.RESULTS_DIR / "party_impact" / "high_impact"

# Parties to analyse (left-to-right order for plots, filtered to big parties)
MAJOR_PARTIES = [p for p in PARTIES_LEFT_TO_RIGHT if p in BIG_PARTIES]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Party impact analysis: find questions that shift party visibility"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Base pipeline config (e.g. configs/full_pipeline/base_data/pipeline_e5_ZH.py)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sweep", "worker", "collect", "phase2", "pre-paraphrases", "compile"],
        default="sweep",
        help="Execution mode",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Question index for worker mode (typically SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Directory for per-question worker CSVs",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Override top-k for party visibility (default: seats per canton)",
    )
    # Phase 2 args
    parser.add_argument(
        "--phase1-csv",
        type=str,
        default=None,
        help="Path to Phase 1 CSV for Phase 2 (auto-detected if omitted)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top questions to analyse in Phase 2",
    )
    parser.add_argument(
        "--target-party",
        type=str,
        default=None,
        help="Target party for Phase 2 (select questions that benefit this party most)",
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


def _build_candidate_party_map(df_candidates: pd.DataFrame) -> dict[int, str]:
    """Build candidate ID -> party name mapping."""
    return df_candidates.set_index("ID_candidate")["_party"].to_dict()


# ---------------------------------------------------------------------------
# Core: party visibility computation
# ---------------------------------------------------------------------------


def compute_party_visibility(
    rec_df: pd.DataFrame,
    candidate_party_map: dict[int, str],
    n: int,
) -> dict[str, float]:
    """
    Compute party visibility from a recommendation DataFrame.

    For each voter, looks at their top-n recommended candidates,
    counts the fraction belonging to each party, then averages
    across all voters.  Equivalent to the Swiss government method
    (each voter has one vote split proportionally).

    Returns dict mapping party name -> visibility fraction (sums to ~1.0).
    """
    # Get match columns sorted by rank
    match_cols = sorted(
        [c for c in rec_df.columns if re.match(r"^_matchID_\d+_", c)],
        key=lambda c: int(re.search(r"_matchID_(\d+)_", c).group(1)),
    )[:n]

    if not match_cols:
        # Try CRW columns if no baseline columns
        match_cols = sorted(
            [c for c in rec_df.columns if re.match(r"^CRW__matchID_\d+_", c)],
            key=lambda c: int(re.search(r"_matchID_(\d+)_", c).group(1)),
        )[:n]

    if not match_cols:
        raise ValueError("No recommendation columns found in DataFrame")

    # Extract top-n candidate IDs (shape: n_voters x n)
    top_n_matrix = rec_df[match_cols].values

    # Map to parties and compute per-voter shares
    party_sums: dict[str, float] = {}
    n_voters = len(top_n_matrix)

    for voter_row in top_n_matrix:
        valid = [candidate_party_map.get(int(c)) for c in voter_row if pd.notna(c)]
        total = len(valid)
        if total == 0:
            continue
        counts = Counter(valid)
        for party, count in counts.items():
            if party is not None:
                party_sums[party] = party_sums.get(party, 0.0) + count / total

    # Average across voters
    return {party: total / n_voters for party, total in party_sums.items()}




# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _setup_pipeline(config):
    """Load dataset, compute base baseline recs, build candidate party map."""
    print("\n--- Loading dataset ---")
    dataset = load_dataset(config)

    print("\n--- Computing base baseline recommendations ---")
    rec_engine = RecommendationEngine(config=config, data_map=dataset)
    base_recs = rec_engine.run_baseline()

    # Build candidate ID -> party mapping
    df_candidates = dataset["candidates"]
    candidate_party_map = _build_candidate_party_map(df_candidates)

    # Get sorted question IDs (excluding clones)
    questions_df = dataset["questions"]
    question_ids = sorted(
        questions_df.loc[
            questions_df["ID_question"] < 9_000_000, "ID_question"
        ].tolist()
    )

    print(f"\n  Questions: {len(question_ids)}")
    print(f"  Candidates: {len(candidate_party_map)}")
    party_dist = Counter(candidate_party_map.values())
    for p in MAJOR_PARTIES:
        print(f"    {p}: {party_dist.get(p, 0)} candidates")

    return {
        "dataset": dataset,
        "base_recs": base_recs,
        "candidate_party_map": candidate_party_map,
        "question_ids": question_ids,
        "questions_df": questions_df,
    }


# ---------------------------------------------------------------------------
# Per-question Phase 1 computation
# ---------------------------------------------------------------------------


def _compute_question_party_impact(
    q_id: int,
    config,
    pipeline: dict,
    n: int,
) -> dict:
    """Clone question 4x identical, compute baseline recs, measure party visibility delta."""
    dataset = pipeline["dataset"]
    candidate_party_map = pipeline["candidate_party_map"]
    text_col = _get_question_text_col(pipeline["questions_df"])

    # 1. Compute base party visibility
    base_visibility = compute_party_visibility(
        pipeline["base_recs"], candidate_party_map, n
    )

    # 2. Clone in-memory (4 identical clones)
    spec = CloneSpec(source_q_id=q_id, clone_type="identical", n_clones=N_CLONES)
    cloned_data = apply_specs(
        specs=[spec],
        dataframes={
            "questions": dataset["questions"],
            "voters": dataset["voters"],
            "candidates": dataset["candidates"],
        },
    )

    # 3. Compute cloned baseline recs
    cloned_engine = RecommendationEngine(config=config, data_map=cloned_data)
    cloned_recs = cloned_engine.run_baseline()
    sys.stdout.flush()

    # 4. Compute cloned party visibility
    cloned_visibility = compute_party_visibility(cloned_recs, candidate_party_map, n)

    # 5. Build result row
    q_text = pipeline["questions_df"].loc[
        pipeline["questions_df"]["ID_question"] == q_id, text_col
    ].iloc[0]

    row = {
        "question_id": q_id,
        "question_text": q_text,
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

    return row


# ---------------------------------------------------------------------------
# Mode: sweep (sequential)
# ---------------------------------------------------------------------------


def _run_sweep(args, config, n: int):
    pipeline = _setup_pipeline(config)
    question_ids = pipeline["question_ids"]

    rows = []
    for i, q_id in enumerate(question_ids):
        print(f"\n--- Question {q_id} ({i + 1}/{len(question_ids)}) ---")
        row = _compute_question_party_impact(q_id, config, pipeline, n)
        rows.append(row)
        print(
            f"  Max delta: {row['max_abs_delta']:.4f} "
            f"({row['max_delta_party']}, "
            f"direction: {'+' if row['max_positive_delta'] > 0 else ''}"
            f"{row['max_positive_delta']:.4f} for {row['max_positive_party']})"
        )

    sweep_df = pd.DataFrame(rows)

    name = _get_clean_name(config)
    output_dir = RESULTS_DIR / "phase1" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_phase1_outputs(sweep_df, config, n, output_dir)

    print("\n=== Party Impact Sweep Complete ===")


# ---------------------------------------------------------------------------
# Mode: worker (single question for SLURM array)
# ---------------------------------------------------------------------------


def _run_worker(args, config, n: int):
    task_id = args.task_id
    if task_id is None:
        print("ERROR: --task-id is required in worker mode", file=sys.stderr)
        sys.exit(1)

    name = _get_clean_name(config)
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "phase1" / name / "workers"
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
    out_path = sweep_dir / f"party_worker_{task_id:03d}_q{q_id}.csv"

    # Skip if already computed
    if out_path.exists():
        print(f"  [SKIP] Worker CSV already exists: {out_path}")
        return

    print(
        f"\n=== Worker: question {q_id} (task {task_id}/{len(question_ids) - 1}) ==="
    )

    row = _compute_question_party_impact(q_id, config, pipeline, n)

    worker_df = pd.DataFrame([row])
    worker_df.to_csv(out_path, index=False)

    print(f"\n  -> Worker CSV: {out_path}")
    print(
        f"  Max delta: {row['max_abs_delta']:.4f} ({row['max_delta_party']})"
    )
    print("\n=== Worker Complete ===")


# ---------------------------------------------------------------------------
# Mode: collect (aggregate + plot)
# ---------------------------------------------------------------------------


def _run_collect(args, config, n: int):
    name = _get_clean_name(config)
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "phase1" / name / "workers"

    worker_files = sorted(sweep_dir.glob("party_worker_*.csv"))
    if not worker_files:
        print(f"ERROR: No worker CSVs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\n=== Collect: reading {len(worker_files)} worker CSVs from {sweep_dir} ==="
    )

    dfs = [pd.read_csv(f) for f in worker_files]
    combined = pd.concat(dfs, ignore_index=True)

    name = _get_clean_name(config)
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
    """Sort, save CSV, generate heatmap + per-party ranking + report."""
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = f"party_impact_{name}_{timestamp}"

    # Sort by max absolute delta (most impactful first)
    df = df.sort_values("max_abs_delta", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = output_dir / f"{base}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> CSV: {csv_path.name}")

    # Plots
    sns.set_theme(style="whitegrid")
    _plot_heatmap(df, output_dir, base)
    _plot_per_party(df, output_dir, base)
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

    # Row labels: question ID + truncated text
    row_labels = [
        f"Q{int(r['question_id'])}  {str(r['question_text'])[:45]}"
        for _, r in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.28)))

    # Scale to percentage points for readability
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
        f"Party Visibility Change from Cloning ({N_CLONES}× identical)\n"
        f"Red = party gains visibility, Blue = party loses visibility"
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
        values = top5[delta_col].values * 100  # percentage points

        color = PARTY2COLOR.get(party, "#888888")
        ax.barh(range(len(labels)), values, color=color, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Δ Visibility (pp)")
        ax.set_title(f"{party}", fontsize=13, fontweight="bold", color=color)
        ax.axvline(0, color="black", linewidth=0.5)

    # Hide extra axes if fewer than 6 parties
    for i in range(len(MAJOR_PARTIES), n_rows * n_cols):
        axes.flat[i].set_visible(False)

    fig.suptitle(
        f"Top Questions Benefiting Each Party ({N_CLONES}× identical clones)",
        fontsize=14,
    )
    fig.tight_layout()

    path = output_dir / f"{base}_per_party.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Per-party ranking: {path.name}")


def _save_phase1_report(
    df: pd.DataFrame, output_dir: Path, base: str, n: int, top_n: int = 10
):
    """Save human-readable summary of Phase 1 results."""
    lines = [
        "=" * 80,
        "PARTY IMPACT ANALYSIS — PHASE 1 REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Clone type: identical, n_clones: {N_CLONES}",
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
        for p in MAJOR_PARTIES:
            d = row.get(f"delta_{p}", 0)
            if abs(d) > 0.0001:
                lines.append(f"       {p:>8s}: {d * 100:+.2f}pp")
        lines.append("")

    # Per-party summary: which question benefits each party most
    lines.extend(["=" * 80, "PER-PARTY BEST QUESTION:", "-" * 80])
    for p in MAJOR_PARTIES:
        delta_col = f"delta_{p}"
        if delta_col in df.columns:
            best_idx = df[delta_col].idxmax()
            best = df.loc[best_idx]
            lines.append(
                f"  {p:>8s}: Q{int(best['question_id'])} "
                f"({best[delta_col] * 100:+.2f}pp) — "
                f'"{str(best["question_text"])[:50]}"'
            )

    lines.append("=" * 80)

    path = output_dir / f"{base}_report.txt"
    path.write_text("\n".join(lines))
    print(f"  -> Report: {path.name}")


# ---------------------------------------------------------------------------
# Mode: phase2 (CRW correction for top-K questions)
# ---------------------------------------------------------------------------


def _run_phase2(args, config, n: int):
    from clone_pipeline.paraphrase_generator import ensure_paraphrases
    from vqs.clone_robust_weighting import CloneRobustReweighter
    from vqs.similarity_metrics import get_calculator

    # 1. Load Phase 1 results
    if args.phase1_csv:
        phase1_path = Path(args.phase1_csv)
    else:
        # Auto-detect latest Phase 1 CSV
        csvs = sorted((RESULTS_DIR / "phase1").glob("**/party_impact_*.csv"))
        if not csvs:
            print("ERROR: No Phase 1 CSV found. Run Phase 1 first.", file=sys.stderr)
            sys.exit(1)
        phase1_path = csvs[-1]

    print(f"\n=== Phase 2: Loading Phase 1 results from {phase1_path.name} ===")
    phase1_df = pd.read_csv(phase1_path)

    top_k = args.top_k
    target_party = args.target_party

    if target_party:
        sort_col = f"delta_{target_party}"
        if sort_col not in phase1_df.columns:
            print(
                f"ERROR: Party '{target_party}' not found in Phase 1 CSV. "
                f"Available: {[c.replace('delta_', '') for c in phase1_df.columns if c.startswith('delta_')]}",
                file=sys.stderr,
            )
            sys.exit(1)
        top_questions = phase1_df.nlargest(top_k, sort_col)
        print(f"  Top-{top_k} questions benefiting {target_party}:")
        for _, row in top_questions.iterrows():
            print(
                f"    Q{int(row['question_id'])}: "
                f"Δ({target_party})={row[sort_col] * 100:+.2f}pp"
            )
    else:
        top_questions = phase1_df.nlargest(top_k, "max_abs_delta")
        print(f"  Top-{top_k} questions by max |Δ|:")
        for _, row in top_questions.iterrows():
            print(
                f"    Q{int(row['question_id'])}: "
                f"|Δ|={row['max_abs_delta'] * 100:.2f}pp ({row['max_delta_party']})"
            )

    # 2. Load dataset
    print("\n--- Loading dataset ---")
    dataset = load_dataset(config)
    candidate_party_map = _build_candidate_party_map(dataset["candidates"])
    questions_df = dataset["questions"]

    # 3. Compute original baseline + original CRW (once)
    print("\n--- Computing original baseline recommendations ---")
    rec_engine = RecommendationEngine(config=config, data_map=dataset)
    base_recs = rec_engine.run_baseline()
    base_visibility = compute_party_visibility(base_recs, candidate_party_map, n)

    print("\n--- Computing original CRW recommendations ---")
    calculator = get_calculator(config)
    base_dist_df = calculator.calculate_distance(dataset, config)
    reweighter = CloneRobustReweighter(config)
    base_weights = reweighter.reweight(base_dist_df)
    base_crw_recs = rec_engine.run_crw(base_weights)
    base_crw_visibility = compute_party_visibility(
        base_crw_recs, candidate_party_map, n
    )

    print("\n  Original baseline visibility:")
    for p in MAJOR_PARTIES:
        print(f"    {p}: {base_visibility.get(p, 0) * 100:.2f}%")
    print("  Original CRW visibility:")
    for p in MAJOR_PARTIES:
        print(f"    {p}: {base_crw_visibility.get(p, 0) * 100:.2f}%")

    # 4. Build cumulative clone scenarios
    q_ids = [int(row["question_id"]) for _, row in top_questions.iterrows()]
    q_texts = {}
    for q_id in q_ids:
        q_texts[q_id] = questions_df.loc[
            questions_df["ID_question"] == q_id,
            _get_question_text_col(questions_df),
        ].iloc[0]

    saved_clone_id = config.clone_id
    q_tag = "_".join(str(q) for q in sorted(q_ids))

    # --- Scenario A: Worst-case (all top questions × 4 mixed clones) ---
    n_mixed = N_CLONES
    print(f"\n{'='*60}")
    print(
        f"  Scenario A: Worst-case "
        f"({len(q_ids)} questions × {n_mixed} mixed clones "
        f"= {len(q_ids) * n_mixed} clones)"
    )
    print(f"{'='*60}")

    worst_specs = []
    for q_id in q_ids:
        worst_specs.extend([
            CloneSpec(
                source_q_id=q_id, clone_type="easy_paraphrase", n_clones=1
            ),
            CloneSpec(
                source_q_id=q_id, clone_type="hard_paraphrase", n_clones=1
            ),
            CloneSpec(
                source_q_id=q_id, clone_type="negation_easy",
                n_clones=1, flip_answers=True,
            ),
            CloneSpec(
                source_q_id=q_id, clone_type="negation_hard",
                n_clones=1, flip_answers=True,
            ),
        ])

    print(f"  Generating/loading paraphrases for {len(worst_specs)} specs...")
    paraphrases_a = ensure_paraphrases(
        specs=worst_specs,
        questions_df=questions_df,
        data_year=config.data_year,
        paraphrase_dir=config.PARAPHRASES_DIR,
    )

    cloned_data_a = apply_specs(
        specs=worst_specs,
        dataframes={
            "questions": dataset["questions"],
            "voters": dataset["voters"],
            "candidates": dataset["candidates"],
        },
        paraphrases=paraphrases_a,
    )

    print("  Computing worst-case baseline...")
    engine_a = RecommendationEngine(config=config, data_map=cloned_data_a)
    recs_a_base = engine_a.run_baseline()
    vis_a_base = compute_party_visibility(recs_a_base, candidate_party_map, n)

    print("  Computing worst-case CRW...")
    config.clone_id = f"party_impact_worst_{q_tag}"
    dist_a = calculator.calculate_distance(cloned_data_a, config)
    reweighter_a = CloneRobustReweighter(config)
    weights_a = reweighter_a.reweight(dist_a)
    recs_a_crw = engine_a.run_crw(weights_a)
    vis_a_crw = compute_party_visibility(recs_a_crw, candidate_party_map, n)
    config.clone_id = saved_clone_id

    print("\n  Worst-case visibility:")
    for p in MAJOR_PARTIES:
        d = (vis_a_base.get(p, 0) - base_visibility.get(p, 0)) * 100
        print(f"    {p}: {vis_a_base.get(p, 0) * 100:.2f}% (Δ{d:+.2f}pp)")

    # --- Scenario B: Realistic (all top questions × 1 easy paraphrase) ---
    print(f"\n{'='*60}")
    print(
        f"  Scenario B: Realistic "
        f"({len(q_ids)} questions × 1 easy paraphrase "
        f"= {len(q_ids)} clones)"
    )
    print(f"{'='*60}")

    real_specs = [
        CloneSpec(source_q_id=q_id, clone_type="easy_paraphrase", n_clones=1)
        for q_id in q_ids
    ]

    paraphrases_b = ensure_paraphrases(
        specs=real_specs,
        questions_df=questions_df,
        data_year=config.data_year,
        paraphrase_dir=config.PARAPHRASES_DIR,
    )

    cloned_data_b = apply_specs(
        specs=real_specs,
        dataframes={
            "questions": dataset["questions"],
            "voters": dataset["voters"],
            "candidates": dataset["candidates"],
        },
        paraphrases=paraphrases_b,
    )

    print("  Computing realistic baseline...")
    engine_b = RecommendationEngine(config=config, data_map=cloned_data_b)
    recs_b_base = engine_b.run_baseline()
    vis_b_base = compute_party_visibility(recs_b_base, candidate_party_map, n)

    print("  Computing realistic CRW...")
    config.clone_id = f"party_impact_real_{q_tag}"
    dist_b = calculator.calculate_distance(cloned_data_b, config)
    reweighter_b = CloneRobustReweighter(config)
    weights_b = reweighter_b.reweight(dist_b)
    recs_b_crw = engine_b.run_crw(weights_b)
    vis_b_crw = compute_party_visibility(recs_b_crw, candidate_party_map, n)
    config.clone_id = saved_clone_id

    print("\n  Realistic visibility:")
    for p in MAJOR_PARTIES:
        d = (vis_b_base.get(p, 0) - base_visibility.get(p, 0)) * 100
        print(f"    {p}: {vis_b_base.get(p, 0) * 100:.2f}% (Δ{d:+.2f}pp)")

    sys.stdout.flush()

    # Build results
    results = {
        "q_ids": q_ids,
        "q_texts": q_texts,
        "visibility": {
            "original": base_visibility,
            "original_crw": base_crw_visibility,
            "worst_case": vis_a_base,
            "worst_case_crw": vis_a_crw,
            "realistic": vis_b_base,
            "realistic_crw": vis_b_crw,
        },
    }

    name = _get_clean_name(config)
    party_subdir = target_party if target_party else "no_target"
    output_dir = RESULTS_DIR / "phase2" / name / party_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_phase2_outputs(results, config, n, output_dir, target_party)

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
):
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    party_tag = f"_{target_party}" if target_party else ""
    base = f"party_impact_{name}_{timestamp}{party_tag}"

    # Save CSV (one row per party, columns = conditions)
    vis = results["visibility"]
    q_ids_str = ";".join(str(q) for q in results["q_ids"])
    rows = []
    for p in MAJOR_PARTIES:
        o = vis["original"].get(p, 0)
        oc = vis["original_crw"].get(p, 0)
        wc = vis["worst_case"].get(p, 0)
        wc_crw = vis["worst_case_crw"].get(p, 0)
        rl = vis["realistic"].get(p, 0)
        rl_crw = vis["realistic_crw"].get(p, 0)
        da_w = wc - o
        dc_w = wc_crw - oc
        da_r = rl - o
        dc_r = rl_crw - oc
        rows.append({
            "target_party": target_party or "",
            "cloned_questions": q_ids_str,
            "party": p,
            "original": o,
            "original_crw": oc,
            "crw_drift": oc - o,
            "worst_case": wc,
            "worst_case_crw": wc_crw,
            "delta_attack_worst": da_w,
            "delta_crw_worst": dc_w,
            "reduction_worst": (1 - abs(dc_w) / abs(da_w)) * 100 if abs(da_w) > 1e-9 else 0,
            "realistic": rl,
            "realistic_crw": rl_crw,
            "delta_attack_realistic": da_r,
            "delta_crw_realistic": dc_r,
            "reduction_realistic": (1 - abs(dc_r) / abs(da_r)) * 100 if abs(da_r) > 1e-9 else 0,
        })
    csv_df = pd.DataFrame(rows)
    csv_path = output_dir / f"{base}_phase2.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\n  -> Phase 2 CSV: {csv_path.name}")

    sns.set_theme(style="whitegrid")
    _plot_phase2_showcase(results, output_dir, base, target_party)
    _plot_phase2_deltas(results, output_dir, base, target_party)
    _plot_phase2_donuts(results, output_dir, base, target_party)
    _plot_phase2_hemicycle(results, output_dir, base, target_party)
    _save_phase2_report(results, output_dir, base, n, target_party)


# ---------------------------------------------------------------------------
# Phase 2 plots
# ---------------------------------------------------------------------------


def _plot_phase2_showcase(
    results: dict, output_dir: Path, base: str, target_party: str | None = None
):
    """Grouped bar chart: absolute party visibility under all conditions."""
    vis = results["visibility"]

    x = np.arange(len(MAJOR_PARTIES))
    width = 0.15

    conditions = [
        ("Original", "original", "#4CAF50"),
        ("Worst-case attack", "worst_case", "#F44336"),
        ("Worst-case + CRW", "worst_case_crw", "#2196F3"),
        ("Realistic attack", "realistic", "#FF8A65"),
        ("Realistic + CRW", "realistic_crw", "#64B5F6"),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (label, key, color) in enumerate(conditions):
        vals = [vis[key].get(p, 0) * 100 for p in MAJOR_PARTIES]
        ax.bar(x + (i - 2) * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(MAJOR_PARTIES, fontsize=12)
    ax.set_ylabel("Party Visibility (%)", fontsize=12)

    n_q = len(results["q_ids"])
    target_str = f" targeting {target_party}" if target_party else ""
    ax.set_title(
        f"Cumulative Party Visibility under Clone Manipulation{target_str}\n"
        f"({n_q} questions cloned: worst-case = 4 mixed each, "
        f"realistic = 1 easy paraphrase each)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)

    all_vals = []
    for _, key, _ in conditions:
        all_vals.extend([vis[key].get(p, 0) * 100 for p in MAJOR_PARTIES])
    ax.set_ylim(0, max(all_vals) * 1.15)

    fig.tight_layout()
    path = output_dir / f"{base}_showcase.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Showcase: {path.name}")


def _plot_phase2_deltas(
    results: dict, output_dir: Path, base: str, target_party: str | None = None
):
    """Bar chart: change from original baseline for each scenario."""
    vis = results["visibility"]
    orig = vis["original"]

    x = np.arange(len(MAJOR_PARTIES))
    width = 0.18

    conditions = [
        ("Worst-case attack", "worst_case", "#F44336"),
        ("Worst-case + CRW", "worst_case_crw", "#2196F3"),
        ("Realistic attack", "realistic", "#FF8A65"),
        ("Realistic + CRW", "realistic_crw", "#64B5F6"),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (label, key, color) in enumerate(conditions):
        vals = [(vis[key].get(p, 0) - orig.get(p, 0)) * 100 for p in MAJOR_PARTIES]
        ax.bar(
            x + (i - 1.5) * width, vals, width,
            label=label, color=color, alpha=0.85,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(MAJOR_PARTIES, fontsize=12)
    ax.set_ylabel("Δ Party Visibility (pp)", fontsize=12)

    # Annotate reduction % for target party
    focus = target_party if target_party else max(
        MAJOR_PARTIES,
        key=lambda p: abs(vis["worst_case"].get(p, 0) - orig.get(p, 0)),
    )
    if focus in MAJOR_PARTIES:
        p_idx = MAJOR_PARTIES.index(focus)
        orig_crw = vis["original_crw"]
        d_worst = (vis["worst_case"].get(focus, 0) - orig.get(focus, 0)) * 100
        d_worst_crw = (vis["worst_case_crw"].get(focus, 0) - orig_crw.get(focus, 0)) * 100
        d_real = (vis["realistic"].get(focus, 0) - orig.get(focus, 0)) * 100
        d_real_crw = (vis["realistic_crw"].get(focus, 0) - orig_crw.get(focus, 0)) * 100

        if abs(d_worst) > 0.001:
            red_w = (1 - abs(d_worst_crw) / abs(d_worst)) * 100
            bar_x = p_idx + (1 - 1.5) * width
            ax.annotate(
                f"{red_w:.0f}% reduced",
                (bar_x, d_worst_crw),
                fontsize=9, ha="center",
                va="bottom" if d_worst_crw >= 0 else "top",
                color="#1565C0", fontweight="bold",
            )
        if abs(d_real) > 0.001:
            red_r = (1 - abs(d_real_crw) / abs(d_real)) * 100
            bar_x = p_idx + (3 - 1.5) * width
            ax.annotate(
                f"{red_r:.0f}% reduced",
                (bar_x, d_real_crw),
                fontsize=9, ha="center",
                va="bottom" if d_real_crw >= 0 else "top",
                color="#1565C0", fontweight="bold",
            )

    n_q = len(results["q_ids"])
    target_str = f" targeting {target_party}" if target_party else ""
    ax.set_title(
        f"Change in Party Visibility from Clone Manipulation{target_str}\n"
        f"({n_q} questions cloned simultaneously)",
        fontsize=11,
    )
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    path = output_dir / f"{base}_deltas.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Delta chart: {path.name}")


def _plot_phase2_donuts(
    results: dict, output_dir: Path, base: str, target_party: str | None = None
):
    """Side-by-side donut charts showing party visibility share per condition.

    Layout: 2 rows × 3 cols (worst-case row + realistic row).
    Each donut shows party-colored slices with external annotations
    including Δpp from the original baseline.
    """
    from matplotlib.patches import Patch

    vis = results["visibility"]
    orig = vis["original"]

    scenarios = [
        (
            "Worst-case (20 clones)",
            [
                ("Original", "original"),
                ("Attacked", "worst_case"),
                ("CRW-corrected", "worst_case_crw"),
            ],
        ),
        (
            "Realistic (5 clones)",
            [
                ("Original", "original"),
                ("Attacked", "realistic"),
                ("CRW-corrected", "realistic_crw"),
            ],
        ),
    ]

    all_parties = list(MAJOR_PARTIES) + ["Others"]
    all_colors = [PARTY2COLOR.get(p, "#888") for p in MAJOR_PARTIES] + ["#E0E0E0"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    for row_idx, (scenario_title, conditions) in enumerate(scenarios):
        for col_idx, (cond_title, key) in enumerate(conditions):
            ax = axes[row_idx][col_idx]

            major_vals = [vis[key].get(p, 0) for p in MAJOR_PARTIES]
            others = max(0.0, 1.0 - sum(major_vals))
            all_vals = major_vals + [others]
            orig_vals = (
                [orig.get(p, 0) for p in MAJOR_PARTIES]
                + [max(0.0, 1.0 - sum(orig.get(p, 0) for p in MAJOR_PARTIES))]
            )

            # Explode Centre slightly in attacked conditions
            explode = [0.0] * len(all_vals)
            if target_party and key != "original":
                tp_list = list(MAJOR_PARTIES)
                if target_party in tp_list:
                    explode[tp_list.index(target_party)] = 0.05

            wedges, _ = ax.pie(
                all_vals,
                labels=None,
                colors=all_colors,
                startangle=90,
                counterclock=False,
                explode=explode,
                wedgeprops=dict(width=0.4, edgecolor="white", linewidth=1.5),
            )

            # Annotate each slice externally
            is_original = key == "original"
            for i, (wedge, party) in enumerate(zip(wedges, all_parties)):
                pct = all_vals[i] * 100
                if pct < 1.5:
                    continue

                ang = np.radians((wedge.theta1 + wedge.theta2) / 2)

                # Label position outside the donut
                lx = 1.25 * np.cos(ang)
                ly = 1.25 * np.sin(ang)
                # Connection point on donut outer edge
                cx = 1.02 * np.cos(ang)
                cy = 1.02 * np.sin(ang)

                txt = f"{party}\n{pct:.1f}%"
                fontcolor = "black"
                if not is_original and party != "Others":
                    delta = (all_vals[i] - orig_vals[i]) * 100
                    if abs(delta) >= 0.05:
                        txt += f"\n({delta:+.1f}pp)"
                        fontcolor = "#C62828" if delta < 0 else "#2E7D32"

                ha = "left" if lx > 0 else "right"
                is_focus = party == target_party and not is_original
                ax.annotate(
                    txt,
                    xy=(cx, cy),
                    xytext=(lx, ly),
                    fontsize=9 if pct > 5 else 7,
                    ha=ha,
                    va="center",
                    fontweight="bold" if is_focus else "normal",
                    color=fontcolor,
                    arrowprops=dict(arrowstyle="-", color="#BDBDBD", lw=0.8),
                )

            title = cond_title
            if col_idx == 0:
                title = f"{scenario_title}\n{cond_title}"
            ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Shared legend
    legend_handles = [
        Patch(facecolor=c, edgecolor="white", label=p)
        for p, c in zip(all_parties, all_colors)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(all_parties),
        fontsize=10,
        frameon=False,
    )

    target_str = f" targeting {target_party}" if target_party else ""
    fig.suptitle(
        f"Party Visibility Distribution under Clone Manipulation{target_str}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    path = output_dir / f"{base}_donuts.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Donuts: {path.name}")


def _plot_phase2_hemicycle(
    results: dict, output_dir: Path, base: str, target_party: str | None = None
):
    """Parliament-style hemicycle showing 'seats' colored by party.

    Three hemicycles: Original → Worst-case attack → CRW-corrected.
    100 dots arranged in 5 concentric arcs, colored proportionally to
    party visibility.  Parties are ordered left-to-right politically
    (SP on the left, SVP on the right), matching Swiss parliament layout.
    """
    from matplotlib.patches import Patch

    vis = results["visibility"]
    orig = vis["original"]

    conditions = [
        ("Original", "original"),
        ("Worst-case attack\n(20 clones)", "worst_case"),
        ("CRW-corrected", "worst_case_crw"),
    ]

    n_seats = 100
    seats_per_row = [12, 16, 20, 24, 28]  # inner to outer, sum = 100
    all_parties = list(MAJOR_PARTIES) + ["Others"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))

    for ax, (title, key) in zip(axes, conditions):
        # Allocate seats proportionally
        major_pcts = {p: vis[key].get(p, 0) for p in MAJOR_PARTIES}
        raw_seats = {p: major_pcts[p] * n_seats for p in MAJOR_PARTIES}

        # Round with largest-remainder method to preserve total
        floored = {p: int(v) for p, v in raw_seats.items()}
        remainders = {p: raw_seats[p] - floored[p] for p in MAJOR_PARTIES}
        allocated = sum(floored.values())
        others_seats = n_seats - int(round(sum(major_pcts.values()) * n_seats))
        major_total = n_seats - others_seats
        to_distribute = major_total - allocated
        for p in sorted(remainders, key=remainders.get, reverse=True):
            if to_distribute <= 0:
                break
            floored[p] += 1
            to_distribute -= 1

        seat_counts = floored
        seat_counts["Others"] = others_seats

        # Build ordered color list
        seat_colors = []
        for p in MAJOR_PARTIES:
            seat_colors.extend(
                [PARTY2COLOR.get(p, "#888")] * seat_counts.get(p, 0)
            )
        seat_colors.extend(["#E0E0E0"] * seat_counts.get("Others", 0))

        # Compute dot positions — concentric arcs
        all_x, all_y = [], []
        seat_idx = 0
        margin = 5  # degrees from horizontal
        for row, n_row in enumerate(seats_per_row):
            r = 2.0 + row * 0.55
            angles = np.linspace(
                np.pi - np.radians(margin),
                np.radians(margin),
                n_row,
            )
            for angle in angles:
                if seat_idx < n_seats:
                    all_x.append(r * np.cos(angle))
                    all_y.append(r * np.sin(angle))
                    seat_idx += 1

        ax.scatter(
            all_x,
            all_y,
            c=seat_colors[: len(all_x)],
            s=90,
            edgecolors="white",
            linewidths=0.6,
            zorder=2,
        )

        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-1.0, 5.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

        # Per-party summary below the hemicycle
        summary_parts = []
        for p in MAJOR_PARTIES:
            s = seat_counts.get(p, 0)
            if key == "original":
                summary_parts.append(f"{p}: {s}")
            else:
                orig_s = round(orig.get(p, 0) * n_seats)
                diff = s - orig_s
                color = "#C62828" if diff < 0 else "#2E7D32" if diff > 0 else "black"
                sign = "+" if diff > 0 else ""
                summary_parts.append(f"{p}: {s} ({sign}{diff})")
        ax.text(
            0,
            -0.5,
            "   ".join(summary_parts),
            ha="center",
            va="top",
            fontsize=9,
            family="monospace",
        )

        # Highlight target party change in center
        if key != "original" and target_party:
            delta = (vis[key].get(target_party, 0) - orig.get(target_party, 0)) * 100
            ax.text(
                0,
                1.5,
                f"{target_party}: {delta:+.1f}pp",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=PARTY2COLOR.get(target_party, "black"),
            )

    # Legend
    legend_handles = [
        Patch(facecolor=PARTY2COLOR.get(p, "#888"), edgecolor="white", label=p)
        for p in MAJOR_PARTIES
    ] + [Patch(facecolor="#E0E0E0", edgecolor="white", label="Others")]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(all_parties),
        fontsize=10,
        frameon=False,
    )

    target_str = f" targeting {target_party}" if target_party else ""
    fig.suptitle(
        f"Parliament Seat Distribution under Clone Manipulation{target_str}",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    path = output_dir / f"{base}_hemicycle.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Hemicycle: {path.name}")


def _save_phase2_report(
    results: dict, output_dir: Path, base: str, n: int,
    target_party: str | None = None,
):
    """Save Phase 2 human-readable report."""
    vis = results["visibility"]
    orig = vis["original"]
    n_q = len(results["q_ids"])

    lines = [
        "=" * 80,
        "PARTY IMPACT ANALYSIS — PHASE 2 REPORT (CUMULATIVE)",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Target party: {target_party or '(none)'}",
        f"Top-k (n): {n}",
        f"Questions cloned: {n_q}",
        "",
        "SELECTED QUESTIONS:",
    ]

    for q_id in results["q_ids"]:
        text = str(results["q_texts"][q_id])[:70]
        lines.append(f"  Q{q_id}: \"{text}\"")

    lines.append("")

    # Scenario A table
    lines.append("=" * 80)
    lines.append(
        f"SCENARIO A — WORST-CASE: 4 mixed clones per question "
        f"({n_q * N_CLONES} total clones)"
    )
    lines.append(
        "  Clone types: easy paraphrase, hard paraphrase, "
        "negation (easy), negation (hard)"
    )
    lines.append("=" * 80)
    lines.append(
        f"{'Party':>8s}  {'Original':>8s}  {'Attacked':>8s}  {'CRW Fix':>8s}  "
        f"{'Δ Attack':>8s}  {'Δ CRW':>8s}  {'Reduced':>7s}"
    )
    lines.append("-" * 70)

    orig_crw = vis["original_crw"]
    for p in MAJOR_PARTIES:
        o = orig.get(p, 0) * 100
        oc = orig_crw.get(p, 0) * 100
        a = vis["worst_case"].get(p, 0) * 100
        c = vis["worst_case_crw"].get(p, 0) * 100
        da = a - o
        dc = c - oc
        reduction = (1 - abs(dc) / abs(da)) * 100 if abs(da) > 0.001 else 0
        lines.append(
            f"{p:>8s}  {o:>7.2f}%  {a:>7.2f}%  {c:>7.2f}%  "
            f"{da:>+7.2f}  {dc:>+7.2f}  {reduction:>6.0f}%"
        )

    lines.append("")

    # Scenario B table
    lines.append("=" * 80)
    lines.append(
        f"SCENARIO B — REALISTIC: 1 easy paraphrase per question "
        f"({n_q} total clones)"
    )
    lines.append("=" * 80)
    lines.append(
        f"{'Party':>8s}  {'Original':>8s}  {'Attacked':>8s}  {'CRW Fix':>8s}  "
        f"{'Δ Attack':>8s}  {'Δ CRW':>8s}  {'Reduced':>7s}"
    )
    lines.append("-" * 70)

    for p in MAJOR_PARTIES:
        o = orig.get(p, 0) * 100
        oc = orig_crw.get(p, 0) * 100
        a = vis["realistic"].get(p, 0) * 100
        c = vis["realistic_crw"].get(p, 0) * 100
        da = a - o
        dc = c - oc
        reduction = (1 - abs(dc) / abs(da)) * 100 if abs(da) > 0.001 else 0
        lines.append(
            f"{p:>8s}  {o:>7.2f}%  {a:>7.2f}%  {c:>7.2f}%  "
            f"{da:>+7.2f}  {dc:>+7.2f}  {reduction:>6.0f}%"
        )

    lines.append("")

    # CRW natural drift
    lines.append("=" * 80)
    lines.append("CRW NATURAL DRIFT (CRW on original questionnaire, no clones)")
    lines.append("=" * 80)
    for p in MAJOR_PARTIES:
        o = orig.get(p, 0) * 100
        c = vis["original_crw"].get(p, 0) * 100
        d = c - o
        lines.append(f"  {p:>8s}: {o:.2f}% -> {c:.2f}% (Δ{d:+.2f}pp)")

    lines.append("")
    lines.append("=" * 80)

    path = output_dir / f"{base}_phase2_report.txt"
    path.write_text("\n".join(lines))
    print(f"  -> Phase 2 report: {path.name}")


# ---------------------------------------------------------------------------
# Pre-paraphrase generation (for parallel party sweep)
# ---------------------------------------------------------------------------


def _run_pre_paraphrases(args, config):
    """Pre-generate paraphrases for ALL parties' top-K questions.

    Collects the union of top-K questions across all major parties from the
    Phase 1 CSV, then generates all needed paraphrases in a single serial
    call.  This primes the cache so that parallel Phase 2 jobs only read
    (no concurrent writes).
    """
    from clone_pipeline.paraphrase_generator import ensure_paraphrases

    # Load Phase 1 CSV
    if args.phase1_csv:
        phase1_path = Path(args.phase1_csv)
    else:
        csvs = sorted((RESULTS_DIR / "phase1").glob("**/party_impact_*.csv"))
        if not csvs:
            print("ERROR: No Phase 1 CSV found.", file=sys.stderr)
            sys.exit(1)
        phase1_path = csvs[-1]

    print(f"  Loading Phase 1 CSV: {phase1_path.name}")
    phase1_df = pd.read_csv(phase1_path)
    top_k = args.top_k

    # Collect union of top-K question IDs across all parties
    all_q_ids: set[int] = set()
    for party in MAJOR_PARTIES:
        col = f"delta_{party}"
        if col not in phase1_df.columns:
            print(f"  WARNING: {col} not in Phase 1 CSV, skipping {party}")
            continue
        top_qs = phase1_df.nlargest(top_k, col)
        party_ids = set(int(row["question_id"]) for _, row in top_qs.iterrows())
        print(f"  {party}: top-{top_k} questions = {sorted(party_ids)}")
        all_q_ids |= party_ids

    print(f"\n  Union of all questions: {len(all_q_ids)} unique IDs")
    print(f"  IDs: {sorted(all_q_ids)}")

    # Load questions for text
    print("\n  Loading questions data...")
    dataset = load_dataset(config)
    questions_df = dataset["questions"]

    # Build specs: 4 mixed types + easy_paraphrase (for realistic scenario)
    specs = []
    for q_id in sorted(all_q_ids):
        specs.extend([
            CloneSpec(source_q_id=q_id, clone_type="easy_paraphrase", n_clones=1),
            CloneSpec(source_q_id=q_id, clone_type="hard_paraphrase", n_clones=1),
            CloneSpec(
                source_q_id=q_id, clone_type="negation_easy",
                n_clones=1, flip_answers=True,
            ),
            CloneSpec(
                source_q_id=q_id, clone_type="negation_hard",
                n_clones=1, flip_answers=True,
            ),
        ])

    print(f"  Total specs: {len(specs)} ({len(all_q_ids)} questions × 4 types)")
    print("  Generating/loading paraphrases...")

    ensure_paraphrases(
        specs=specs,
        questions_df=questions_df,
        data_year=config.data_year,
        paraphrase_dir=config.PARAPHRASES_DIR,
    )

    print("\n=== Pre-paraphrase generation complete ===")


# ---------------------------------------------------------------------------
# Compile: aggregate all 6 party Phase 2 results
# ---------------------------------------------------------------------------


def _run_compile(args, config):
    """Load all per-party Phase 2 CSVs and produce aggregated outputs."""
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = f"party_impact_{name}_{timestamp}_compiled"

    phase2_dir = RESULTS_DIR / "phase2" / name

    # Find Phase 2 CSVs with party tags (search in phase2/{name}/*/)
    csvs = {}
    for party in MAJOR_PARTIES:
        pattern = f"**/*_{party}_phase2.csv"
        matches = sorted(phase2_dir.glob(pattern))
        if not matches:
            print(f"  WARNING: No Phase 2 CSV for {party}, skipping")
            continue
        csvs[party] = matches[-1]  # most recent
        print(f"  {party}: {matches[-1].name}")

    if len(csvs) < 2:
        print("ERROR: Need at least 2 party CSVs to compile.", file=sys.stderr)
        sys.exit(1)

    # Load all CSVs: dict[target_party] -> DataFrame
    dfs = {}
    for party, path in csvs.items():
        dfs[party] = pd.read_csv(path)

    print(f"\n  Loaded {len(dfs)} party results, compiling...")

    # Load Phase 1 CSV to recover cloned question IDs per target party
    phase1_q_ids = {}
    if args.phase1_csv:
        phase1_path = Path(args.phase1_csv)
    else:
        phase1_csvs = sorted(
            (RESULTS_DIR / "phase1").glob("**/party_impact_*.csv")
        )
        phase1_path = phase1_csvs[-1] if phase1_csvs else None

    if phase1_path and phase1_path.exists():
        phase1_df = pd.read_csv(phase1_path)
        top_k = args.top_k
        for party in MAJOR_PARTIES:
            col = f"delta_{party}"
            if col in phase1_df.columns:
                top_qs = phase1_df.nlargest(top_k, col)["question_id"].astype(int).tolist()
                phase1_q_ids[party] = ";".join(str(q) for q in top_qs)
        print(f"  Phase 1 CSV: {phase1_path.name} (recovered question IDs)")
    else:
        print("  WARNING: No Phase 1 CSV found, cloned_questions will be empty")

    compiled_dir = phase2_dir / "compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)

    # Save compiled CSV: all rows from all per-party CSVs with derived columns
    compiled_rows = []
    for target in MAJOR_PARTIES:
        if target not in dfs:
            continue
        df = dfs[target]
        # Get cloned question IDs from per-party CSV or Phase 1 fallback
        q_ids_str = ""
        if "cloned_questions" in df.columns and df["cloned_questions"].notna().any():
            q_ids_str = df["cloned_questions"].dropna().iloc[0]
        elif target in phase1_q_ids:
            q_ids_str = phase1_q_ids[target]
        for _, row in df.iterrows():
            p = row["party"]
            o = row["original"]
            oc = row["original_crw"]
            # Support both old (no absolute cols) and new CSV formats
            has_abs = "worst_case" in df.columns
            if has_abs:
                wc = row["worst_case"]
                wc_crw = row["worst_case_crw"]
                rl = row["realistic"]
                rl_crw = row["realistic_crw"]
            else:
                wc = wc_crw = rl = rl_crw = float("nan")
            da_w = wc - o
            dc_w = wc_crw - oc
            da_r = rl - o
            dc_r = rl_crw - oc
            red_w = (1 - abs(dc_w) / abs(da_w)) * 100 if abs(da_w) > 1e-9 else 0
            red_r = (1 - abs(dc_r) / abs(da_r)) * 100 if abs(da_r) > 1e-9 else 0
            r = {
                "target_party": target,
                "cloned_questions": q_ids_str,
                "party": p,
                "original": o,
                "original_crw": oc,
                "crw_drift": oc - o,
                "worst_case": wc,
                "worst_case_crw": wc_crw,
                "delta_attack_worst": da_w,
                "delta_crw_worst": dc_w,
                "reduction_worst": red_w,
                "realistic": rl,
                "realistic_crw": rl_crw,
                "delta_attack_realistic": da_r,
                "delta_crw_realistic": dc_r,
                "reduction_realistic": red_r,
            }
            compiled_rows.append(r)
    compiled_csv = pd.DataFrame(compiled_rows)
    csv_path = compiled_dir / f"{base}.csv"
    compiled_csv.to_csv(csv_path, index=False)
    print(f"  -> Compiled CSV: {csv_path.name} ({len(compiled_csv)} rows)")

    sns.set_theme(style="whitegrid")
    _compile_report(dfs, compiled_dir, base)
    _compile_heatmap(dfs, compiled_dir, base)
    _compile_bar_chart(dfs, compiled_dir, base)
    _compile_effect_size_chart(dfs, compiled_dir, base)

    print(f"\n=== Compilation complete ===")


def _compile_report(dfs: dict, output_dir: Path, base: str):
    """Summary table: one row per target party showing attack gain and CRW reduction."""
    lines = [
        "=" * 90,
        "PARTY IMPACT ANALYSIS — COMPILED REPORT (ALL PARTIES)",
        "=" * 90,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Parties compiled: {', '.join(dfs.keys())}",
        "",
    ]

    for scenario, attack_col, crw_col, label in [
        ("A", "worst_case", "worst_case_crw", "WORST-CASE (4 mixed clones × 5 questions = 20 clones)"),
        ("B", "realistic", "realistic_crw", "REALISTIC (1 easy paraphrase × 5 questions = 5 clones)"),
    ]:
        lines.append("=" * 90)
        lines.append(f"SCENARIO {scenario} — {label}")
        lines.append("=" * 90)
        lines.append(
            f"{'Target':>8s}  {'Original':>8s}  {'Attacked':>8s}  {'CRW Fix':>8s}  "
            f"{'Δ Attack':>8s}  {'Δ CRW':>8s}  {'Reduced':>7s}"
        )
        lines.append("-" * 75)

        for target in MAJOR_PARTIES:
            if target not in dfs:
                continue
            df = dfs[target]
            row = df[df["party"] == target].iloc[0]
            o = row["original"] * 100
            oc = row["original_crw"] * 100
            a = row[attack_col] * 100
            c = row[crw_col] * 100
            da = a - o
            dc = c - oc
            reduction = (1 - abs(dc) / abs(da)) * 100 if abs(da) > 0.001 else 0
            lines.append(
                f"{target:>8s}  {o:>7.2f}%  {a:>7.2f}%  {c:>7.2f}%  "
                f"{da:>+7.2f}  {dc:>+7.2f}  {reduction:>6.0f}%"
            )

        # Averages
        gains, reductions = [], []
        for target in MAJOR_PARTIES:
            if target not in dfs:
                continue
            df = dfs[target]
            row = df[df["party"] == target].iloc[0]
            o = row["original"] * 100
            oc = row["original_crw"] * 100
            a = row[attack_col] * 100
            c = row[crw_col] * 100
            da = a - o
            dc = c - oc
            gains.append(da)
            if abs(da) > 0.001:
                reductions.append((1 - abs(dc) / abs(da)) * 100)
        lines.append("-" * 75)
        lines.append(
            f"{'AVG':>8s}  {'':>8s}  {'':>8s}  {'':>8s}  "
            f"{np.mean(gains):>+7.2f}  {'':>8s}  {np.mean(reductions):>6.0f}%"
        )
        lines.append("")

    # Side-by-side comparison: worst-case vs realistic
    lines.append("=" * 90)
    lines.append("COMPARISON — WORST-CASE vs REALISTIC (target party only)")
    lines.append("=" * 90)
    lines.append(
        f"{'Target':>8s}  {'Δ Worst':>8s}  {'CRW':>8s}  {'Red.':>6s}  │  "
        f"{'Δ Real.':>8s}  {'CRW':>8s}  {'Red.':>6s}  │  "
        f"{'Abs. CRW':>9s}  {'vs Drift':>8s}"
    )
    lines.append("-" * 90)

    for target in MAJOR_PARTIES:
        if target not in dfs:
            continue
        df = dfs[target]
        row = df[df["party"] == target].iloc[0]
        o = row["original"] * 100
        oc = row["original_crw"] * 100

        # Worst-case
        da_w = row["worst_case"] * 100 - o
        dc_w = row["worst_case_crw"] * 100 - oc
        red_w = (1 - abs(dc_w) / abs(da_w)) * 100 if abs(da_w) > 0.001 else 0
        correction_w = abs(da_w) - abs(dc_w)  # absolute pp recovered by CRW

        # Realistic
        da_r = row["realistic"] * 100 - o
        dc_r = row["realistic_crw"] * 100 - oc
        red_r = (1 - abs(dc_r) / abs(da_r)) * 100 if abs(da_r) > 0.001 else 0
        correction_r = abs(da_r) - abs(dc_r)

        # Natural drift
        drift = abs(oc - o)
        ratio_r = correction_r / drift if drift > 0.001 else float("inf")

        lines.append(
            f"{target:>8s}  {da_w:>+7.2f}  {dc_w:>+7.2f}  {red_w:>5.0f}%  │  "
            f"{da_r:>+7.2f}  {dc_r:>+7.2f}  {red_r:>5.0f}%  │  "
            f"{correction_r:>8.2f}pp  {ratio_r:>7.1f}×"
        )

    lines.append("")

    # Collateral damage table: when party X attacks, how much do others lose?
    lines.append("=" * 90)
    lines.append("COLLATERAL DAMAGE — WORST-CASE (Δpp for non-target parties)")
    lines.append("=" * 90)

    header = f"{'Attacker':>8s}"
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
            delta = (row["worst_case"] - row["original"]) * 100
            if p == target:
                row_str += f"  {'[' + f'{delta:+.2f}' + ']':>8s}"
            else:
                row_str += f"  {delta:>+7.2f} "
        lines.append(row_str)

    lines.append("")

    # Significance: CRW correction vs natural drift
    lines.append("=" * 90)
    lines.append("EFFECT SIZE — CRW CORRECTION vs NATURAL DRIFT")
    lines.append("=" * 90)
    lines.append(
        "Natural drift = how much CRW changes visibility on the *original*"
    )
    lines.append(
        "questionnaire (no clones). If CRW correction >> drift, the effect"
    )
    lines.append(
        "is meaningful and not an artifact of CRW's baseline behavior."
    )
    lines.append("")
    lines.append(
        f"{'Target':>8s}  {'Drift':>8s}  {'Corr.(W)':>9s}  {'Ratio':>7s}  "
        f"{'Corr.(R)':>9s}  {'Ratio':>7s}"
    )
    lines.append("-" * 65)

    for target in MAJOR_PARTIES:
        if target not in dfs:
            continue
        df = dfs[target]
        row = df[df["party"] == target].iloc[0]
        o = row["original"] * 100
        oc = row["original_crw"] * 100
        drift = abs(oc - o)

        da_w = row["worst_case"] * 100 - o
        dc_w = row["worst_case_crw"] * 100 - oc
        corr_w = abs(da_w) - abs(dc_w)

        da_r = row["realistic"] * 100 - o
        dc_r = row["realistic_crw"] * 100 - oc
        corr_r = abs(da_r) - abs(dc_r)

        ratio_w = corr_w / drift if drift > 0.001 else float("inf")
        ratio_r = corr_r / drift if drift > 0.001 else float("inf")

        lines.append(
            f"{target:>8s}  {drift:>7.2f}pp  {corr_w:>8.2f}pp  {ratio_w:>6.1f}×  "
            f"{corr_r:>8.2f}pp  {ratio_r:>6.1f}×"
        )

    lines.append("")
    lines.append("=" * 90)

    path = output_dir / f"{base}_report.txt"
    path.write_text("\n".join(lines))
    print(f"  -> Compiled report: {path.name}")


def _compile_heatmap(dfs: dict, output_dir: Path, base: str):
    """Heatmap: rows = target (attacker), columns = affected party, cells = Δpp."""
    for scenario, attack_col, title_suffix in [
        ("worst_case", "worst_case", "Worst-case (20 clones)"),
        ("realistic", "realistic", "Realistic (5 clones)"),
    ]:
        targets = [p for p in MAJOR_PARTIES if p in dfs]
        matrix = np.zeros((len(targets), len(MAJOR_PARTIES)))

        for i, target in enumerate(targets):
            df = dfs[target]
            for j, party in enumerate(MAJOR_PARTIES):
                row = df[df["party"] == party].iloc[0]
                matrix[i, j] = (row[attack_col] - row["original"]) * 100

        fig, ax = plt.subplots(figsize=(10, 7))

        # Diverging colormap: red for gains, blue for losses
        vmax = max(abs(matrix.max()), abs(matrix.min()))
        im = ax.imshow(
            matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
        )

        ax.set_xticks(range(len(MAJOR_PARTIES)))
        ax.set_xticklabels(MAJOR_PARTIES, fontsize=11)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets, fontsize=11)
        ax.set_xlabel("Affected Party", fontsize=12)
        ax.set_ylabel("Attacker (Target Party)", fontsize=12)

        # Annotate cells
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
            f"Party Visibility Change by Attacker — {title_suffix}",
            fontsize=13, fontweight="bold", pad=12,
        )
        fig.tight_layout()

        suffix = "worst" if scenario == "worst_case" else "realistic"
        path = output_dir / f"{base}_heatmap_{suffix}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Heatmap ({title_suffix}): {path.name}")


def _compile_bar_chart(dfs: dict, output_dir: Path, base: str):
    """Combined bar chart: attack gain vs CRW-corrected for all target parties."""
    targets = [p for p in MAJOR_PARTIES if p in dfs]

    for scenario, attack_col, crw_col, title_suffix in [
        ("worst_case", "worst_case", "worst_case_crw", "Worst-case (20 clones)"),
        ("realistic", "realistic", "realistic_crw", "Realistic (5 clones)"),
    ]:
        attack_gains = []
        crw_gains = []
        for target in targets:
            df = dfs[target]
            row = df[df["party"] == target].iloc[0]
            o = row["original"]
            oc = row["original_crw"]
            attack_gains.append((row[attack_col] - o) * 100)
            crw_gains.append((row[crw_col] - oc) * 100)

        x = np.arange(len(targets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 7))

        bars_attack = ax.bar(
            x - width / 2, attack_gains, width,
            label="Attack (no CRW)", color="#F44336", alpha=0.85,
        )
        bars_crw = ax.bar(
            x + width / 2, crw_gains, width,
            label="CRW-corrected", color="#2196F3", alpha=0.85,
        )

        # Annotate reduction %
        for i, (da, dc) in enumerate(zip(attack_gains, crw_gains)):
            if abs(da) > 0.001:
                reduction = (1 - abs(dc) / abs(da)) * 100
                ax.annotate(
                    f"{reduction:.0f}%\nreduced",
                    xy=(x[i] + width / 2, dc),
                    xytext=(x[i] + width / 2, dc + 0.15 * np.sign(dc)),
                    fontsize=9, ha="center", va="bottom",
                    color="#1565C0", fontweight="bold",
                )

        # Color x-tick labels by party
        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontsize=12)
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_color(PARTY2COLOR.get(targets[i], "black"))
            label.set_fontweight("bold")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Δ Target Party Visibility (pp)", fontsize=12)
        ax.set_title(
            f"Self-Benefit from Clone Attack — {title_suffix}\n"
            f"(each party clones its top-5 questions)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=11)
        fig.tight_layout()

        suffix = "worst" if scenario == "worst_case" else "realistic"
        path = output_dir / f"{base}_bar_{suffix}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Bar chart ({title_suffix}): {path.name}")


def _compile_effect_size_chart(dfs: dict, output_dir: Path, base: str):
    """Bar chart: CRW correction vs natural drift per party (effect size)."""
    targets = [p for p in MAJOR_PARTIES if p in dfs]

    drifts = []
    corr_worst = []
    corr_real = []

    for target in targets:
        df = dfs[target]
        row = df[df["party"] == target].iloc[0]
        o = row["original"] * 100
        oc = row["original_crw"] * 100

        drift = abs(oc - o)
        drifts.append(drift)

        da_w = row["worst_case"] * 100 - o
        dc_w = row["worst_case_crw"] * 100 - oc
        corr_worst.append(abs(da_w) - abs(dc_w))

        da_r = row["realistic"] * 100 - o
        dc_r = row["realistic_crw"] * 100 - oc
        corr_real.append(abs(da_r) - abs(dc_r))

    x = np.arange(len(targets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_drift = ax.bar(
        x - width, drifts, width,
        label="Natural CRW drift (no clones)", color="#BDBDBD", edgecolor="#757575",
    )
    bars_real = ax.bar(
        x, corr_real, width,
        label="CRW correction (realistic)", color="#42A5F5", edgecolor="#1565C0",
    )
    bars_worst = ax.bar(
        x + width, corr_worst, width,
        label="CRW correction (worst-case)", color="#1565C0", edgecolor="#0D47A1",
    )

    # Annotate ratios above correction bars
    for i in range(len(targets)):
        drift = drifts[i]
        for bars, corr_val, y_offset in [
            (bars_real, corr_real[i], 0),
            (bars_worst, corr_worst[i], 0),
        ]:
            if drift > 0.001:
                ratio = corr_val / drift
                ax.text(
                    bars[i].get_x() + bars[i].get_width() / 2,
                    corr_val + 0.03,
                    f"{ratio:.0f}×",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#0D47A1",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=12)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(PARTY2COLOR.get(targets[i], "black"))
        label.set_fontweight("bold")

    ax.set_ylabel("Visibility change (pp)", fontsize=12)
    ax.set_title(
        "Effect Size: CRW Correction vs Natural Drift\n"
        "(drift = CRW's effect on original questionnaire without clones)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")

    # Add table below the chart
    cell_text = []
    col_labels = ["Drift (pp)", "Corr. realistic (pp)", "Ratio", "Corr. worst-case (pp)", "Ratio"]
    for i, target in enumerate(targets):
        drift = drifts[i]
        ratio_r = corr_real[i] / drift if drift > 0.001 else float("inf")
        ratio_w = corr_worst[i] / drift if drift > 0.001 else float("inf")
        cell_text.append([
            f"{drift:.2f}",
            f"{corr_real[i]:.2f}",
            f"{ratio_r:.1f}×",
            f"{corr_worst[i]:.2f}",
            f"{ratio_w:.1f}×",
        ])

    table = ax.table(
        cellText=cell_text,
        rowLabels=targets,
        colLabels=col_labels,
        cellLoc="center",
        loc="bottom",
        bbox=[0.0, -0.45, 1.0, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#E3F2FD")
        table[0, j].set_text_props(fontweight="bold")
    # Style row labels
    for i in range(len(targets)):
        table[i + 1, -1].set_text_props(
            fontweight="bold",
            color=PARTY2COLOR.get(targets[i], "black"),
        )
    # Highlight ratio columns
    for i in range(len(targets)):
        for j in [2, 4]:  # ratio columns
            table[i + 1, j].set_facecolor("#E8F5E9")
            table[i + 1, j].set_text_props(fontweight="bold")

    fig.subplots_adjust(bottom=0.30)
    path = output_dir / f"{base}_effect_size.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Effect size chart: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()
    config = load_config(Path(args.config))
    n = _resolve_n(config, args.n)
    name = _get_clean_name(config)

    print(f"\n=== Party Impact Analysis ({args.mode} mode) ===")
    print(f"  Config : {name}")
    print(f"  Top-k (n): {n}")
    print(f"  Clones per question (Phase 1): {N_CLONES}")
    print(f"  Major parties: {', '.join(MAJOR_PARTIES)}")

    if args.mode == "sweep":
        _run_sweep(args, config, n)
    elif args.mode == "worker":
        _run_worker(args, config, n)
    elif args.mode == "collect":
        _run_collect(args, config, n)
    elif args.mode == "phase2":
        _run_phase2(args, config, n)
    elif args.mode == "pre-paraphrases":
        _run_pre_paraphrases(args, config)
    elif args.mode == "compile":
        _run_compile(args, config)


if __name__ == "__main__":
    main()
