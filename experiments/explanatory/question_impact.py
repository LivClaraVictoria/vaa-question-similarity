"""
Question impact sweep: identifies which questions, when cloned, cause the
most dramatic change in voter recommendations.

Supports three modes:
    - sweep (default): Run all questions sequentially in one process.
    - worker:  Compute a single question (by --task-id index). For SLURM job arrays.
    - collect: Read per-question worker CSVs from --sweep-dir, aggregate, and plot.

Clone types and CRW mode:
    --clone-types: Comma-separated clone types (default: identical).
    --n-clones:    Number of clones per question (default: 4).

    When the config specifies alpha > 0 AND a distance metric (e.g. E5-INSTRUCT),
    the script automatically runs in CRW mode: computes embedding distances, CRW
    weights, and compares both baseline and CRW recommendations. Otherwise, it
    compares baseline-only (original behavior).

Usage:
    # Baseline-only, identical clones (original behavior):
    python -m question_impact_main \\
        --config configs/base_pipeline/pipeline_e5_ZH.py

    # CRW mode, 5 clone types, 4 clones each:
    python -m question_impact_main \\
        --config configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py \\
        --clone-types easy_paraphrase,hard_paraphrase,negation_easy,negation_hard,perfect_mix \\
        --n-clones 4

    # Worker (one question, for SLURM array):
    python -m question_impact_main --mode worker --task-id 3 \\
        --config configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py \\
        --sweep-dir /path/to/sweep_dir \\
        --clone-types easy_paraphrase,hard_paraphrase --n-clones 4

    # Collect (aggregate workers + plot):
    python -m question_impact_main --mode collect \\
        --config configs/base_pipeline/pipeline_e5_instruct_ZH_a03.py \\
        --sweep-dir /path/to/sweep_dir

Outputs (saved to experiment_results/question_impact/):
    - question_impact_*.csv          — per-question metrics sorted by composite rank
    - question_impact_*_ranking.png  — impact ranking bar chart
    - question_impact_*_correlation_analysis.png — scatter plots of predictors vs impact
    - question_impact_*_corr_matrices.png — answer correlation heatmaps
    - question_impact_*_report.txt   — top-10 worst-case summary
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, kendalltau

from clone_pipeline.applicator import apply_specs
from clone_pipeline.paraphrase_generator import ensure_paraphrases
from clone_pipeline.spec import CloneSpec
from configs import base_constants as default_config
from cross_run_analysis.analyzer import CrossRunAnalyzer
from experiments._common import _get_clean_name, _get_question_text_col, _resolve_n
from vqs.config_utils import load_config
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine
from vqs.similarity_metrics import get_calculator

N_CLONES = 4
RESULTS_DIR = default_config.RESULTS_DIR / "question_impact"

# Clone types that require flipping voter/candidate answers
FLIP_TYPES = {"negation", "negation_easy", "negation_hard"}

# perfect_mix is a composite: 1 clone each of these 4 types
PERFECT_MIX_COMPONENTS = ["easy_paraphrase", "hard_paraphrase", "negation_easy", "negation_hard"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Question impact sweep: find worst-case questions to clone"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Base pipeline config (e.g. configs/base_pipeline/pipeline_e5_ZH.py)",
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
        help="Override top-k for Jaccard (default: derived from config)",
    )
    parser.add_argument(
        "--clone-types", type=str, default="identical",
        help="Comma-separated clone types (default: identical). "
             "E.g.: easy_paraphrase,hard_paraphrase,negation_easy,negation_hard,perfect_mix",
    )
    parser.add_argument(
        "--n-clones", type=int, default=None,
        help="Number of clones per question (default: 10). Overrides N_CLONES.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_crw_mode(config) -> bool:
    """Detect whether the config supports CRW (has alpha > 0 and a distance metric)."""
    alpha = getattr(config, "alpha", 0)
    dist = getattr(config, "dist", None)
    return alpha > 0 and dist is not None


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _setup_pipeline(config, clone_types: list[str] | None = None, n_clones: int | None = None):
    """Load dataset, compute base baseline recs, pre-compute question stats.

    When CRW mode is active (config has alpha > 0 and dist), also computes
    base distances, CRW weights, CRW recs, and loads paraphrases.
    """
    print("\n--- Loading dataset ---")
    dataset = load_dataset(config)

    print("\n--- Computing base baseline recommendations ---")
    rec_engine = RecommendationEngine(config=config, data_map=dataset)
    base_recs = rec_engine.run_baseline()

    # Extract ranked lists from base recs (only standard, no CRW columns)
    base_rankings = CrossRunAnalyzer._extract_rankings(base_recs)

    # Get sorted question IDs (excluding clones) for deterministic task-id mapping
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

    result = {
        "dataset": dataset,
        "base_rankings": base_rankings,
        "question_ids": question_ids,
        "questions_df": questions_df,
        "nan_info": nan_info,
        "var_info": var_info,
    }

    # CRW mode: compute base distances, CRW weights, CRW recs, load paraphrases
    crw_mode = _is_crw_mode(config)
    result["crw_mode"] = crw_mode

    if crw_mode:
        print(f"\n--- CRW mode: computing base distances ({config.dist}, alpha={config.alpha}) ---")
        calculator = get_calculator(config)
        base_dist_df = calculator.calculate_distance(dataset, config)

        reweighter = CloneRobustReweighter(config)
        base_weights = reweighter.reweight(base_dist_df)
        base_crw_recs = rec_engine.run_crw(base_weights)

        # Extract CRW ranked lists for comparison
        # run_crw returns only CRW columns; we need to combine with baseline for extraction
        match_cols = [c for c in base_crw_recs.columns if "match" in c or "Dist" in c]
        crw_prefixed = base_crw_recs[match_cols].add_prefix("CRW_")
        base_combined = base_recs.join(crw_prefixed)
        base_crw_rankings = CrossRunAnalyzer._extract_rankings(base_combined)

        result["calculator"] = calculator
        result["base_crw_rankings"] = base_crw_rankings

        # Validate perfect_mix clone count
        _n = n_clones if n_clones is not None else N_CLONES
        if "perfect_mix" in clone_types:
            n_components = len(PERFECT_MIX_COMPONENTS)
            if _n % n_components != 0:
                raise ValueError(
                    f"n_clones={_n} is not divisible by {n_components} "
                    f"(PERFECT_MIX_COMPONENTS={PERFECT_MIX_COMPONENTS}). "
                    f"For perfect_mix, n_clones must be a multiple of {n_components} "
                    f"so each component type gets an equal number of clones."
                )

        # Load paraphrases if any non-identical clone types
        if clone_types and any(ct != "identical" for ct in clone_types):
            print("\n--- Loading paraphrases ---")
            paraphrase_dir = getattr(config, "PARAPHRASES_DIR", default_config.PARAPHRASES_DIR)
            data_year = getattr(config, "data_year", 2023)
            # Build dummy specs to trigger ensure_paraphrases for all questions × types
            _n = n_clones if n_clones is not None else N_CLONES
            # Expand perfect_mix into its 4 component types for paraphrase loading
            expanded_types = set()
            for ct in clone_types:
                if ct == "perfect_mix":
                    expanded_types.update(PERFECT_MIX_COMPONENTS)
                elif ct != "identical":
                    expanded_types.add(ct)
            dummy_specs = []
            for q_id in question_ids:
                for ct in sorted(expanded_types):
                    dummy_specs.append(CloneSpec(
                        source_q_id=q_id, clone_type=ct,
                        n_clones=_n,
                        flip_answers=(ct in FLIP_TYPES),
                    ))
            result["paraphrases"] = ensure_paraphrases(
                dummy_specs, questions_df, data_year, paraphrase_dir,
            )
        else:
            result["paraphrases"] = None

    return result


# ---------------------------------------------------------------------------
# Per-question comparison
# ---------------------------------------------------------------------------


def _compare_baseline_only(
    base_rankings: pd.DataFrame,
    cloned_recs: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """Compare two baseline-only recommendation DataFrames using Jaccard, Spearman, Kendall."""
    cloned_rankings = CrossRunAnalyzer._extract_rankings(cloned_recs)

    # Join on voterID
    merged = base_rankings.join(cloned_rankings, lsuffix="_a", rsuffix="_b", how="inner")

    # Create a lightweight analyzer just for its metric methods
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
            "swaps": analyzer._swaps(jac),
            "spearman": rank.get("spearman", np.nan),
            "kendall": rank.get("kendall", np.nan),
            "any_rank_change": pos.get("any_change", np.nan),
            "n_changed": pos.get("n_changed", np.nan),
            "avg_pos_moved": pos.get("avg_pos_moved", np.nan),
            "max_pos_moved": pos.get("max_pos_moved", np.nan),
        })

    return pd.DataFrame(results)


def _compute_question_impact(
    q_id: int, config, pipeline: dict, n: int,
    clone_types: list[str] | None = None, n_clones: int | None = None,
) -> list[dict]:
    """Clone a single question, compute recs, compare with base.

    Returns a list of summary dicts (one per clone_type).
    In CRW mode, each dict includes both baseline and CRW metrics.
    """
    if clone_types is None:
        clone_types = ["identical"]
    if n_clones is None:
        n_clones = N_CLONES

    dataset = pipeline["dataset"]
    text_col = _get_question_text_col(pipeline["questions_df"])
    crw_mode = pipeline.get("crw_mode", False)
    paraphrases = pipeline.get("paraphrases")

    q_text = pipeline["questions_df"].loc[
        pipeline["questions_df"]["ID_question"] == q_id, text_col
    ].iloc[0]
    nan = pipeline["nan_info"][q_id]
    var = pipeline["var_info"][q_id]
    question_order = pipeline["question_ids"].index(q_id) + 1

    rows = []
    original_clone_id = getattr(config, "clone_id", None)

    for clone_type in clone_types:
        # perfect_mix: 1 clone each of 4 types (total = n_clones)
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

        # Compute cloned baseline recs
        cloned_engine = RecommendationEngine(config=config, data_map=cloned_data)
        cloned_recs = cloned_engine.run_baseline()

        sys.stdout.flush()

        # Baseline comparison
        per_voter_base = _compare_baseline_only(pipeline["base_rankings"], cloned_recs, n)

        row = {
            "question_id": q_id,
            "question_text": q_text,
            "question_order": question_order,
            "clone_type": clone_type,
            "n_clones": n_clones,
            "base_jaccard_mean": per_voter_base["jaccard"].mean(),
            "base_jaccard_median": per_voter_base["jaccard"].median(),
            "base_jaccard_p10": per_voter_base["jaccard"].quantile(0.1),
            "base_spearman_mean": per_voter_base["spearman"].mean(),
            "base_kendall_mean": per_voter_base["kendall"].mean(),
            "any_rank_change_pct": per_voter_base["any_rank_change"].mean(),
            "n_changed_mean": per_voter_base["n_changed"].mean(),
            "avg_pos_moved_mean": per_voter_base["avg_pos_moved"].mean(),
            "max_pos_moved_mean": per_voter_base["max_pos_moved"].mean(),
            "voter_nan_pct": nan["voter_nan_pct"],
            "candidate_nan_pct": nan["candidate_nan_pct"],
            "candidate_var": var["candidate_var"],
            "voter_var": var["voter_var"],
            "combined_var": var["combined_var"],
        }

        # CRW comparison (if applicable)
        if crw_mode:
            # Set unique clone_id to avoid cache collision for distances
            config.clone_id = f"_qsweep_{q_id}_{clone_type}_n{n_clones}"

            calculator = pipeline["calculator"]
            cloned_dist = calculator.calculate_distance(cloned_data, config)

            reweighter = CloneRobustReweighter(config)
            cloned_weights = reweighter.reweight(cloned_dist)
            cloned_crw_recs = cloned_engine.run_crw(cloned_weights)

            # Combine baseline + CRW for ranking extraction
            match_cols = [c for c in cloned_crw_recs.columns if "match" in c or "Dist" in c]
            crw_prefixed = cloned_crw_recs[match_cols].add_prefix("CRW_")
            cloned_combined = cloned_recs.join(crw_prefixed)
            cloned_crw_rankings = CrossRunAnalyzer._extract_rankings(cloned_combined)

            # Compare CRW rankings: base CRW vs cloned CRW
            base_crw_rankings = pipeline["base_crw_rankings"]
            merged = base_crw_rankings.join(
                cloned_crw_rankings, lsuffix="_a", rsuffix="_b", how="inner",
            )
            analyzer = CrossRunAnalyzer.from_n(n)
            crw_results = []
            for vid, crw_a, crw_b in zip(
                merged.index, merged["ranked_crw_a"], merged["ranked_crw_b"]
            ):
                jac = analyzer._jaccard(crw_a, crw_b, n)
                rank = analyzer._rank_stats(crw_a, crw_b)
                crw_results.append({
                    "jaccard": jac,
                    "spearman": rank.get("spearman", np.nan),
                    "kendall": rank.get("kendall", np.nan),
                })
            per_voter_crw = pd.DataFrame(crw_results)

            row["crw_jaccard_mean"] = per_voter_crw["jaccard"].mean()
            row["crw_jaccard_median"] = per_voter_crw["jaccard"].median()
            row["crw_jaccard_p10"] = per_voter_crw["jaccard"].quantile(0.1)
            row["crw_spearman_mean"] = per_voter_crw["spearman"].mean()
            row["crw_kendall_mean"] = per_voter_crw["kendall"].mean()

            # Restore original clone_id
            config.clone_id = original_clone_id

        rows.append(row)

        print(
            f"  {clone_type}: base_Jac={row['base_jaccard_mean']:.4f}"
            + (f", crw_Jac={row['crw_jaccard_mean']:.4f}" if crw_mode else "")
        )

    return rows


# ---------------------------------------------------------------------------
# Mode: sweep (sequential)
# ---------------------------------------------------------------------------


def _run_sweep(args, config, n: int, clone_types: list[str], n_clones: int):
    pipeline = _setup_pipeline(config, clone_types=clone_types, n_clones=n_clones)
    question_ids = pipeline["question_ids"]

    rows = []
    for i, q_id in enumerate(question_ids):
        print(f"\n--- Question {q_id} ({i + 1}/{len(question_ids)}) ---")
        question_rows = _compute_question_impact(
            q_id, config, pipeline, n, clone_types=clone_types, n_clones=n_clones,
        )
        rows.extend(question_rows)

    sweep_df = pd.DataFrame(rows)

    # Use collect logic for output
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_collect_outputs(sweep_df, config, n, output_dir, pipeline["dataset"])

    print("\n=== Question Impact Sweep Complete ===")


# ---------------------------------------------------------------------------
# Mode: worker (single question for SLURM array)
# ---------------------------------------------------------------------------


def _run_worker(args, config, n: int, clone_types: list[str], n_clones: int):
    task_id = args.task_id
    if task_id is None:
        print("ERROR: --task-id is required in worker mode", file=sys.stderr)
        sys.exit(1)

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "workers"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _setup_pipeline(config, clone_types=clone_types, n_clones=n_clones)
    question_ids = pipeline["question_ids"]

    if task_id < 0 or task_id >= len(question_ids):
        print(
            f"ERROR: --task-id {task_id} out of range [0, {len(question_ids) - 1}]",
            file=sys.stderr,
        )
        sys.exit(1)

    q_id = question_ids[task_id]
    out_path = sweep_dir / f"question_worker_{task_id:03d}_q{q_id}.csv"

    # Skip if already computed
    if out_path.exists():
        print(f"  [SKIP] Worker CSV already exists: {out_path}")
        return

    print(f"\n=== Worker: question {q_id} (task {task_id}/{len(question_ids) - 1}) ===")

    question_rows = _compute_question_impact(
        q_id, config, pipeline, n, clone_types=clone_types, n_clones=n_clones,
    )

    worker_df = pd.DataFrame(question_rows)
    worker_df.to_csv(out_path, index=False)

    print(f"\n  -> Worker CSV: {out_path} ({len(question_rows)} rows)")
    print(f"\n=== Worker Complete ===")


# ---------------------------------------------------------------------------
# Mode: collect (aggregate + plot + correlation analysis)
# ---------------------------------------------------------------------------


def _run_collect(args, config, n: int):
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "workers"

    worker_files = sorted(sweep_dir.glob("question_worker_*.csv"))
    if not worker_files:
        print(f"ERROR: No worker CSVs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Collect: reading {len(worker_files)} worker CSVs from {sweep_dir} ===")

    dfs = [pd.read_csv(f) for f in worker_files]
    combined = pd.concat(dfs, ignore_index=True)

    # Load dataset for correlation analysis
    print("\n--- Loading dataset for correlation analysis ---")
    dataset = load_dataset(config)

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_collect_outputs(combined, config, n, output_dir, dataset)

    print("\n=== Collect Complete ===")


# ---------------------------------------------------------------------------
# Collect output generation
# ---------------------------------------------------------------------------


def _save_collect_outputs(
    df: pd.DataFrame,
    config,
    n: int,
    output_dir: Path,
    dataset: dict,
):
    """Compute composite ranks, correlation analysis, and save all outputs."""
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = f"question_impact_{name}_{timestamp}"

    # Detect which metric columns are present (backward compat)
    has_clone_types = "clone_type" in df.columns
    crw_mode = "crw_jaccard_mean" in df.columns

    # Determine the primary Jaccard column for ranking
    jac_col = "base_jaccard_mean" if "base_jaccard_mean" in df.columns else "jaccard_mean"
    spe_col = "base_spearman_mean" if "base_spearman_mean" in df.columns else "spearman_mean"
    ken_col = "base_kendall_mean" if "base_kendall_mean" in df.columns else "kendall_mean"

    # --- Composite rank (based on baseline distortion) ---
    df["rank_jaccard"] = df[jac_col].rank()  # lower jaccard = rank 1
    df["rank_spearman"] = df[spe_col].rank()
    df["rank_kendall"] = df[ken_col].rank()
    df["composite_rank"] = (
        df["rank_jaccard"] + df["rank_spearman"] + df["rank_kendall"]
    ) / 3
    df["impact"] = 1 - df[jac_col]

    # CRW improvement metric
    if crw_mode:
        df["crw_improvement_jaccard"] = df["crw_jaccard_mean"] - df[jac_col]

    # --- Answer correlation analysis ---
    print("\n--- Computing answer correlation analysis ---")
    df_voters = dataset["voters"]
    df_candidates = dataset["candidates"]
    question_ids = sorted(df["question_id"].unique().tolist())

    voter_corr, cand_corr, redundancy = _compute_answer_correlations(
        df_voters, df_candidates, question_ids
    )

    # Merge redundancy metrics into main df
    for metric in ["voter_r2", "candidate_r2", "voter_max_abs_corr", "candidate_max_abs_corr",
                    "voter_sum_r2", "candidate_sum_r2", "voter_n_corr_neighbors", "candidate_n_corr_neighbors"]:
        df[metric] = df["question_id"].map(lambda qid, m=metric: redundancy.get(qid, {}).get(m, np.nan))

    # Sort by composite rank
    df = df.sort_values("composite_rank").reset_index(drop=True)

    # --- Save CSV ---
    csv_path = output_dir / f"{base}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> CSV: {csv_path.name}")

    # --- Plots ---
    sns.set_theme(style="whitegrid")

    n_clones = df["n_clones"].iloc[0] if "n_clones" in df.columns else N_CLONES

    if has_clone_types and crw_mode:
        _plot_crw_by_clone_type(df, output_dir, base)
        _plot_clone_type_ranking(df, output_dir, base, n_clones)
    else:
        _plot_ranking(df, output_dir, base, n_clones)

    _plot_correlation_analysis(df, output_dir, base, n_clones)
    _plot_redundancy_analysis(df, output_dir, base, n_clones)

    _plot_corr_matrices(voter_corr, cand_corr, question_ids, output_dir, base)
    _save_report(df, output_dir, base, n, n_clones=n_clones)


def _compute_answer_correlations(
    df_voters: pd.DataFrame,
    df_candidates: pd.DataFrame,
    question_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compute Pearson correlation matrices and per-question redundancy metrics.

    Returns correlation matrices and a dict mapping question_id to a dict of
    redundancy metrics: r2, max_abs_corr, sum_r2, n_corr_neighbors (for both
    voter and candidate answers).
    """
    from sklearn.linear_model import LinearRegression

    voter_ans_cols = [f"answer_{q}" for q in question_ids if f"answer_{q}" in df_voters.columns]
    cand_ans_cols = [f"answer_{q}" for q in question_ids if f"answer_{q}" in df_candidates.columns]

    voter_corr = df_voters[voter_ans_cols].corr()
    cand_corr = df_candidates[cand_ans_cols].corr()

    CORR_NEIGHBOR_THRESHOLD = 0.3

    redundancy = {}

    for q_id in question_ids:
        col = f"answer_{q_id}"
        metrics = {}

        for prefix, corr_mat, df_ans, ans_cols in [
            ("voter", voter_corr, df_voters, voter_ans_cols),
            ("candidate", cand_corr, df_candidates, cand_ans_cols),
        ]:
            if col not in corr_mat.columns:
                metrics[f"{prefix}_r2"] = np.nan
                metrics[f"{prefix}_max_abs_corr"] = np.nan
                metrics[f"{prefix}_sum_r2"] = np.nan
                metrics[f"{prefix}_n_corr_neighbors"] = np.nan
                continue

            abs_corrs = corr_mat[col].drop(col, errors="ignore").abs()

            # Max |r|
            metrics[f"{prefix}_max_abs_corr"] = abs_corrs.max() if len(abs_corrs) > 0 else np.nan

            # Σr²
            metrics[f"{prefix}_sum_r2"] = (abs_corrs**2).sum() if len(abs_corrs) > 0 else np.nan

            # Count |r| > threshold
            metrics[f"{prefix}_n_corr_neighbors"] = int((abs_corrs > CORR_NEIGHBOR_THRESHOLD).sum())

            # R² from regressing this question on all others
            other_cols = [c for c in ans_cols if c != col]
            if other_cols:
                subset = df_ans[[col] + other_cols].dropna()
                if len(subset) > len(other_cols) + 1:
                    y = subset[col].values
                    X = subset[other_cols].values
                    model = LinearRegression().fit(X, y)
                    metrics[f"{prefix}_r2"] = model.score(X, y)
                else:
                    metrics[f"{prefix}_r2"] = np.nan
            else:
                metrics[f"{prefix}_r2"] = np.nan

        redundancy[q_id] = metrics

    return voter_corr, cand_corr, redundancy


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_ranking(df: pd.DataFrame, output_dir: Path, base: str, n_clones: int = N_CLONES):
    """Horizontal bar chart: questions ranked by composite impact."""
    jac_col = "base_jaccard_mean" if "base_jaccard_mean" in df.columns else "jaccard_mean"
    spe_col = "base_spearman_mean" if "base_spearman_mean" in df.columns else "spearman_mean"
    ken_col = "base_kendall_mean" if "base_kendall_mean" in df.columns else "kendall_mean"

    df_sorted = df.sort_values("composite_rank").head(30)  # top 30

    fig, ax = plt.subplots(figsize=(14, max(8, len(df_sorted) * 0.35)))

    y = np.arange(len(df_sorted))
    height = 0.25

    # Plot three metrics side by side (as "distortion" = 1 - metric)
    ax.barh(y - height, 1 - df_sorted[jac_col], height, label="1 - Jaccard", color="#E53935")
    ax.barh(y, 1 - df_sorted[spe_col], height, label="1 - Spearman", color="#1E88E5")
    ax.barh(y + height, 1 - df_sorted[ken_col], height, label="1 - Kendall", color="#43A047")

    # Labels: question ID + truncated text
    labels = []
    for _, row in df_sorted.iterrows():
        text = str(row["question_text"])[:60]
        labels.append(f"Q{int(row['question_id'])}  {text}")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Distortion (higher = more impact from cloning)")
    clone_type = df["clone_type"].iloc[0] if "clone_type" in df.columns else "identical"
    ax.set_title(f"Question Impact Ranking (top {len(df_sorted)}, {n_clones} {clone_type} clones)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = output_dir / f"{base}_ranking.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Ranking plot: {path.name}")


def _scatter_panel(ax, df: pd.DataFrame, col: str, label: str):
    """Draw a single scatter panel: col vs impact with Spearman annotation."""
    if col not in df.columns or df[col].isna().all():
        ax.set_visible(False)
        return

    x = df[col].values
    y = df["impact"].values
    mask = ~(np.isnan(x) | np.isnan(y))

    ax.scatter(x[mask], y[mask], alpha=0.6, s=40, color="#1565C0")

    if mask.sum() >= 3:
        rho, p = spearmanr(x[mask], y[mask])
        ax.set_title(f"{label}\n(Spearman r={rho:.3f}, p={p:.3f})", fontsize=10)
    else:
        ax.set_title(label, fontsize=10)

    ax.set_xlabel(label)
    ax.set_ylabel("Impact (1 - Jaccard mean)")

    top3 = df.nlargest(3, "impact")
    for _, row in top3.iterrows():
        if pd.notna(row[col]):
            ax.annotate(
                f"Q{int(row['question_id'])}",
                (row[col], row["impact"]),
                fontsize=7, alpha=0.8,
                xytext=(5, 5), textcoords="offset points",
            )


def _plot_correlation_analysis(df: pd.DataFrame, output_dir: Path, base: str, n_clones: int = N_CLONES):
    """Plot A: question property predictors vs impact (2x3 grid)."""
    clone_type = df["clone_type"].iloc[0] if "clone_type" in df.columns else "identical"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    scatter_configs = [
        ("combined_var", "Combined Variance", axes[0, 0]),
        ("voter_var", "Voter Variance", axes[0, 1]),
        ("candidate_var", "Candidate Variance", axes[0, 2]),
        ("voter_nan_pct", "Voter NaN %\n(fraction who skipped this question)", axes[1, 0]),
        ("question_order", "Question Position in Survey\n(1 = first question)", axes[1, 1]),
    ]

    for col, label, ax in scatter_configs:
        _scatter_panel(ax, df, col, label)

    axes[1, 2].set_visible(False)

    fig.suptitle(
        f"Predictors of Clone Impact ({n_clones} {clone_type} clones)\n"
        f"Y-axis: impact = 1 − mean Jaccard (higher = more disruption to recommendations)",
        fontsize=12,
    )
    fig.tight_layout()

    path = output_dir / f"{base}_correlation_analysis.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Correlation analysis: {path.name}")


def _plot_redundancy_analysis(df: pd.DataFrame, output_dir: Path, base: str, n_clones: int = N_CLONES):
    """Plot B: redundancy metrics vs impact (2x2 grid)."""
    clone_type = df["clone_type"].iloc[0] if "clone_type" in df.columns else "identical"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    scatter_configs = [
        ("voter_r2", "Voter R²\n(OLS on all other questions)", axes[0, 0]),
        ("voter_max_abs_corr", "Voter Max |r|\n(nearest neighbor correlation)", axes[0, 1]),
        ("voter_sum_r2", "Voter Σr²\n(total shared variance)", axes[1, 0]),
        ("voter_n_corr_neighbors", "Voter Count |r| > 0.3\n(number of correlated neighbors)", axes[1, 1]),
    ]

    for col, label, ax in scatter_configs:
        _scatter_panel(ax, df, col, label)

    fig.suptitle(
        f"Redundancy Metrics vs Clone Impact ({n_clones} {clone_type} clones)\n"
        f"Y-axis: impact = 1 − mean Jaccard (higher = more disruption to recommendations)",
        fontsize=12,
    )
    fig.tight_layout()

    path = output_dir / f"{base}_redundancy_analysis.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> Redundancy analysis: {path.name}")


def _plot_corr_matrices(
    voter_corr: pd.DataFrame,
    cand_corr: pd.DataFrame,
    question_ids: list[int],
    output_dir: Path,
    base: str,
):
    """Heatmaps of voter and candidate answer correlation matrices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Shorten column labels to question IDs
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


def _plot_crw_by_clone_type(df: pd.DataFrame, output_dir: Path, base: str):
    """Grouped bar chart: mean CRW Jaccard per clone type, with baseline reference."""
    clone_types = df["clone_type"].unique()
    summary = df.groupby("clone_type").agg(
        base_jaccard_mean=("base_jaccard_mean", "mean"),
        crw_jaccard_mean=("crw_jaccard_mean", "mean"),
        crw_jaccard_std=("crw_jaccard_mean", "std"),
        base_jaccard_std=("base_jaccard_mean", "std"),
    ).reindex([ct for ct in [
        "easy_paraphrase", "negation_easy", "perfect_mix",
        "hard_paraphrase", "negation_hard",
    ] if ct in clone_types])

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(summary))
    width = 0.35

    ax.bar(
        x - width / 2, summary["base_jaccard_mean"], width,
        label="Baseline (no CRW)", color="#E53935", alpha=0.8,
        yerr=summary["base_jaccard_std"], capsize=3,
    )
    ax.bar(
        x + width / 2, summary["crw_jaccard_mean"], width,
        label="CRW-corrected", color="#1565C0", alpha=0.8,
        yerr=summary["crw_jaccard_std"], capsize=3,
    )

    # Value labels
    for i, (_, row) in enumerate(summary.iterrows()):
        improvement = row["crw_jaccard_mean"] - row["base_jaccard_mean"]
        ax.text(
            i + width / 2, row["crw_jaccard_mean"] + row["crw_jaccard_std"] + 0.01,
            f"+{improvement:.3f}", ha="center", fontsize=9, fontweight="bold", color="#1565C0",
        )

    clone_display = {
        "easy_paraphrase": "Easy Para.", "hard_paraphrase": "Hard Para.",
        "negation_easy": "Neg. Easy", "negation_hard": "Neg. Hard",
        "perfect_mix": "Perfect Mix", "identical": "Identical",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([clone_display.get(ct, ct) for ct in summary.index], fontsize=11)
    ax.set_ylabel("Jaccard Similarity (mean across 75 questions)", fontsize=11)
    ax.set_title(
        "CRW Correction by Clone Type\n"
        "(higher = less distortion; error bars = std across questions)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = output_dir / f"{base}_crw_by_clone_type.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> CRW by clone type: {path.name}")


def _plot_clone_type_ranking(df: pd.DataFrame, output_dir: Path, base: str, n_clones: int):
    """For each clone type, bar chart of top-20 most impacted questions (CRW Jaccard)."""
    clone_types = sorted(df["clone_type"].unique())

    for clone_type in clone_types:
        ct_df = df[df["clone_type"] == clone_type].sort_values("crw_jaccard_mean").head(20)

        fig, ax = plt.subplots(figsize=(14, max(6, len(ct_df) * 0.35)))
        y = np.arange(len(ct_df))
        height = 0.35

        ax.barh(y - height / 2, ct_df["base_jaccard_mean"], height,
                label="Baseline", color="#E53935", alpha=0.7)
        ax.barh(y + height / 2, ct_df["crw_jaccard_mean"], height,
                label="CRW", color="#1565C0", alpha=0.7)

        labels = [
            f"Q{int(row['question_id'])}  {str(row['question_text'])[:50]}"
            for _, row in ct_df.iterrows()
        ]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Jaccard Similarity (higher = less distortion)")

        clone_display = {
            "easy_paraphrase": "Easy Paraphrase", "hard_paraphrase": "Hard Paraphrase",
            "negation_easy": "Negation + Easy Para.", "negation_hard": "Negation + Hard Para.",
            "perfect_mix": "Perfect Mix", "identical": "Identical",
        }
        ax.set_title(
            f"{clone_display.get(clone_type, clone_type)} - Top 20 Most Impacted Questions "
            f"({n_clones} clones)",
        )
        ax.legend(loc="lower right")
        fig.tight_layout()

        path = output_dir / f"{base}_ranking_{clone_type}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"  -> Ranking ({clone_type}): {path.name}")


def _save_report(df: pd.DataFrame, output_dir: Path, base: str, n: int, top_n: int = 10, n_clones: int = N_CLONES):
    """Save human-readable worst-case report."""
    jac_col = "base_jaccard_mean" if "base_jaccard_mean" in df.columns else "jaccard_mean"
    spe_col = "base_spearman_mean" if "base_spearman_mean" in df.columns else "spearman_mean"
    ken_col = "base_kendall_mean" if "base_kendall_mean" in df.columns else "kendall_mean"
    has_clone_types = "clone_type" in df.columns
    crw_mode = "crw_jaccard_mean" in df.columns

    clone_types_str = ", ".join(sorted(df["clone_type"].unique())) if has_clone_types else "identical"

    lines = [
        "=" * 80,
        "QUESTION IMPACT SWEEP REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Clone types: {clone_types_str}",
        f"Clones per question: {n_clones}",
        f"Jaccard top-k (n): {n}",
        f"Questions tested: {df['question_id'].nunique()}",
        f"Total rows: {len(df)}",
        f"CRW mode: {'yes' if crw_mode else 'no'}",
    ]

    if crw_mode and has_clone_types:
        lines.extend(["", "CRW CORRECTION SUMMARY BY CLONE TYPE:", "-" * 80])
        summary = df.groupby("clone_type").agg(
            base_jac=(jac_col, "mean"), crw_jac=("crw_jaccard_mean", "mean"),
            base_spe=(spe_col, "mean"), crw_spe=("crw_spearman_mean", "mean"),
        )
        lines.append(f"  {'Clone Type':>20s}  {'Base Jac':>8s}  {'CRW Jac':>8s}  "
                     f"{'Improve':>8s}  {'Base Spe':>8s}  {'CRW Spe':>8s}")
        for ct, row in summary.iterrows():
            imp = row["crw_jac"] - row["base_jac"]
            lines.append(
                f"  {ct:>20s}  {row['base_jac']:>8.4f}  {row['crw_jac']:>8.4f}  "
                f"{imp:>+8.4f}  {row['base_spe']:>8.4f}  {row['crw_spe']:>8.4f}"
            )
    else:
        lines.extend([
            "",
            f"TOP {top_n} WORST-CASE QUESTIONS (by composite rank):",
            "-" * 80,
            f"{'Rank':>4}  {'QID':>6}  {'Jaccard':>8}  {'Spearman':>8}  {'Kendall':>8}  "
            f"{'V.NaN%':>6}  {'C.NaN%':>6}  {'CombVar':>8}  Text",
            "-" * 80,
        ])
        for i, (_, row) in enumerate(df.head(top_n).iterrows()):
            text = str(row["question_text"])[:50]
            lines.append(
                f"{i + 1:>4}  {int(row['question_id']):>6}  "
                f"{row[jac_col]:>8.4f}  {row[spe_col]:>8.4f}  "
                f"{row[ken_col]:>8.4f}  "
                f"{row['voter_nan_pct'] * 100:>5.1f}%  {row['candidate_nan_pct'] * 100:>5.1f}%  "
                f"{row['combined_var']:>8.1f}  {text}"
            )

    lines.extend([
        "",
        "=" * 80,
        "FULL STATISTICS:",
        f"  Mean Jaccard (baseline):   {df[jac_col].mean():.4f}",
        f"  Median Jaccard (baseline): {df[jac_col].median():.4f}",
        f"  Min Jaccard (baseline):    {df[jac_col].min():.4f}",
        f"  Max Jaccard (baseline):    {df[jac_col].max():.4f}",
    ])

    if crw_mode:
        lines.extend([
            f"  Mean Jaccard (CRW):   {df['crw_jaccard_mean'].mean():.4f}",
            f"  Median Jaccard (CRW): {df['crw_jaccard_mean'].median():.4f}",
            f"  Mean improvement:     {(df['crw_jaccard_mean'] - df[jac_col]).mean():.4f}",
        ])

    lines.append("=" * 80)

    path = output_dir / f"{base}_report.txt"
    path.write_text("\n".join(lines))
    print(f"  -> Report: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = _parse_args()
    config = load_config(Path(args.config))
    n = _resolve_n(config, args.n)
    name = _get_clean_name(config)

    clone_types = [ct.strip() for ct in args.clone_types.split(",")]
    n_clones = args.n_clones if args.n_clones is not None else N_CLONES
    crw_mode = _is_crw_mode(config)

    print(f"\n=== Question Impact Sweep ({args.mode} mode) ===")
    print(f"  Config : {name}")
    print(f"  Top-k (n): {n}")
    print(f"  Clone types: {clone_types}")
    print(f"  Clones per question: {n_clones}")
    print(f"  CRW mode: {crw_mode} (alpha={getattr(config, 'alpha', 0)}, dist={getattr(config, 'dist', None)})")

    if args.mode == "sweep":
        _run_sweep(args, config, n, clone_types, n_clones)
    elif args.mode == "worker":
        _run_worker(args, config, n, clone_types, n_clones)
    elif args.mode == "collect":
        _run_collect(args, config, n)


if __name__ == "__main__":
    main()
