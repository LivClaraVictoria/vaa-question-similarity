"""
Noise slider — robustness plot.

Interpolates between "perfect clones" (Experiment 1) and "approximate clones"
(Experiment 3) by injecting controlled noise into voter/candidate answers at
cloned questions. For each of 75 source questions we build 4 mixed clones
(easy_paraphrase, hard_paraphrase, negation_easy, negation_hard), sweep over
a small grid of noise rates λ ∈ [0, 1], and — for each λ — average the
per-voter distortion (Jaccard, Spearman, Kendall) across 20 seeds.

Three modes:
    - prepare : generate paraphrases for all 75 questions × 4 types (one-time).
    - worker  : run a single question (for SLURM array). Writes worker CSV.
    - sweep   : sequential run over all questions (local sanity-check mode).
    - collect : aggregate worker CSVs → master + aggregated CSV + plot + report.

See: /home/liweiss/.claude/plans/noise-slider-experiment-implement-vectorized-yeti.md
"""

import argparse
import hashlib
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
    FLIP_TYPES,
    PERFECT_MIX_COMPONENTS,
)
from experiments.noise_slider._perturb import admissible_set, perturb_column
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


ALPHA = 0.4
NUM_SEEDS = 20
LAMBDA_GRID = [0.00, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
N_CLONES_PER_TYPE = 1  # k=4 mixed ⇒ 1 clone per type, 4 clones per question.
RESULTS_DIR = Path("experiment_results/noise_slider/robustness")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Noise slider robustness plot: λ × seed sweep, mixed k=4 clones."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Base pipeline config (e.g. configs/base_pipeline/pipeline_e5_instruct_ZH_a04.py)",
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
        "--lambdas", type=str, default=None,
        help="Comma-separated λ values (default: plan grid)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=NUM_SEEDS,
        help=f"Seeds per λ (default: {NUM_SEEDS})",
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA,
        help=f"CRW alpha (default: {ALPHA})",
    )
    parser.add_argument(
        "-n", type=int, default=None,
        help="Override top-k for Jaccard (default: derived from config)",
    )
    parser.add_argument(
        "--subset-questions", type=int, default=None,
        help="Limit to the first N questions (smoke-test helper)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Deterministic seed derivation
# ---------------------------------------------------------------------------


def _derive_seed(q_id: int, lam: float, seed_idx: int, side: str) -> int:
    """SHA256-based deterministic seed so worker CSVs are reproducible across processes
    (Python's built-in hash() randomises per process when PYTHONHASHSEED is unset).
    """
    key = f"{int(q_id)}|{lam:.6f}|{int(seed_idx)}|{side}".encode()
    digest = hashlib.sha256(key).hexdigest()[:16]
    return int(digest, 16)


# ---------------------------------------------------------------------------
# Paraphrase loading (read-only)
# ---------------------------------------------------------------------------


def _load_paraphrases_readonly(config) -> dict:
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
# Pipeline setup (analogous to perfect_clones._setup_pipeline)
# ---------------------------------------------------------------------------


def _setup_pipeline(config):
    """Load base side, get original question IDs, load paraphrases, grab question texts."""
    print("\n--- Setting up base pipeline ---")
    base_side = _setup_side(config)

    questions_df = base_side["dataset"]["questions"]
    question_ids = sorted(
        questions_df.loc[
            questions_df["ID_question"] < 9_000_000, "ID_question"
        ].tolist()
    )

    print(f"  Questions: {len(question_ids)}")
    print(f"  Clone mix: {PERFECT_MIX_COMPONENTS} (1 clone each → 4 clones per question)")

    paraphrases = _load_paraphrases_readonly(config)

    # Verify every question has at least 1 paraphrase for all 4 types.
    missing = []
    for q_id in question_ids:
        q_id_str = str(q_id)
        for pt in PERFECT_MIX_COMPONENTS:
            existing = paraphrases.get(q_id_str, {}).get(pt, [])
            if len(existing) < N_CLONES_PER_TYPE:
                missing.append((q_id, pt, len(existing)))
    if missing:
        print(
            f"ERROR: {len(missing)} question/type combos lack enough paraphrases.\n"
            f"Run with --mode prepare first.\n"
            f"Examples: {missing[:5]}",
            file=sys.stderr,
        )
        sys.exit(1)

    text_col = _get_question_text_col(questions_df)
    question_texts = {
        q_id: questions_df.loc[questions_df["ID_question"] == q_id, text_col].iloc[0]
        for q_id in question_ids
    }

    return {
        "base_side": base_side,
        "question_ids": question_ids,
        "question_texts": question_texts,
        "paraphrases": paraphrases,
    }


def _build_specs(q_id: int) -> list[CloneSpec]:
    """One CloneSpec per PERFECT_MIX_COMPONENT, flip_answers=True for negations."""
    return [
        CloneSpec(
            source_q_id=q_id,
            clone_type=ct,
            n_clones=N_CLONES_PER_TYPE,
            flip_answers=(ct in FLIP_TYPES),
        )
        for ct in PERFECT_MIX_COMPONENTS
    ]


# ---------------------------------------------------------------------------
# Core: λ × seed loop for one question
# ---------------------------------------------------------------------------


def _run_single_question(
    q_id: int,
    config,
    pipeline: dict,
    lambda_grid: list[float],
    n_seeds: int,
    n_jaccard: int,
    alpha: float,
) -> list[dict]:
    """Run the full λ × seed grid for one source question."""
    base_side = pipeline["base_side"]
    dataset = base_side["dataset"]
    paraphrases = pipeline["paraphrases"]
    q_text = pipeline["question_texts"][q_id]

    specs = _build_specs(q_id)
    clone_ids_all: list[int] = [cid for spec in specs for cid in spec.clone_ids]

    # ---- Apply clones in-memory (λ=0 starting point) ----
    cloned_data = apply_specs(
        specs=specs,
        dataframes={
            "questions": dataset["questions"],
            "voters": dataset["voters"],
            "candidates": dataset["candidates"],
        },
        paraphrases=paraphrases,
    )

    cloned_voters_base = cloned_data["voters"]
    cloned_candidates_base = cloned_data["candidates"]
    cloned_questions = cloned_data["questions"]

    # ---- Per-question admissible set ----
    src_col = f"answer_{q_id}"
    admissible = admissible_set(
        dataset["voters"][src_col],
        dataset["candidates"][src_col],
    )
    print(f"  Admissible set for Q{q_id}: {admissible.tolist()}  (|A|={admissible.size})")

    # ---- Distances on the cloned questions (λ-independent; text-based) ----
    cloned_config = SimpleNamespace(**vars(config))
    cloned_config.clone_id = f"noise_slider_q{q_id}_mixed_k{len(clone_ids_all)}"
    cloned_config.alpha = alpha

    calculator = get_calculator(cloned_config)
    cloned_dist = calculator.calculate_distance(
        {"questions": cloned_questions,
         "voters": cloned_voters_base,
         "candidates": cloned_candidates_base},
        cloned_config,
    )

    # ---- CRW weights at α (λ-independent; distances depend only on text) ----
    cloned_reweighter = CloneRobustReweighter(cloned_config)
    cloned_weights = cloned_reweighter.reweight(cloned_dist)

    # ---- Base side recs (once — no λ effect) ----
    config.alpha = alpha
    base_rec_engine = base_side["rec_engine"]
    base_baseline = base_side["baseline"]
    base_dist = base_side["dist_df"]

    base_reweighter = CloneRobustReweighter(config)
    base_weights = base_reweighter.reweight(base_dist)
    base_crw = base_rec_engine.run_crw(base_weights)

    base_match_cols = [c for c in base_crw.columns if "match" in c or "Dist" in c]
    base_combined = base_baseline.join(base_crw[base_match_cols].add_prefix("CRW_"))

    # ---- λ × seed loop ----
    analyzer = CrossRunAnalyzer.from_n(n_jaccard)
    rows = []

    for lam in lambda_grid:
        for seed_idx in range(n_seeds):
            seed_v = _derive_seed(q_id, lam, seed_idx, "voters")
            seed_c = _derive_seed(q_id, lam, seed_idx, "candidates")
            rng_v = np.random.default_rng(seed_v)
            rng_c = np.random.default_rng(seed_c)

            # λ=0 short-circuit: no perturbation, no RNG draws.
            if lam == 0.0:
                perturbed_voters = cloned_voters_base
                perturbed_candidates = cloned_candidates_base
            else:
                perturbed_voters = cloned_voters_base.copy()
                perturbed_candidates = cloned_candidates_base.copy()
                for clone_id in clone_ids_all:
                    col = f"answer_{clone_id}"
                    if col in perturbed_voters.columns:
                        v_new, _ = perturb_column(
                            perturbed_voters[col].to_numpy(),
                            lam, admissible, rng_v,
                        )
                        perturbed_voters[col] = v_new
                    if col in perturbed_candidates.columns:
                        c_new, _ = perturb_column(
                            perturbed_candidates[col].to_numpy(),
                            lam, admissible, rng_c,
                        )
                        perturbed_candidates[col] = c_new

            # Fresh rec engine on perturbed data, run baseline + CRW.
            cloned_rec_engine = RecommendationEngine(
                config=cloned_config,
                data_map={
                    "candidates": perturbed_candidates,
                    "voters": perturbed_voters,
                    "questions": cloned_questions,
                },
            )
            cloned_baseline = cloned_rec_engine.run_baseline()
            cloned_crw = cloned_rec_engine.run_crw(cloned_weights)

            cloned_match_cols = [
                c for c in cloned_crw.columns if "match" in c or "Dist" in c
            ]
            cloned_combined = cloned_baseline.join(
                cloned_crw[cloned_match_cols].add_prefix("CRW_")
            )

            results = analyzer.analyze_from_dfs(base_combined, cloned_combined)

            # One row per voter.
            for _, r in results.iterrows():
                rows.append({
                    "question_id": int(q_id),
                    "question_text": q_text,
                    "lambda": float(lam),
                    "seed": int(seed_idx),
                    "voterID": r["voterID"],
                    "base_jaccard": r["base_jaccard"],
                    "crw_jaccard": r["crw_jaccard"],
                    "base_spearman": r["base_spearman"],
                    "crw_spearman": r["crw_spearman"],
                    "base_kendall": r["base_kendall"],
                    "crw_kendall": r["crw_kendall"],
                    "rng_seed_voters": seed_v,
                    "rng_seed_candidates": seed_c,
                })

            mean_crw_jac = results["crw_jaccard"].mean()
            mean_base_jac = results["base_jaccard"].mean()
            print(
                f"  λ={lam:.2f} seed={seed_idx:02d}  "
                f"base_jac={mean_base_jac:.4f}  crw_jac={mean_crw_jac:.4f}"
            )

    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


METRICS = ["jaccard", "spearman", "kendall"]
SIDES = ["base", "crw"]


def _aggregate(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (q_id, lambda): mean over voters-in-seed → 20 per-seed values.
    Across seeds: mean, 90%-trimmed min (drop 1 smallest), 90%-trimmed max.
    Across q_ids at fixed lambda: mean of mean / mean of lo / mean of hi.

    Returns one row per λ with {side}_{metric}_{stat} columns.
    """
    metric_cols = [f"{side}_{m}" for side in SIDES for m in METRICS]

    # Step 1: mean over voters at each (q_id, lambda, seed).
    per_seed = (
        master_df
        .groupby(["question_id", "lambda", "seed"])[metric_cols]
        .mean()
        .reset_index()
    )

    # Step 2: trimmed stats across seeds for each (q_id, lambda).
    def _trimmed(v: pd.Series) -> tuple[float, float, float]:
        """Mean / trimmed-min (drop 1) / trimmed-max (drop 1)."""
        arr = np.sort(v.to_numpy())
        if arr.size >= 3:
            lo = arr[1]
            hi = arr[-2]
        elif arr.size == 2:
            lo, hi = arr[0], arr[-1]
        else:
            lo = hi = arr[0] if arr.size else np.nan
        return float(np.mean(arr)), float(lo), float(hi)

    agg_rows = []
    for (q_id, lam), grp in per_seed.groupby(["question_id", "lambda"]):
        row = {"question_id": q_id, "lambda": lam}
        for col in metric_cols:
            mu, lo, hi = _trimmed(grp[col])
            row[f"{col}_mean"] = mu
            row[f"{col}_lo"] = lo
            row[f"{col}_hi"] = hi
        agg_rows.append(row)

    per_q_lambda = pd.DataFrame(agg_rows)

    # Step 3: collapse across questions at each lambda (simple mean).
    stat_cols = [
        f"{col}_{stat}" for col in metric_cols for stat in ("mean", "lo", "hi")
    ]
    by_lambda = (
        per_q_lambda
        .groupby("lambda")[stat_cols]
        .mean()
        .reset_index()
        .sort_values("lambda")
        .reset_index(drop=True)
    )
    return by_lambda


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(agg_df: pd.DataFrame, out_path: Path, config, n_jaccard: int) -> None:
    """Three solid CRW lines + three dashed baseline lines + 90% fills."""
    sns.set_theme(style="whitegrid")
    colors = {"jaccard": "#2196F3", "spearman": "#4CAF50", "kendall": "#FF9800"}

    fig, ax = plt.subplots(figsize=(10, 6))

    lam = agg_df["lambda"].to_numpy()

    for m in METRICS:
        c = colors[m]
        # CRW (solid, with fill)
        crw_mean = 1.0 - agg_df[f"crw_{m}_mean"].to_numpy()
        crw_lo = 1.0 - agg_df[f"crw_{m}_hi"].to_numpy()   # metric→distortion flips hi/lo
        crw_hi = 1.0 - agg_df[f"crw_{m}_lo"].to_numpy()
        ax.plot(lam, crw_mean, color=c, linewidth=2, label=f"CRW {m.capitalize()} distortion")
        ax.fill_between(lam, crw_lo, crw_hi, color=c, alpha=0.15)

        # Baseline (dashed, no fill)
        base_mean = 1.0 - agg_df[f"base_{m}_mean"].to_numpy()
        ax.plot(
            lam, base_mean, color=c, linewidth=1.5, linestyle="--",
            marker="o", markersize=4, alpha=0.75,
            label=f"{m.capitalize()} — no CRW",
        )

    ax.set_xlabel("Noise rate λ", fontsize=11)
    ax.set_ylabel("Distortion (1 − metric)", fontsize=11)
    ax.set_title(
        f"Noise Slider — Robustness\n"
        f"(mixed k=4 clones, α={ALPHA}, {NUM_SEEDS} seeds, top-{n_jaccard}, "
        f"{config.district})",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Plot: {out_path.name}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _save_report(
    master: pd.DataFrame,
    agg: pd.DataFrame,
    config,
    n_jaccard: int,
    alpha: float,
    out_path: Path,
) -> None:
    lines = [
        "=" * 90,
        "NOISE SLIDER — ROBUSTNESS REPORT",
        "=" * 90,
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Config    : {_get_clean_name(config)}",
        f"Questions : {master['question_id'].nunique()}",
        f"λ grid    : {sorted(master['lambda'].unique().tolist())}",
        f"Seeds/λ   : {master['seed'].nunique()}",
        f"Alpha     : {alpha}",
        f"Top-k (n) : {n_jaccard}",
        "",
        "Aggregated by λ (distortion = 1 − metric, lower = better):",
        "-" * 90,
    ]
    header = (
        f"{'λ':>6}  "
        f"{'base_jac':>9} {'crw_jac':>9}  "
        f"{'base_spr':>9} {'crw_spr':>9}  "
        f"{'base_ken':>9} {'crw_ken':>9}"
    )
    lines.append(header)
    lines.append("-" * 90)
    for _, row in agg.iterrows():
        lines.append(
            f"{row['lambda']:>6.2f}  "
            f"{1 - row['base_jaccard_mean']:>9.4f} "
            f"{1 - row['crw_jaccard_mean']:>9.4f}  "
            f"{1 - row['base_spearman_mean']:>9.4f} "
            f"{1 - row['crw_spearman_mean']:>9.4f}  "
            f"{1 - row['base_kendall_mean']:>9.4f} "
            f"{1 - row['crw_kendall_mean']:>9.4f}"
        )
    lines.append("=" * 90)
    out_path.write_text("\n".join(lines))
    print(f"  -> Report: {out_path.name}")


# ---------------------------------------------------------------------------
# Mode: prepare (generate paraphrases for 75 q × 4 types × 1 clone each)
# ---------------------------------------------------------------------------


def _run_prepare(config):
    print("\n--- Loading dataset for paraphrase generation ---")
    dataset = load_dataset(config)
    questions_df = dataset["questions"]

    question_ids = sorted(
        questions_df.loc[
            questions_df["ID_question"] < 9_000_000, "ID_question"
        ].tolist()
    )

    print(f"  Questions : {len(question_ids)}")
    print(f"  Types     : {PERFECT_MIX_COMPONENTS}")
    print(f"  Per type  : {N_CLONES_PER_TYPE}")

    specs = [
        CloneSpec(source_q_id=q_id, clone_type=ct, n_clones=N_CLONES_PER_TYPE)
        for q_id in question_ids
        for ct in PERFECT_MIX_COMPONENTS
    ]

    paraphrases = ensure_paraphrases(
        specs=specs,
        questions_df=questions_df,
        data_year=config.data_year,
        paraphrase_dir=config.PARAPHRASES_DIR,
    )

    ready = 0
    total = len(question_ids) * len(PERFECT_MIX_COMPONENTS)
    for q_id in question_ids:
        for ct in PERFECT_MIX_COMPONENTS:
            if len(paraphrases.get(str(q_id), {}).get(ct, [])) >= N_CLONES_PER_TYPE:
                ready += 1
    print(f"\n  Ready: {ready}/{total} (q, type) combos have ≥ {N_CLONES_PER_TYPE} paraphrases")
    if ready < total:
        print("WARNING: Some questions still lack paraphrases!", file=sys.stderr)
    else:
        print("All paraphrases ready.")
    print("\n=== Prepare Complete ===")


# ---------------------------------------------------------------------------
# Mode: sweep (sequential, all questions)
# ---------------------------------------------------------------------------


def _run_sweep(args, config, lambda_grid, n_seeds, n_jaccard, alpha):
    pipeline = _setup_pipeline(config)
    question_ids = pipeline["question_ids"]
    if args.subset_questions:
        question_ids = question_ids[: args.subset_questions]
        print(f"  [subset] Restricted to {len(question_ids)} questions.")

    all_rows = []
    for i, q_id in enumerate(question_ids):
        print(f"\n--- Question {q_id} ({i + 1}/{len(question_ids)}) ---")
        rows = _run_single_question(
            q_id, config, pipeline, lambda_grid, n_seeds, n_jaccard, alpha,
        )
        all_rows.extend(rows)

    master = pd.DataFrame(all_rows)

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_collect_outputs(master, config, n_jaccard, alpha, output_dir)
    print("\n=== Noise Slider Sweep Complete ===")


# ---------------------------------------------------------------------------
# Mode: worker (single question for SLURM array)
# ---------------------------------------------------------------------------


def _run_worker(args, config, lambda_grid, n_seeds, n_jaccard, alpha):
    task_id = args.task_id
    if task_id is None:
        print("ERROR: --task-id is required in worker mode", file=sys.stderr)
        sys.exit(1)

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "workers"
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
    out_path = sweep_dir / f"noise_slider_worker_{task_id:03d}_q{q_id}.csv"

    if out_path.exists():
        print(f"  [SKIP] Worker CSV already exists: {out_path}")
        return

    print(f"\n=== Worker: question {q_id} (task {task_id}/{len(question_ids) - 1}) ===")
    rows = _run_single_question(
        q_id, config, pipeline, lambda_grid, n_seeds, n_jaccard, alpha,
    )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n  -> Worker CSV: {out_path}")
    print("\n=== Worker Complete ===")


# ---------------------------------------------------------------------------
# Mode: collect (aggregate worker CSVs → plot + report)
# ---------------------------------------------------------------------------


def _run_collect(args, config, n_jaccard, alpha):
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else RESULTS_DIR / "workers"

    worker_files = sorted(sweep_dir.glob("noise_slider_worker_*.csv"))
    if not worker_files:
        print(f"ERROR: No worker CSVs found in {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Collect: reading {len(worker_files)} worker CSVs from {sweep_dir} ===")
    dfs = [pd.read_csv(f) for f in worker_files]
    master = pd.concat(dfs, ignore_index=True)

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_collect_outputs(master, config, n_jaccard, alpha, output_dir)
    print("\n=== Collect Complete ===")


# ---------------------------------------------------------------------------
# Shared save step for sweep and collect
# ---------------------------------------------------------------------------


def _save_collect_outputs(
    master: pd.DataFrame,
    config,
    n_jaccard: int,
    alpha: float,
    output_dir: Path,
) -> None:
    name = _get_clean_name(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{name}_{timestamp}"

    master_path = output_dir / f"master_{base}.csv"
    master.to_csv(master_path, index=False)
    print(f"  -> Master CSV: {master_path.name}  ({len(master)} rows)")

    agg = _aggregate(master)
    agg_path = output_dir / f"aggregated_{base}.csv"
    agg.to_csv(agg_path, index=False)
    print(f"  -> Aggregated CSV: {agg_path.name}")

    plot_path = output_dir / f"plot_{base}.png"
    _plot(agg, plot_path, config, n_jaccard)

    report_path = output_dir / f"report_{base}.txt"
    _save_report(master, agg, config, n_jaccard, alpha, report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = _parse_args(argv)
    config = load_config(Path(args.config))

    if args.lambdas:
        lambda_grid = sorted([float(s.strip()) for s in args.lambdas.split(",")])
    else:
        lambda_grid = LAMBDA_GRID

    n_seeds = args.n_seeds
    alpha = args.alpha
    n_jaccard = _resolve_n(config, args.n)
    name = _get_clean_name(config)

    print(f"\n=== Noise Slider ({args.mode} mode) ===")
    print(f"  Config : {name}")
    print(f"  λ grid : {lambda_grid}")
    print(f"  Seeds  : {n_seeds}")
    print(f"  Alpha  : {alpha}")
    print(f"  Top-k  : {n_jaccard}")

    if args.mode == "prepare":
        _run_prepare(config)
    elif args.mode == "sweep":
        _run_sweep(args, config, lambda_grid, n_seeds, n_jaccard, alpha)
    elif args.mode == "worker":
        _run_worker(args, config, lambda_grid, n_seeds, n_jaccard, alpha)
    elif args.mode == "collect":
        _run_collect(args, config, n_jaccard, alpha)


if __name__ == "__main__":
    main()
