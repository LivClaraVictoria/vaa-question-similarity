"""
Part D — Out-of-sample deployment simulation for the behavioral distance metric.

Realistic pipeline: estimate the answer-based distance from a small pilot (default 5% of
voters) + all candidates, derive CRW question weights, then apply them to the held-out
voters' recommendations. Repeated over several random pilots (seeds) to measure stability.

NO clones / noise / distortion injection here — this purely measures (a) how much CRW shifts
held-out recommendations vs the uniform-weight baseline, and (b) how stable that reweighting
is across independent pilots.

Outputs (experiment_results/behavioral_metric/deployment_sim/):
  - <base>_per_seed.csv    row per (seed, alpha): held-out impact + weight stats
  - <base>_aggregated.csv  row per alpha: across-seed mean/lo/hi impact + stability metrics
  - <base>_impact.png      Jaccard/Spearman/Kendall distortion vs alpha (across-seed band)
  - <base>_stability.png   weight + recommendation stability vs alpha
  - <base>_report.txt      distortion + stability table, downweighted-question list

Usage:
    python -m main behavioral-deploy --config configs/base_pipeline/pipeline_behavioral_l1_ZH.py
    python -m main behavioral-deploy --config ... --seeds 0,1 --alphas 0.0,0.2,0.4
"""

import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from configs import base_constants as default_config
from experiments._common import _get_clean_name, _resolve_n
from experiments.behavioral_metric._common import (
    baseline_vs_crw,
    crw_vs_crw,
    split_voters,
    summarize,
)
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.config_utils import load_config
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine
from vqs.similarity_metrics import get_calculator

N_GRID = 16  # default number of alpha values when not given explicitly


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Behavioral-metric out-of-sample deployment sim")
    p.add_argument(
        "--config",
        type=str,
        default="configs/base_pipeline/pipeline_behavioral_l1_ZH.py",
    )
    p.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated pilot seeds")
    p.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help="Pilot fraction of voters (default: config.train_voter_fraction or 0.05)",
    )
    p.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated alphas (default: auto-calibrated from the distance distribution)",
    )
    p.add_argument("-n", "--n", type=int, default=None, help="Top-k for Jaccard (default: from config)")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Alpha calibration
# ---------------------------------------------------------------------------


def _calibrate_alphas(dist_df: pd.DataFrame) -> tuple[list[float], dict]:
    """Build a default alpha grid from the off-diagonal distance distribution."""
    off = dist_df["Distance"].to_numpy(dtype=float)
    stats = {
        "min": float(off.min()),
        "median": float(np.median(off)),
        "p75": float(np.quantile(off, 0.75)),
        "max": float(off.max()),
    }
    hi = max(stats["p75"], 2 * stats["min"], 0.3)
    hi = float(min(1.0, hi))
    # Start just above 0: at exactly alpha=0 the CRW integral is empty (degenerate, all
    # weights 0). For 0 < alpha < min off-diagonal distance, weights stay uniform (no effect).
    alphas = sorted({round(x, 3) for x in np.linspace(0.01, hi, N_GRID)})
    return alphas, stats


# ---------------------------------------------------------------------------
# Per-seed computation
# ---------------------------------------------------------------------------


def _slim_recs(combined: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep voterID + the top-n baseline and CRW match columns (bounds memory for the
    across-seed comparison; preserves rank order since columns are rank-ordered)."""
    std_cols = [c for c in combined.columns if c.startswith("_matchID_")][:n]
    crw_cols = [c for c in combined.columns if c.startswith("CRW__matchID_")][:n]
    keep = (["voterID"] if "voterID" in combined.columns else []) + std_cols + crw_cols
    slim = combined[keep].copy()
    if "voterID" not in slim.columns:
        slim = slim.reset_index()  # voterID may live in the index
    return slim


def _run_seed(config, questions, candidates, voters, seed, frac, alphas, n):
    """Run the full pilot→deploy pipeline for one seed across all alphas.

    Returns:
        impact_rows  : list of per-(seed, alpha) metric dicts
        weights_by_a : {alpha: pd.Series(weight indexed by ID_question)}
        slim_by_a    : {alpha: slim combined recommendation df (for stability)}
    """
    train_v, test_v = split_voters(voters, frac, seed)
    print(f"\n=== Seed {seed}: pilot={len(train_v)} voters, held-out={len(test_v)} ===")

    # Distance from pilot voters + all candidates (cached per seed via split-aware hash)
    config.train_voter_fraction = frac
    config.split_seed = seed
    dist_df = get_calculator(config).calculate_distance(
        {"questions": questions, "voters": train_v, "candidates": candidates}, config
    )

    # Baseline (uniform weights) recommendations for the held-out voters — once per seed
    rec_engine = RecommendationEngine(
        config=config, data_map={"candidates": candidates, "voters": test_v}
    )
    baseline = rec_engine.run_baseline()

    impact_rows, weights_by_a, slim_by_a = [], {}, {}
    for alpha in alphas:
        config.alpha = alpha
        weights_df = CloneRobustReweighter(config).reweight(dist_df)
        crw = rec_engine.run_crw(weights_df)

        match_cols = [c for c in crw.columns if "match" in c or "Dist" in c]
        combined = baseline.join(crw[match_cols].add_prefix("CRW_"))

        impact = baseline_vs_crw(combined, n)
        jac, spe, ken = (summarize(impact[m]) for m in ("jaccard", "spearman", "kendall"))
        w = weights_df["Weight"]
        impact_rows.append(
            {
                "seed": seed,
                "alpha": alpha,
                "jaccard_mean": jac["mean"], "jaccard_median": jac["median"], "jaccard_p10": jac["p10"],
                "spearman_mean": spe["mean"],
                "kendall_mean": ken["mean"],
                "n_downweighted": int((w < 0.99).sum()),
                "min_weight": float(w.min()),
                "mean_weight": float(w.mean()),
            }
        )
        weights_by_a[alpha] = weights_df.set_index("ID_question")["Weight"]
        slim_by_a[alpha] = _slim_recs(combined, n)
        print(
            f"  α={alpha:.3f}  Jaccard={jac['mean']:.4f}  Spearman={spe['mean']:.4f}  "
            f"Kendall={ken['mean']:.4f}  downweighted={int((w < 0.99).sum())}"
        )

    return impact_rows, weights_by_a, slim_by_a


# ---------------------------------------------------------------------------
# Across-seed stability
# ---------------------------------------------------------------------------


def _stability(alphas, weights_per_seed, slim_per_seed, seeds, n) -> pd.DataFrame:
    """Per-alpha across-seed stability: weight-vector agreement + held-out CRW-ranking agreement."""
    rows = []
    seed_pairs = list(combinations(seeds, 2))
    for alpha in alphas:
        # Weight stability: mean pairwise Spearman of the 75-dim weight vectors + per-question CV
        wmat = pd.DataFrame({s: weights_per_seed[s][alpha] for s in seeds})  # ID_question × seeds
        w_rhos = [spearmanr(wmat[a], wmat[b]).correlation for a, b in seed_pairs]
        cv = (wmat.std(axis=1) / wmat.mean(axis=1).replace(0, np.nan)).abs()

        # Recommendation stability: mean pairwise CRW Jaccard/Spearman across seeds (common voters)
        rec_jac, rec_spe = [], []
        for a, b in seed_pairs:
            cmp = crw_vs_crw(slim_per_seed[a][alpha], slim_per_seed[b][alpha], n)
            rec_jac.append(cmp["jaccard"].mean())
            rec_spe.append(cmp["spearman"].mean())

        rows.append(
            {
                "alpha": alpha,
                "weight_stability_spearman": float(np.nanmean(w_rhos)),
                "weight_cv_mean": float(cv.mean()),
                "rec_stability_jaccard": float(np.mean(rec_jac)),
                "rec_stability_spearman": float(np.mean(rec_spe)),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_impact(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Across-seed mean / lo / hi of the per-seed impact means, one row per alpha."""
    rows = []
    for alpha, grp in per_seed.groupby("alpha"):
        row = {"alpha": alpha}
        for m in ("jaccard_mean", "spearman_mean", "kendall_mean"):
            row[f"{m}"] = float(grp[m].mean())
            row[f"{m}_lo"] = float(grp[m].min())
            row[f"{m}_hi"] = float(grp[m].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _plot_impact(agg, out_path):
    sns_colors = {"jaccard_mean": "#2196F3", "spearman_mean": "#4CAF50", "kendall_mean": "#FF9800"}
    labels = {"jaccard_mean": "Jaccard", "spearman_mean": "Spearman", "kendall_mean": "Kendall"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, c in sns_colors.items():
        # distortion = 1 - agreement
        ax.plot(agg["alpha"], 1 - agg[m], color=c, label=f"{labels[m]} distortion")
        ax.fill_between(agg["alpha"], 1 - agg[f"{m}_hi"], 1 - agg[f"{m}_lo"], color=c, alpha=0.18)
    ax.set_xlabel("Alpha (α)")
    ax.set_ylabel("Distortion  (1 − agreement, baseline vs CRW)")
    ax.set_title("Held-out recommendation distortion vs α\n(behavioral metric from 5% pilot; band = across seeds)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_stability(stab, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stab["alpha"], stab["weight_stability_spearman"], color="#6A1B9A", marker="o", label="Weight stability (pairwise Spearman)")
    ax.plot(stab["alpha"], stab["rec_stability_jaccard"], color="#00838F", marker="s", label="Held-out CRW Jaccard (cross-seed)")
    ax.plot(stab["alpha"], stab["rec_stability_spearman"], color="#00ACC1", linestyle="--", label="Held-out CRW Spearman (cross-seed)")
    ax.set_xlabel("Alpha (α)")
    ax.set_ylabel("Cross-seed agreement  (1.0 = perfectly stable)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stability of the data-driven reweighting across independent 5% pilots")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _write_report(path, base_name, seeds, frac, n, dist_stats, agg, stab,
                  weights_ref, alpha_ref, questions):
    qtext = dict(
        zip(questions["ID_question"].tolist(),
            questions.rename(columns=str.lower)["question_en"].tolist())
    )
    merged = agg.merge(stab, on="alpha")

    lines = [
        f"=== Behavioral-metric deployment simulation — {base_name} ===",
        "",
        f"Pilot fraction: {frac:.0%}   Seeds: {seeds}   Jaccard top-k: {n}",
        f"Distance distribution (off-diagonal): min={dist_stats['min']:.3f} "
        f"median={dist_stats['median']:.3f} p75={dist_stats['p75']:.3f} max={dist_stats['max']:.3f}",
        f"  (α below the min distance {dist_stats['min']:.3f} leaves all weights at 1 → no effect)",
        "",
        "--- Per-alpha held-out distortion + stability (across-seed mean) ---",
        f"{'alpha':>7} {'Jac.dist':>9} {'Spe.dist':>9} {'Ken.dist':>9} {'w-stab':>8} {'rec-stab':>9}",
    ]
    for _, r in merged.iterrows():
        lines.append(
            f"{r['alpha']:>7.3f} {1 - r['jaccard_mean']:>9.4f} {1 - r['spearman_mean']:>9.4f} "
            f"{1 - r['kendall_mean']:>9.4f} {r['weight_stability_spearman']:>8.3f} "
            f"{r['rec_stability_jaccard']:>9.3f}"
        )

    lines += ["", f"--- Most-downweighted questions at reference α={alpha_ref:.3f} (seed {seeds[0]}) ---"]
    low = weights_ref.sort_values().head(10)
    for qid, w in low.items():
        lines.append(f"  w={w:.3f}  Q{qid}: {str(qtext.get(qid, ''))[:80]}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = _parse_args(argv)
    config = load_config(Path(args.config))

    seeds = [int(s) for s in args.seeds.split(",")]
    frac = args.train_fraction or getattr(config, "train_voter_fraction", None) or 0.05
    n = _resolve_n(config, args.n)

    print("\n=== Behavioral Deployment Simulation ===")
    print(f"  Config         : {Path(config.__file__).stem}")
    print(f"  Pilot fraction : {frac:.0%}   Seeds: {seeds}   Top-k: {n}")

    # Load once; voters/candidates reused across seeds
    dataset = load_dataset(config)
    questions, candidates, voters = dataset["questions"], dataset["candidates"], dataset["voters"]

    # Calibrate alphas from the first seed's distance distribution (cached; recomputed in _run_seed)
    peek = _peek_distance(config, questions, candidates, voters, seeds[0], frac)
    default_alphas, dist_stats = _calibrate_alphas(peek)
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else default_alphas
    print(f"  Alphas         : {alphas}")

    # Run all seeds
    per_seed_rows, weights_per_seed, slim_per_seed = [], {}, {}
    for seed in seeds:
        rows, w_by_a, slim_by_a = _run_seed(config, questions, candidates, voters, seed, frac, alphas, n)
        per_seed_rows.extend(rows)
        weights_per_seed[seed] = w_by_a
        slim_per_seed[seed] = slim_by_a

    per_seed = pd.DataFrame(per_seed_rows)
    agg = _aggregate_impact(per_seed)

    if len(seeds) > 1:
        stab = _stability(alphas, weights_per_seed, slim_per_seed, seeds, n)
    else:
        stab = pd.DataFrame({"alpha": alphas, "weight_stability_spearman": np.nan,
                             "weight_cv_mean": np.nan, "rec_stability_jaccard": np.nan,
                             "rec_stability_spearman": np.nan})

    # --- Save ---
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else default_config.BEHAVIORAL_METRIC_RESULTS_DIR / "deployment_sim"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M")
    base_name = f"deployment_{_get_clean_name(config)}_{ts}"

    per_seed.to_csv(out_dir / f"{base_name}_per_seed.csv", index=False)
    agg.merge(stab, on="alpha").to_csv(out_dir / f"{base_name}_aggregated.csv", index=False)
    _plot_impact(agg, out_dir / f"{base_name}_impact.png")
    _plot_stability(stab, out_dir / f"{base_name}_stability.png")

    alpha_ref = alphas[-1]
    _write_report(
        out_dir / f"{base_name}_report.txt", base_name, seeds, frac, n, dist_stats, agg, stab,
        weights_per_seed[seeds[0]][alpha_ref], alpha_ref, questions,
    )

    print(f"\n  -> {out_dir}/{base_name}_*.{{csv,png,txt}}")
    print("\n=== Deployment simulation complete ===")


def _peek_distance(config, questions, candidates, voters, seed, frac):
    """Compute the distance matrix for one seed (used for alpha calibration; cached)."""
    train_v, _ = split_voters(voters, frac, seed)
    config.train_voter_fraction = frac
    config.split_seed = seed
    return get_calculator(config).calculate_distance(
        {"questions": questions, "voters": train_v, "candidates": candidates}, config
    )


if __name__ == "__main__":
    main()
