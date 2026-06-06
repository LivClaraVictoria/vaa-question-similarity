"""
Per-voter inter-pilot recommendation stability.

The deployment reports cross-seed agreement only as a *mean per α*. A per-seed mean is averaged
over ~9k voters, so it is trivially tight. The meaningful question is per voter:

    "Does an individual held-out voter get the same CRW recommendation regardless of which 5%
     pilot trained the weights?"

For a few reference α's this recomputes CRW recommendations for each pilot (held-out voters), then
for every pilot-pair compares each common voter's CRW ranking — Jaccard@k (top-set) and Spearman
(order). The output is the *distribution* of those per-voter agreements (violin per α), which shows
how many voters get identical advice vs a tail that shifts.

Needs the recommendation engine (the expensive step), so run via SLURM, not the login node.

Usage:
    python -m main behavioral-rec-stability --config configs/base_pipeline/pipeline_behavioral_l1_ZH.py
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

from configs import base_constants as default_config
from experiments._common import _get_clean_name, _resolve_n
from experiments.behavioral_metric._common import crw_vs_crw, split_voters
from experiments.behavioral_metric.deployment_simulation import _calibrate_alphas, _peek_distance
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.config_utils import load_config
from vqs.data_loader import load_dataset
from vqs.recommendation_engine import RecommendationEngine
from vqs.similarity_metrics import get_calculator


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-voter inter-pilot recommendation stability")
    p.add_argument("--config", type=str, default="configs/base_pipeline/pipeline_behavioral_l1_ZH.py")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--train-fraction", type=float, default=None)
    p.add_argument("--alphas", type=str, default="0.13,0.25,0.37",
                   help="Reference alphas (low/mid/high). Each adds 5 CRW recommendation passes.")
    p.add_argument("-n", "--n", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args(argv)


def _slim_crw(crw_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """voterID + top-n CRW match columns (prefixed CRW__matchID_ so crw_vs_crw can read them)."""
    match_cols = [c for c in crw_df.columns if c.startswith("_matchID_")][:n]
    slim = crw_df[match_cols].add_prefix("CRW_").copy()
    slim["voterID"] = crw_df["voterID"].values
    return slim


def _crw_recs(config, q, c, v, seeds, frac, alphas, n):
    """{alpha: {seed: slim CRW-only recommendation df}} for the held-out voters."""
    out = {a: {} for a in alphas}
    for seed in seeds:
        train, test = split_voters(v, frac, seed)
        config.train_voter_fraction, config.split_seed = frac, seed
        dist_df = get_calculator(config).calculate_distance(
            {"questions": q, "voters": train, "candidates": c}, config)
        engine = RecommendationEngine(config=config, data_map={"candidates": c, "voters": test})
        for a in alphas:
            config.alpha = a
            weights = CloneRobustReweighter(config).reweight(dist_df)
            out[a][seed] = _slim_crw(engine.run_crw(weights), n)
            print(f"  seed {seed} α={a}: CRW recs done ({len(out[a][seed])} held-out voters)")
    return out


def _plot(perv: pd.DataFrame, alphas, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (col, title) in zip(axes, [("jaccard", "Top-36 set (Jaccard)"),
                                       ("spearman", "Ranking order (Spearman)")]):
        data = [perv[np.isclose(perv["alpha"], a)][col].dropna().values for a in alphas]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor("#3949AB"); b.set_alpha(0.5)
        ax.set_xticks(range(1, len(alphas) + 1))
        ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_xlabel("Alpha (α)")
        ax.set_ylabel("Per-voter cross-pilot agreement")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Inter-pilot recommendation stability across individual held-out voters\n"
                 "(each voter compared between every pair of 5% pilots; 1.0 = identical advice)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main(argv=None):
    args = _parse_args(argv)
    config = load_config(Path(args.config))
    seeds = [int(s) for s in args.seeds.split(",")]
    frac = args.train_fraction or getattr(config, "train_voter_fraction", None) or 0.05
    alphas = [float(a) for a in args.alphas.split(",")]
    n = _resolve_n(config, args.n)

    print(f"\n=== Per-voter inter-pilot rec stability ===")
    print(f"  Seeds: {seeds}   Pilot: {frac:.0%}   α: {alphas}   top-k: {n}")

    ds = load_dataset(config)
    q, c, v = ds["questions"], ds["candidates"], ds["voters"]
    recs = _crw_recs(config, q, c, v, seeds, frac, alphas, n)

    rows = []
    for a in alphas:
        for s1, s2 in combinations(seeds, 2):
            cmp = crw_vs_crw(recs[a][s1], recs[a][s2], n)
            cmp["alpha"] = a
            cmp["pair"] = f"{s1}-{s2}"
            rows.append(cmp)
    perv = pd.concat(rows, ignore_index=True)

    out_dir = (Path(args.output_dir) if args.output_dir
               else default_config.BEHAVIORAL_METRIC_RESULTS_DIR / "rec_stability")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"rec_stability_{_get_clean_name(config)}_{datetime.now().strftime('%m%d_%H%M')}"
    perv.to_csv(out_dir / f"{base}_per_voter.csv", index=False)
    _plot(perv, alphas, out_dir / f"{base}.png")

    print("\n  Summary (median / %% voters at exactly 1.0):")
    for a in alphas:
        sub = perv[np.isclose(perv["alpha"], a)]
        for col in ("jaccard", "spearman"):
            med = sub[col].median()
            frac_one = (sub[col] >= 0.999).mean() * 100
            print(f"    α={a:g} {col:8s}: median={med:.3f}  exact-match={frac_one:5.1f}%")
    print(f"  -> {out_dir}/{base}.{{png,_per_voter.csv}}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
