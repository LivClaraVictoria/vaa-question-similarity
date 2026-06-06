"""
Per-question weight-stability plot for the behavioral-metric deployment.

For each meaningfully-downweighted question, plot its CRW weight as a function of alpha, with a
shaded across-seed (min–max) band. Tight bands => the data-driven reweighting is pilot-independent.

CRW weights recompute from the per-seed distance matrices cached by the deployment run (no
recommendation pass needed), so this runs locally in seconds.

Usage:
    python -m main behavioral-weight-stability --config configs/base_pipeline/pipeline_behavioral_l1_ZH.py
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs import base_constants as default_config
from experiments._common import _get_clean_name
from experiments.behavioral_metric._common import split_voters  # noqa: F401 (reused via _peek_distance)
from experiments.behavioral_metric.deployment_simulation import _calibrate_alphas, _peek_distance
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.config_utils import load_config
from vqs.data_loader import load_dataset

MAX_QUESTIONS = 12  # cap the number of lines so the figure stays readable


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-question CRW weight stability across pilots")
    p.add_argument("--config", type=str, default="configs/base_pipeline/pipeline_behavioral_l1_ZH.py")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--train-fraction", type=float, default=None)
    p.add_argument("--alphas", type=str, default=None, help="Default: same auto-calibration as deploy")
    p.add_argument("--threshold", type=float, default=0.97, help="Show questions whose weight dips below this")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args(argv)


def _collect_weights(config, q, c, v, seeds, frac, alphas) -> pd.DataFrame:
    """Long table seed × alpha × question -> CRW weight (distances are cache hits)."""
    rows = []
    for seed in seeds:
        dist_df = _peek_distance(config, q, c, v, seed, frac)  # cached per (frac, seed)
        for alpha in alphas:
            config.alpha = alpha
            w = CloneRobustReweighter(config).reweight(dist_df)
            for qid, weight in zip(w["ID_question"], w["Weight"]):
                rows.append({"seed": seed, "alpha": alpha, "ID_question": int(qid), "weight": float(weight)})
    return pd.DataFrame(rows)


def _select_questions(long_df: pd.DataFrame, threshold: float, max_q: int) -> list[int]:
    """Questions whose across-seed mean weight dips below `threshold` at any alpha; lowest first."""
    mean_by = long_df.groupby(["ID_question", "alpha"])["weight"].mean()
    min_mean = mean_by.groupby("ID_question").min().sort_values()
    moved = min_mean[min_mean < threshold]
    return list(moved.index[:max_q])


def _plot(long_df, qids, qtext, min_dist, out_path):
    alphas = sorted(long_df["alpha"].unique())
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(qids), 1)))

    fig, ax = plt.subplots(figsize=(11, 7))
    for color, qid in zip(colors, qids):
        g = long_df[long_df["ID_question"] == qid].groupby("alpha")["weight"]
        mean = g.mean().reindex(alphas)
        lo = g.min().reindex(alphas)
        hi = g.max().reindex(alphas)
        ax.plot(alphas, mean.values, color=color, lw=1.8,
                label=f"Q{qid}: {str(qtext.get(qid, ''))[:32]}")
        ax.fill_between(alphas, lo.values, hi.values, color=color, alpha=0.15)

    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.axvline(min_dist, color="black", lw=0.9, ls="--", alpha=0.6,
               label=f"min distance ({min_dist:.3f}) — CRW activates")
    ax.set_xlabel("Alpha (α)")
    ax.set_ylabel("CRW weight  (1.0 = unchanged)")
    ax.set_title("Per-question CRW weight vs α\nline = across-seed mean, band = min–max over pilots")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main(argv=None):
    args = _parse_args(argv)
    config = load_config(Path(args.config))
    seeds = [int(s) for s in args.seeds.split(",")]
    frac = args.train_fraction or getattr(config, "train_voter_fraction", None) or 0.05

    print("\n=== Per-question weight-stability ===")
    print(f"  Config: {Path(config.__file__).stem}   Seeds: {seeds}   Pilot: {frac:.0%}")

    ds = load_dataset(config)
    q, c, v = ds["questions"], ds["candidates"], ds["voters"]

    peek = _peek_distance(config, q, c, v, seeds[0], frac)
    default_alphas, stats = _calibrate_alphas(peek)
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else default_alphas

    long_df = _collect_weights(config, q, c, v, seeds, frac, alphas)
    qids = _select_questions(long_df, args.threshold, MAX_QUESTIONS)
    qtext = dict(zip(q["ID_question"].tolist(),
                     q.rename(columns=str.lower)["question_en"].tolist()))

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else default_config.BEHAVIORAL_METRIC_RESULTS_DIR / "weight_stability"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"weight_stability_{_get_clean_name(config)}_{datetime.now().strftime('%m%d_%H%M')}"

    long_df.to_csv(out_dir / f"{base}.csv", index=False)
    _plot(long_df, qids, qtext, stats["min"], out_dir / f"{base}.png")

    print(f"  Selected {len(qids)} questions (weight < {args.threshold} at some α)")
    print(f"  -> {out_dir}/{base}.{{png,csv}}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
