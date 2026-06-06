"""
Part C — Compare the behavioral L1 distance metric against the arccos answer-correlation
metric, both computed on V∪C (voters + candidates).

The two metrics capture different notions of answer redundancy:
  - BEHAVIORAL-L1: absolute answer agreement (do people give the same answers?)
  - ARCCOS|r|:     co-movement around each question's mean (blind to constant shift/scale)

This characterizes their difference: overall rank agreement, top-5 closest pairs under each,
and the pairs where they most disagree (which isolate the shift/scale-invariance distinction).

Usage:
    python -m main behavioral-compare --config configs/base_pipeline/pipeline_behavioral_l1_ZH.py
"""

import argparse
import copy
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from configs import base_constants as default_config
from vqs.config_utils import load_config
from vqs.data_loader import load_dataset
from vqs.similarity_metrics import get_calculator

BEHAVIORAL_DIST = "BEHAVIORAL-L1"
ARCCOS_DIST = "ANSWER-CORRELATION-ARCCOS"
N_TOP = 5
N_DISAGREEMENTS = 10


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare BEHAVIORAL-L1 vs ANSWER-CORRELATION-ARCCOS on V∪C"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_pipeline/pipeline_behavioral_l1_ZH.py",
        help="Base config (district/year/source). Both metrics are run on its V∪C.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args(argv)


def _config_for(base, dist: str):
    """Clone the base config for a specific answer metric, on full data (no split)."""
    cfg = copy.copy(base)
    cfg.dist = dist
    cfg.correlation_answer_source = "both"
    # Full V∪C — never a split. Distinct cache hash from any deployment run.
    cfg.train_voter_fraction = None
    cfg.split_seed = None
    return cfg


def _pair_label(row) -> str:
    return f"[{row['ID1']}↔{row['ID2']}]  {str(row['Qu1'])[:55]}  ||  {str(row['Qu2'])[:55]}"


def _build_comparison(dataset, base) -> pd.DataFrame:
    cfg_beh = _config_for(base, BEHAVIORAL_DIST)
    cfg_arc = _config_for(base, ARCCOS_DIST)

    print(f"\n--- Computing {BEHAVIORAL_DIST} on V∪C ---")
    d_beh = get_calculator(cfg_beh).calculate_distance(dataset, cfg_beh)
    print(f"\n--- Computing {ARCCOS_DIST} on V∪C ---")
    d_arc = get_calculator(cfg_arc).calculate_distance(dataset, cfg_arc)

    merged = d_beh.merge(
        d_arc[["ID1", "ID2", "Distance"]],
        on=["ID1", "ID2"],
        suffixes=("_beh", "_arc"),
    ).rename(columns={"Distance_beh": "d_beh", "Distance_arc": "d_arc"})

    # Per-metric ranks (rank 0 = closest pair)
    merged["rank_beh"] = merged["d_beh"].rank(method="min").astype(int) - 1
    merged["rank_arc"] = merged["d_arc"].rank(method="min").astype(int) - 1
    merged["rank_gap"] = (merged["rank_beh"] - merged["rank_arc"]).abs()
    return merged


def _write_report(merged: pd.DataFrame, rho: float, report_path: Path):
    top_beh = merged.nsmallest(N_TOP, "d_beh")
    top_arc = merged.nsmallest(N_TOP, "d_arc")
    overlap = set(zip(top_beh.ID1, top_beh.ID2)) & set(zip(top_arc.ID1, top_arc.ID2))
    disagree = merged.nlargest(N_DISAGREEMENTS, "rank_gap")

    lines = [
        "=== Behavioral L1  vs  ArcCos answer-correlation  (V∪C) ===",
        "",
        f"Pairs compared: {len(merged)}",
        f"Spearman(rank d_beh, rank d_arc) = {rho:.4f}",
        "  (weak/moderate ρ is expected: L1 measures absolute answer agreement, while",
        "   arccos|r| measures co-movement and is blind to constant shifts/scaling.)",
        "",
        f"--- Top {N_TOP} closest pairs by BEHAVIORAL-L1 ---",
    ]
    for _, r in top_beh.iterrows():
        lines.append(f"  d_beh={r.d_beh:.3f} (arc={r.d_arc:.3f})  {_pair_label(r)}")
    lines += ["", f"--- Top {N_TOP} closest pairs by ARCCOS|r| ---"]
    for _, r in top_arc.iterrows():
        lines.append(f"  d_arc={r.d_arc:.3f} (beh={r.d_beh:.3f})  {_pair_label(r)}")
    lines += [
        "",
        f"Top-{N_TOP} overlap between the two metrics: {len(overlap)}/{N_TOP} pairs",
        "",
        f"--- {N_DISAGREEMENTS} biggest rank disagreements ---",
        "  (close under one metric, far under the other — the shift/scale-invariance cases)",
    ]
    for _, r in disagree.iterrows():
        lines.append(
            f"  rank_beh={r.rank_beh:>4}  rank_arc={r.rank_arc:>4}  "
            f"d_beh={r.d_beh:.3f}  d_arc={r.d_arc:.3f}  {_pair_label(r)}"
        )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main(argv=None):
    args = _parse_args(argv)
    base = load_config(Path(args.config))

    dataset = load_dataset(base)
    merged = _build_comparison(dataset, base)

    rho, _ = spearmanr(merged["d_beh"], merged["d_arc"])

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_config.BEHAVIORAL_METRIC_RESULTS_DIR / "metric_comparison"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%m%d_%H%M")
    name = Path(base.__file__).stem
    base_name = f"metric_comparison_{name}_{ts}"

    # CSV: all pairs, both distances + ranks
    csv_path = out_dir / f"{base_name}.csv"
    merged.to_csv(csv_path, index=False)
    print(f"\n  -> CSV:    {csv_path}")

    # Scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(merged["d_arc"], merged["d_beh"], s=10, alpha=0.4, color="#3949AB")
    ax.set_xlabel("ArcCos answer-correlation distance  (arccos|r|)")
    ax.set_ylabel("Behavioral L1 distance  (normalized)")
    ax.set_title(f"Behavioral L1 vs ArcCos|r|  (V∪C)\nSpearman ρ = {rho:.3f}")
    fig.tight_layout()
    scatter_path = out_dir / f"{base_name}_scatter.png"
    fig.savefig(scatter_path, dpi=300)
    plt.close(fig)
    print(f"  -> Scatter: {scatter_path}")

    # Report
    report_path = out_dir / f"{base_name}_report.txt"
    _write_report(merged, rho, report_path)
    print(f"  -> Report:  {report_path}")
    print("\n=== Metric comparison complete ===")


if __name__ == "__main__":
    main()
