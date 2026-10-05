"""
CRW drift on the unmanipulated questionnaire: how much does CRW alone change each voter's ranking
(plain vs CRW, no clones)? Reads a pipeline recommendations parquet (which holds both rankings and
a .json with n_jaccard) and writes a per-voter CSV and a summary CSV to a `crw_drift/` folder beside
`pipeline_outputs/` (i.e. <results root>/crw_drift/).

Metrics per voter: Jaccard@k and swaps in the top-k, Spearman / Kendall on the full list, share and
size of rank changes. Higher Jaccard / correlation = CRW changes less.

Usage (any canton; the parquet is canton-specific, the CRW weights are not):
    python -m experiments.explanatory.crw_drift \\
        --recs experiment_results/cantons/BE/pipeline_outputs/recommendations/cleaned/recs_*.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cross_run_analysis.analyzer import CrossRunAnalyzer


def compute_crw_drift(recs_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = json.loads(recs_path.with_suffix(".json").read_text())["n_jaccard"]
    rankings = CrossRunAnalyzer._extract_rankings(pd.read_parquet(recs_path))
    analyzer = CrossRunAnalyzer.from_n(n)

    rows = []
    for voter, plain, crw in zip(rankings.index, rankings["ranked_standard"], rankings["ranked_crw"]):
        rank = analyzer._rank_stats(plain, crw)
        pos = analyzer._position_stats(plain, crw)
        rows.append(
            {
                "voterID": voter,
                "jaccard_topk": analyzer._jaccard(plain, crw, n),
                "swaps_topk": n - len(set(plain[:n]) & set(crw[:n])),
                "spearman": rank.get("spearman", np.nan),
                "kendall": rank.get("kendall", np.nan),
                "any_rank_change": pos.get("any_change", np.nan),
                "n_changed": pos.get("n_changed", np.nan),
                "avg_pos_moved": pos.get("avg_pos_moved", np.nan),
                "max_pos_moved": pos.get("max_pos_moved", np.nan),
            }
        )
    per_voter = pd.DataFrame(rows)

    metrics = [c for c in per_voter.columns if c != "voterID"]
    summary = per_voter[metrics].agg(["mean", "median", "min", "max", "std"]).T
    summary["p10"] = per_voter[metrics].quantile(0.10)
    summary["p90"] = per_voter[metrics].quantile(0.90)
    summary.insert(0, "n_voters", len(per_voter))
    summary.insert(1, "k", n)
    summary.insert(2, "n_candidates", len(rankings.iloc[0]["ranked_standard"]))
    summary.index.name = "metric"
    return per_voter, summary.round(5)


def default_out_dir(recs_path: Path) -> Path:
    root = next(p.parent for p in recs_path.resolve().parents if p.name == "pipeline_outputs")
    return root / "crw_drift"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recs", required=True, help="Path to a recs_*.parquet (with its .json beside it).")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <results root>/crw_drift).")
    args = parser.parse_args()

    recs_path = Path(args.recs)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(recs_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_voter, summary = compute_crw_drift(recs_path)
    per_voter.to_csv(out_dir / f"{recs_path.stem}_crw_drift_per_voter.csv", index=False)
    summary.to_csv(out_dir / f"{recs_path.stem}_crw_drift_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
