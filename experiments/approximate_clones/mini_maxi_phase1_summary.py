"""
Reads the mini-maxi Phase 1 CSV and produces a master CSV enriched with
per-party corr-weighted scores, rankings, and selection indicators.

Usage:
    python scripts/mini_maxi_phase1_summary.py [--phase1-csv PATH] [--top-k 5]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARTIES = ["SP", "Green", "GLP", "Centre", "FDP", "SVP"]

DEFAULT_PHASE1_CSV = (
    "experiment_results/party_impact/mini_maxi/phase1/"
    "pipeline_e5_instruct_ZH_a03/"
    "mini_maxi_pipeline_e5_instruct_ZH_a03_0311_1538.csv"
)


def main():
    parser = argparse.ArgumentParser(description="Mini-maxi Phase 1 master summary")
    parser.add_argument(
        "--phase1-csv", type=str, default=DEFAULT_PHASE1_CSV,
        help="Path to Phase 1 CSV",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.phase1_csv)
    top_k = args.top_k

    for party in PARTIES:
        delta_col = f"delta_{party}"
        score_col = f"score_{party}"
        rank_delta_col = f"rank_delta_{party}"
        rank_corrwt_col = f"rank_corrwt_{party}"
        sel_delta_col = f"selected_delta_{party}"
        sel_corrwt_col = f"selected_corrwt_{party}"

        # Score: delta * max_abs_r, only for positive-delta questions
        positive_mask = df[delta_col] > 0
        df[score_col] = np.where(positive_mask, df[delta_col] * df["max_abs_r"], np.nan)

        # Rank by delta (among positive-delta questions, 1 = highest)
        df[rank_delta_col] = np.nan
        if positive_mask.any():
            df.loc[positive_mask, rank_delta_col] = (
                df.loc[positive_mask, delta_col].rank(ascending=False, method="min").astype(int)
            )

        # Rank by corr-weighted score (among positive-delta questions)
        df[rank_corrwt_col] = np.nan
        if positive_mask.any():
            df.loc[positive_mask, rank_corrwt_col] = (
                df.loc[positive_mask, score_col].rank(ascending=False, method="min").astype(int)
            )

        # Selection indicators
        df[sel_delta_col] = False
        if positive_mask.any():
            top_delta_idx = df.loc[positive_mask].nlargest(top_k, delta_col).index
            df.loc[top_delta_idx, sel_delta_col] = True

        df[sel_corrwt_col] = False
        if positive_mask.any():
            top_corrwt_idx = df.loc[positive_mask].nlargest(top_k, score_col).index
            df.loc[top_corrwt_idx, sel_corrwt_col] = True

    # Save master CSV
    output_path = Path(args.phase1_csv).parent / "mini_maxi_master_phase1.csv"
    df.to_csv(output_path, index=False)
    print(f"Master CSV saved to: {output_path}")
    print(f"  {len(df)} questions, {len(df.columns)} columns\n")

    # Print summary per party
    for party in PARTIES:
        score_col = f"score_{party}"
        delta_col = f"delta_{party}"
        sel_col = f"selected_corrwt_{party}"

        selected = df[df[sel_col]].sort_values(score_col, ascending=False)
        print(f"{'=' * 80}")
        print(f"TOP-{top_k} CORR-WEIGHTED QUESTIONS FOR {party}")
        print(f"{'=' * 80}")
        print(f"  {'QID':<8} {'Category':<30} {'Delta (pp)':>10} {'max|r|':>8} {'Score':>10}")
        print(f"  {'-' * 68}")
        for _, row in selected.iterrows():
            qid = int(row["question_id"])
            cat = row["category"][:29]
            delta_pp = row[delta_col] * 100
            max_r = row["max_abs_r"]
            score = row[score_col]
            print(f"  Q{qid:<7} {cat:<30} {delta_pp:>+9.2f} {max_r:>8.3f} {score:>10.6f}")
        print()


if __name__ == "__main__":
    main()
