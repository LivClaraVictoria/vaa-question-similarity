"""
Shared helpers for the behavioral-metric experiments:
  - metric_comparison.py   (Part C: compare two answer-based distance matrices)
  - deployment_simulation.py (Part D: out-of-sample 5%-voter deployment + stability)

Recommendation comparisons reuse CrossRunAnalyzer's ranking extraction and metric helpers
rather than reimplementing Jaccard / Spearman / Kendall.
"""

import numpy as np
import pandas as pd

from cross_run_analysis.analyzer import CrossRunAnalyzer


# ---------------------------------------------------------------------------
# Voter split
# ---------------------------------------------------------------------------


def split_voters(voters: pd.DataFrame, train_fraction: float, seed: int):
    """Random, reproducible voter split.

    Returns (train_voters, test_voters) as copies. `train` (the pilot sample) estimates the
    distance metric; `test` (held out) is where recommendations are evaluated.
    """
    n = len(voters)
    perm = np.random.RandomState(seed).permutation(n)
    n_train = max(1, int(round(n * train_fraction)))
    return voters.iloc[perm[:n_train]].copy(), voters.iloc[perm[n_train:]].copy()


# ---------------------------------------------------------------------------
# Per-voter recommendation comparisons (reuse CrossRunAnalyzer)
# ---------------------------------------------------------------------------


def baseline_vs_crw(combined_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per-voter Jaccard@n / Spearman / Kendall between the baseline (`_matchID_`) and CRW
    (`CRW__matchID_`) rankings *inside one* combined recommendation DataFrame.

    Returns columns: voterID, jaccard, spearman, kendall.
    """
    az = CrossRunAnalyzer.from_n(n)
    ranked = CrossRunAnalyzer._extract_rankings(combined_df)

    rows = []
    for vid, std, crw in zip(ranked.index, ranked["ranked_standard"], ranked["ranked_crw"]):
        rank = az._rank_stats(std, crw)
        rows.append(
            {
                "voterID": vid,
                "jaccard": az._jaccard(std, crw, n),
                "spearman": rank.get("spearman", np.nan),
                "kendall": rank.get("kendall", np.nan),
            }
        )
    return pd.DataFrame(rows)


def crw_vs_crw(combined_a: pd.DataFrame, combined_b: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per-voter Jaccard@n / Spearman / Kendall between the CRW rankings of two runs
    (e.g. two seeds), over the voters present in both. Reuses `analyze_from_dfs`.

    Returns columns: voterID, jaccard, spearman, kendall.
    """
    az = CrossRunAnalyzer.from_n(n)
    res = az.analyze_from_dfs(combined_a, combined_b)
    return res[["voterID", "crw_jaccard", "crw_spearman", "crw_kendall"]].rename(
        columns={
            "crw_jaccard": "jaccard",
            "crw_spearman": "spearman",
            "crw_kendall": "kendall",
        }
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize(series: pd.Series) -> dict:
    """mean / median / 10th-percentile summary of a per-voter metric series."""
    s = pd.Series(series).dropna()
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p10": float(s.quantile(0.10)),
    }
