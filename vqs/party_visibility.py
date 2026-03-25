"""
Party visibility computation utilities shared across party impact experiments.
"""

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dependencies" / "rsfp"))
from rsfp.constants import BIG_PARTIES, PARTIES_LEFT_TO_RIGHT, PARTY2COLOR  # noqa: F401

# Parties to analyse (left-to-right order for plots, filtered to big parties)
MAJOR_PARTIES = [p for p in PARTIES_LEFT_TO_RIGHT if p in BIG_PARTIES]


def _build_candidate_party_map(df_candidates: pd.DataFrame) -> dict[int, str]:
    """Build candidate ID -> party name mapping."""
    return df_candidates.set_index("ID_candidate")["_party"].to_dict()


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
