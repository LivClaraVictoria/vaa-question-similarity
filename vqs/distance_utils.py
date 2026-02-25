"""Utility functions for analyzing distance DataFrames with clone awareness."""

import numpy as np
import pandas as pd

from clone_pipeline.spec import CLONE_ID_BASE


def is_clone_id(qid: int) -> bool:
    """Check if a question ID is a synthetic clone (>= 9,000,000)."""
    return qid >= CLONE_ID_BASE


def get_source_from_clone(clone_qid: int) -> int:
    """Reverse-engineer the source question ID from a clone ID."""
    remaining = clone_qid - CLONE_ID_BASE
    return remaining // 1000


def classify_pair(id1: int, id2: int) -> str:
    """Classify a distance pair by clone relationship.

    Returns one of: 'real-real', 'clone-source', 'clone-clone (same source)',
    'clone-clone (diff source)', 'clone-other'
    """
    c1, c2 = is_clone_id(id1), is_clone_id(id2)
    if not c1 and not c2:
        return "real-real"
    if c1 and c2:
        src1, src2 = get_source_from_clone(id1), get_source_from_clone(id2)
        if src1 == src2:
            return "clone-clone (same source)"
        return "clone-clone (diff source)"
    clone, real = (id1, id2) if c1 else (id2, id1)
    src = get_source_from_clone(clone)
    if real == src:
        return "clone-source"
    return "clone-other"


def compute_min_non_clone_distance(dist_df: pd.DataFrame) -> dict:
    """Compute the minimum non-zero distance in the distance DataFrame.

    For identical clones, clone-source and clone-clone (same source) pairs
    have distance 0. The threshold is the minimum positive distance, which
    is where the CRW adjacency graph first changes.

    Args:
        dist_df: Distance DataFrame with columns ID1, ID2, Distance

    Returns:
        dict with keys:
            threshold: min positive distance (where adjacency first changes)
            min_real_real: min distance between two original questions
            pair_counts: dict of pair_type -> count
            per-type min/max distances
    """
    pair_types = dist_df.apply(
        lambda r: classify_pair(int(r["ID1"]), int(r["ID2"])), axis=1
    )

    result = {"pair_counts": pair_types.value_counts().to_dict()}

    for ptype in ["real-real", "clone-source", "clone-clone (same source)", "clone-other"]:
        key = ptype.replace(" ", "_").replace("(", "").replace(")", "")
        mask = pair_types == ptype
        if mask.any():
            subset = dist_df.loc[mask, "Distance"]
            result[f"min_{key}"] = float(subset.min())
            result[f"max_{key}"] = float(subset.max())
        else:
            result[f"min_{key}"] = None
            result[f"max_{key}"] = None

    # The threshold: minimum POSITIVE distance in the entire DataFrame
    positive_dists = dist_df.loc[dist_df["Distance"] > 0, "Distance"]
    result["threshold"] = float(positive_dists.min()) if len(positive_dists) > 0 else None

    # Also report min_real_real separately for clarity
    rr_mask = pair_types == "real-real"
    if rr_mask.any():
        rr_positive = dist_df.loc[rr_mask & (dist_df["Distance"] > 0), "Distance"]
        result["min_real_real"] = float(rr_positive.min()) if len(rr_positive) > 0 else None
    else:
        result["min_real_real"] = None

    return result
