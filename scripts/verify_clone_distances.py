"""
Verify that identical clones have near-zero distances in cached distance results.

Usage:
    python -m scripts.verify_clone_distances --clone-id identical_combinedvar_n10
    python -m scripts.verify_clone_distances --clone-id identical_q32214_n10
    python -m scripts.verify_clone_distances  # runs both
"""

import argparse
import json
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np

from clone_pipeline.spec import CLONE_ID_BASE, CLONE_TYPE_OFFSETS


# --- Config for hash computation ---
DISTANCE_HASH_PARAMS = ["data_year", "dist", "data_choice", "clone_id", "embedding_instruction", "embedding_task"]
CACHE_DIR = Path("cache/distance_calculations")
EXPERIMENT_DIR = Path("experiment_results/distance_metric/cloned_results")


def compute_distance_hash(clone_id: str) -> str:
    params = {
        "data_year": 2023,
        "dist": "E5",
        "data_choice": "cloned",
        "clone_id": clone_id,
        "embedding_instruction": None,
        "embedding_task": None,
    }
    param_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode()).hexdigest()[:12]


def load_distance_df(clone_id: str) -> pd.DataFrame | None:
    """Try loading from cache first, then experiment results."""
    h = compute_distance_hash(clone_id)

    # Try cache
    cache_matches = list(CACHE_DIR.glob(f"*{h}.parquet"))
    if cache_matches:
        path = cache_matches[0]
        print(f"Loaded from cache: {path.name}")
        return pd.read_parquet(path)

    # Try experiment results
    exp_dir = EXPERIMENT_DIR / clone_id
    if exp_dir.exists():
        exp_matches = list(exp_dir.glob(f"*{h}.parquet"))
        if exp_matches:
            path = exp_matches[0]
            print(f"Loaded from experiment results: {path}")
            return pd.read_parquet(path)

    print(f"No distance data found for clone_id='{clone_id}' (hash={h})")
    print(f"  Checked: {CACHE_DIR}/*{h}.parquet")
    print(f"  Checked: {exp_dir}/*{h}.parquet")
    return None


def load_clone_metadata(clone_id: str) -> dict | None:
    meta_path = Path("data/cloned") / clone_id / "clone_metadata.json"
    if not meta_path.exists():
        print(f"No metadata at {meta_path}")
        return None
    with open(meta_path) as f:
        return json.load(f)


def is_clone_id(qid: int) -> bool:
    return qid >= CLONE_ID_BASE


def get_source_from_clone(clone_qid: int) -> int:
    """Reverse-engineer the source question ID from a clone ID."""
    remaining = clone_qid - CLONE_ID_BASE
    return remaining // 1000


def verify_clone_distances(clone_id: str) -> None:
    print(f"\n{'='*70}")
    print(f"Verifying clone distances for: {clone_id}")
    print(f"{'='*70}")

    # Load metadata
    meta = load_clone_metadata(clone_id)
    if meta:
        specs = meta["specs"]
        print(f"\nClone metadata:")
        for s in specs:
            print(f"  Source: Q{s['source_q_id']}, type: {s['clone_type']}, "
                  f"n_clones: {s['n_clones']}")
            print(f"  Clone IDs: {s['clone_ids']}")
        expected_clone_ids = set()
        source_ids = set()
        for s in specs:
            expected_clone_ids.update(s["clone_ids"])
            source_ids.add(s["source_q_id"])
    else:
        expected_clone_ids = None
        source_ids = None

    # Load distances
    df = load_distance_df(clone_id)
    if df is None:
        print("SKIP: No distance data available. Run the pipeline first.")
        return

    print(f"\nDistance DataFrame: {df.shape[0]} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Identify clone IDs in the distance data
    all_ids = sorted(set(df["ID1"].tolist() + df["ID2"].tolist()))
    data_clone_ids = {x for x in all_ids if is_clone_id(x)}
    data_real_ids = {x for x in all_ids if not is_clone_id(x)}

    print(f"\nIDs in distance data:")
    print(f"  Real questions: {len(data_real_ids)}")
    print(f"  Clone questions: {len(data_clone_ids)}")
    print(f"  Clone IDs: {sorted(data_clone_ids)}")

    # Check if clone IDs match metadata
    if expected_clone_ids is not None:
        if data_clone_ids == expected_clone_ids:
            print(f"\n  OK: Clone IDs in distance data match metadata")
        else:
            print(f"\n  MISMATCH: Clone IDs differ!")
            print(f"    In metadata: {sorted(expected_clone_ids)}")
            print(f"    In distances: {sorted(data_clone_ids)}")
            print(f"    Distance data is STALE — re-run the pipeline.")
            return

    # Classify pairs
    def classify_pair(id1, id2):
        c1, c2 = is_clone_id(id1), is_clone_id(id2)
        if c1 and c2:
            src1, src2 = get_source_from_clone(id1), get_source_from_clone(id2)
            if src1 == src2:
                return "clone-clone (same source)"
            return "clone-clone (diff source)"
        elif c1 or c2:
            clone, real = (id1, id2) if c1 else (id2, id1)
            src = get_source_from_clone(clone)
            if real == src:
                return "clone-source"
            return "clone-other"
        return "real-real"

    df["pair_type"] = df.apply(lambda r: classify_pair(r["ID1"], r["ID2"]), axis=1)

    print(f"\nPair type counts:")
    for ptype, count in df["pair_type"].value_counts().items():
        print(f"  {ptype}: {count}")

    print(f"\nDistance statistics by pair type:")
    print(f"{'Pair Type':<30} {'Count':>6} {'Mean':>10} {'Median':>10} {'Max':>10} {'Min':>10}")
    print("-" * 80)
    for ptype in ["clone-source", "clone-clone (same source)", "clone-other", "real-real"]:
        subset = df[df["pair_type"] == ptype]
        if len(subset) == 0:
            continue
        dists = subset["Distance"]
        print(f"{ptype:<30} {len(subset):>6} {dists.mean():>10.6f} "
              f"{dists.median():>10.6f} {dists.max():>10.6f} {dists.min():>10.6f}")

    # Verdict
    clone_pairs = df[df["pair_type"].isin(["clone-source", "clone-clone (same source)"])]
    if len(clone_pairs) > 0:
        max_clone_dist = clone_pairs["Distance"].max()
        if max_clone_dist == 0.0:
            print(f"\n  PASS: All clone distances are exactly 0.0")
        elif max_clone_dist < 1e-3:
            print(f"\n  WARN: Clone distances non-zero — floating-point rounding in embedding "
                  f"model (max={max_clone_dist:.2e}). Fix: add identical-text short-circuit "
                  f"in similarity_metrics.py.")
        else:
            print(f"\n  FAIL: Clone distances too large (max={max_clone_dist:.2e})")
    else:
        print(f"\n  NO CLONE PAIRS FOUND — data may be stale or source question not matched")


def main():
    parser = argparse.ArgumentParser(description="Verify identical clone distances.")
    parser.add_argument(
        "--clone-id",
        type=str,
        default=None,
        help="Clone ID to verify (e.g., 'identical_combinedvar_n10'). "
             "If not specified, verifies both identical experiments.",
    )
    args = parser.parse_args()

    if args.clone_id:
        verify_clone_distances(args.clone_id)
    else:
        for cid in ["identical_combinedvar_n10", "identical_q32214_n10"]:
            verify_clone_distances(cid)


if __name__ == "__main__":
    main()
