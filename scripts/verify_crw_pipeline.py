"""Verify CRW pipeline correctness via a minimum-distance alpha sanity check.

Logic: if alpha is set strictly below the minimum non-zero pairwise question distance,
then at every integration threshold r in [0, alpha] the adjacency matrix is the identity —
each question is only "within distance r" of itself. Every question therefore forms its
own singleton class, so _class_uniform_weighting_fn assigns weight = 1.0 to all questions,
and the integral evaluates uniformly to N for all N questions. CRW then produces identical
recommendations to the unweighted baseline → Jaccard = Spearman = Kendall = 1.0.

Usage:
    python -m scripts.verify_crw_pipeline --config configs/full_pipeline/base_data/pipeline_e5_ZH.py
    python -m scripts.verify_crw_pipeline --config configs/full_pipeline/base_data/pipeline_e5_ZH.py -n 35
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from main import load_config
from vqs.data_loader import load_dataset
from vqs.similarity_metrics import get_calculator
from vqs.clone_robust_weighting import CloneRobustReweighter
from vqs.recommendation_engine import RecommendationEngine
from vqs.recommendation_saver import (
    _generate_stats,
    _get_jaccard_n,
    _calculate_jaccard,
    _calculate_rank_metrics,
)


def _to_distances(dist_df):
    """Return a numpy array of pairwise distances, converting from similarity if needed."""
    if "Distance" in dist_df.columns:
        return dist_df["Distance"].values
    elif "Similarity" in dist_df.columns:
        sim = dist_df["Similarity"].values
        return np.sqrt(np.maximum(0.0, 2.0 * (1.0 - sim)))
    else:
        raise ValueError(
            "Distance DataFrame has neither 'Distance' nor 'Similarity' column."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sanity check: verify CRW produces identical recommendations to baseline "
            "when alpha is set below the minimum pairwise question distance."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a full pipeline config (must have load_voters=True, load_candidates=True).",
    )
    parser.add_argument(
        "-n",
        "--n_jaccard",
        type=int,
        default=None,
        help="Top-k for Jaccard similarity (default: inferred from config).",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    config_name = Path(config.__file__).stem

    print(f"\n{'='*60}")
    print(f"CRW Pipeline Sanity Check")
    print(f"Config: {config_name}  |  Dist metric: {config.dist}")
    print(f"{'='*60}")

    # --- Setup ---
    print("\nLoading dataset...")
    dataset = load_dataset(config)
    n_questions = len(dataset["questions"])
    voters = dataset.get("voters")
    n_voters = len(voters) if voters is not None else "N/A"
    print(f"Questions: {n_questions}  |  Voters: {n_voters}")

    print("\nComputing/loading distances (cached if available)...")
    calculator = get_calculator(config)
    dist_df = calculator.calculate_distance(dataset, config)

    # --- Step 1: Find minimum non-zero pairwise distance ---
    print(f"\n{'─'*60}")
    print("Step 1: Minimum pairwise distance")
    print(f"{'─'*60}")

    distances = _to_distances(dist_df)
    nonzero_mask = distances > 1e-12
    if not nonzero_mask.any():
        print("ERROR: All pairwise distances are zero — cannot determine test alpha.")
        sys.exit(1)

    min_dist = float(distances[nonzero_mask].min())
    test_alpha = min_dist / 2.0

    print(f"Min non-zero pairwise distance:  {min_dist:.8f}")
    print(f"Test alpha:                      {test_alpha:.8f}  (= min_dist / 2)")
    print(f"Condition: alpha ({test_alpha:.6f}) < min_dist ({min_dist:.6f})  ✓")

    config.alpha = test_alpha

    # --- Step 2: CRW weight uniformity ---
    print(f"\n{'─'*60}")
    print("Step 2: CRW weight uniformity")
    print(f"{'─'*60}")

    reweighter = CloneRobustReweighter(config)
    weights_df = reweighter.reweight(dist_df)

    weights = weights_df["Weight"].values
    w_min, w_max, w_std = float(weights.min()), float(weights.max()), float(weights.std())
    print(f"CRW weights: min={w_min:.6f}  max={w_max:.6f}  std={w_std:.2e}")

    weights_uniform = w_std < 1e-8
    if weights_uniform:
        print("✓ PASS: All CRW weights are uniform")
    else:
        print("✗ FAIL: CRW weights are NOT uniform")
        print("        Expected equal weights since alpha < min pairwise distance.")

    # --- Step 3: Baseline vs CRW recommendations ---
    print(f"\n{'─'*60}")
    print("Step 3: Baseline vs CRW recommendations")
    print(f"{'─'*60}")

    rec_engine = RecommendationEngine(config, dataset)
    combined_recs = rec_engine.evaluate_pipeline(weights_df)

    method = config.rec_dist_method
    base_cols = sorted(
        [c for c in combined_recs.columns if c.startswith("_matchID_") and c.endswith(f"_{method}")],
        key=lambda c: int(c.split("_")[2]),
    )
    crw_cols = sorted(
        [c for c in combined_recs.columns if c.startswith("CRW__matchID_") and c.endswith(f"_{method}")],
        key=lambda c: int(c.split("_")[3]),
    )

    n_jac = args.n_jaccard if args.n_jaccard is not None else _get_jaccard_n(combined_recs, config, base_cols, crw_cols)
    print(f"Jaccard top-k: {n_jac}")

    summary = _generate_stats(combined_recs, config, method, base_cols, crw_cols, n_jac)
    print(summary)

    jaccard_scores, _ = _calculate_jaccard(config, combined_recs, base_cols, crw_cols, n_jac)
    rank_stats = _calculate_rank_metrics(combined_recs, base_cols, crw_cols)

    jaccard_mean = float(np.mean(jaccard_scores))
    pct_changed = rank_stats["voters_with_change"]
    recs_identical = jaccard_mean > 1 - 1e-6 and pct_changed < 1e-6

    if recs_identical:
        print("✓ PASS: CRW recommendations are identical to baseline")
    else:
        print("✗ FAIL: CRW recommendations differ from baseline")
        print(f"        Jaccard mean = {jaccard_mean:.6f}  (expected 1.0)")
        print(f"        Voters with rank changes = {pct_changed:.2f}%  (expected 0%)")

    # --- Final verdict ---
    print(f"\n{'='*60}")
    passed = weights_uniform and recs_identical
    if passed:
        print("✓  PASS — CRW pipeline sanity check succeeded.")
        print("   CRW with alpha < min_dist produces identical recommendations to baseline.")
    else:
        print("✗  FAIL — CRW pipeline sanity check FAILED.")
        if not weights_uniform:
            print("   Step 2 failed: CRW weights are not uniform.")
        if not recs_identical:
            print("   Step 3 failed: CRW recommendations differ from baseline.")
    print(f"{'='*60}\n")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
