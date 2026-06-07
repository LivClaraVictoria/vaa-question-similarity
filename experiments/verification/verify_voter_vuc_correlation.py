"""
Sensitivity check: voters-only vs VuC (voters ∪ candidates) for ANSWER-CORRELATION-ARCCOS.

Computes Spearman rank correlation between the two upper-triangle distance vectors and
reports the top-5 question pairs whose rank order changes most between the two sources.

Usage:
    python -m experiments.verification.verify_voter_vuc_correlation
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vqs.config_utils import load_config
from vqs.data_loader import load_dataset
from vqs.similarity_metrics import get_calculator

CONFIG_PATH = Path("configs/base_pipeline/pipeline_answer_corr_arccos_ZH.py")


def compute_distances(source: str, dataset: dict, config) -> pd.DataFrame:
    config.correlation_answer_source = source
    calculator = get_calculator(config)
    return calculator.calculate_distance(dataset, config)


def main():
    print("=" * 70)
    print("SENSITIVITY CHECK: voters-only vs VuC for ANSWER-CORRELATION-ARCCOS")
    print("=" * 70)

    config = load_config(CONFIG_PATH)
    dataset = load_dataset(config)

    n_voters = len(dataset.get("voters", pd.DataFrame()))
    n_cands = len(dataset.get("candidates", pd.DataFrame()))
    print(f"\nDataset (ZH): {n_voters} voters, {n_cands} candidates")
    print(f"  Candidates as fraction of total: {n_cands / (n_voters + n_cands):.1%}")

    print("\nComputing voters-only distances...")
    df_v = compute_distances("voters", dataset, config)

    print("Computing VuC distances...")
    df_vuc = compute_distances("both", dataset, config)

    # Align on (ID1, ID2) — same pairs guaranteed, but order may differ
    merged = df_v[["ID1", "ID2", "Qu1", "Qu2", "Distance"]].merge(
        df_vuc[["ID1", "ID2", "Distance"]],
        on=["ID1", "ID2"],
        suffixes=("_voters", "_vuc"),
    )
    assert len(merged) == len(df_v), "Mismatch in number of pairs after merge"

    d_v = merged["Distance_voters"].to_numpy()
    d_vuc = merged["Distance_vuc"].to_numpy()

    rho, p = spearmanr(d_v, d_vuc)
    mean_abs_diff = np.abs(d_v - d_vuc).mean()
    max_abs_diff = np.abs(d_v - d_vuc).max()

    print(f"\nDistance matrix agreement ({len(merged)} pairs):")
    print(f"  Spearman ρ:      {rho:.4f}  (p={p:.2e})")
    print(f"  Mean |Δd|:       {mean_abs_diff:.4f}")
    print(f"  Max  |Δd|:       {max_abs_diff:.4f}")

    merged["rank_v"] = merged["Distance_voters"].rank()
    merged["rank_vuc"] = merged["Distance_vuc"].rank()
    merged["rank_change"] = (merged["rank_vuc"] - merged["rank_v"]).abs()

    print(f"\nTop-5 pairs by absolute rank change:")
    print(f"  {'Qu1':30s}  {'Qu2':30s}  {'d_voters':>9}  {'d_vuc':>9}  {'|Δrank|':>8}")
    print(f"  {'─' * 92}")
    for _, row in merged.nlargest(5, "rank_change").iterrows():
        q1 = str(row["Qu1"])[:30]
        q2 = str(row["Qu2"])[:30]
        print(
            f"  {q1:30s}  {q2:30s}  "
            f"{row['Distance_voters']:>9.4f}  {row['Distance_vuc']:>9.4f}  "
            f"{row['rank_change']:>8.0f}"
        )

    print(f"\n{'=' * 70}")
    if rho >= 0.97:
        print("RESULT: ρ ≥ 0.97 — voters-only is an excellent approximation of VuC.")
        print("  → Add a one-sentence footnote in thesis. No re-running needed.")
    elif rho >= 0.90:
        print("RESULT: 0.90 ≤ ρ < 0.97 — good approximation; check CRW weight stability.")
    else:
        print("RESULT: ρ < 0.90 — meaningful difference. Consider re-running with VuC.")
    print("=" * 70)


if __name__ == "__main__":
    main()
