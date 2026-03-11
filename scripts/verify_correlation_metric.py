"""
Verify that the ANSWER-CORRELATION metric correctly assigns distance ≈ 0
to clone-source pairs for both easy_paraphrase (r ≈ +1) and negation_easy (r ≈ -1).

The metric uses |Pearson_r|, so both should produce distance ≈ 0.
This confirms the absolute value is implemented correctly.

Usage:
    python -m scripts.verify_correlation_metric
"""

from pathlib import Path

import pandas as pd

from main import load_config
from vqs.data_loader import load_dataset
from vqs.similarity_metrics import get_calculator

CLONE_ID_BASE = 9_000_000

CONFIGS = {
    "easy_paraphrase": Path(
        "configs/full_pipeline/cloned/easy_paraphrase_top5impact_n4_answer_corr_ZH.py"
    ),
    "negation_easy": Path(
        "configs/full_pipeline/cloned/negation_easy_top5impact_n4_answer_corr_ZH.py"
    ),
}


def source_id_from_clone(clone_id: int) -> int:
    """Reverse-engineer source question ID from clone ID."""
    return (clone_id - CLONE_ID_BASE) // 1000


def extract_clone_source_pairs(dist_df: pd.DataFrame) -> pd.DataFrame:
    """Filter distance rows to only clone-source pairs."""
    rows = []
    for _, row in dist_df.iterrows():
        id1, id2 = int(row["ID1"]), int(row["ID2"])
        # Check if one is a clone and the other is its source
        if id1 >= CLONE_ID_BASE and id2 < CLONE_ID_BASE:
            if source_id_from_clone(id1) == id2:
                rows.append({
                    "source_id": id2,
                    "clone_id": id1,
                    "distance": row["Distance"],
                    "abs_r": 1.0 - row["Distance"],
                })
        elif id2 >= CLONE_ID_BASE and id1 < CLONE_ID_BASE:
            if source_id_from_clone(id2) == id1:
                rows.append({
                    "source_id": id1,
                    "clone_id": id2,
                    "distance": row["Distance"],
                    "abs_r": 1.0 - row["Distance"],
                })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("VERIFICATION: Answer-Correlation Metric on Clone-Source Pairs")
    print("=" * 70)

    for clone_type, config_path in CONFIGS.items():
        print(f"\n{'─' * 70}")
        print(f"Clone type: {clone_type}")
        print(f"{'─' * 70}")

        config = load_config(config_path)
        dataset = load_dataset(config)
        calculator = get_calculator(config)
        dist_df = calculator.calculate_distance(dataset, config)

        pairs = extract_clone_source_pairs(dist_df)

        if pairs.empty:
            print("  WARNING: No clone-source pairs found!")
            continue

        print(f"\n  {'Source ID':>10}  {'Clone ID':>10}  {'|r|':>8}  {'Distance':>10}")
        print(f"  {'─' * 44}")
        for _, row in pairs.sort_values(["source_id", "clone_id"]).iterrows():
            print(
                f"  {int(row['source_id']):>10}  {int(row['clone_id']):>10}  "
                f"{row['abs_r']:>8.4f}  {row['distance']:>10.4f}"
            )

        print(f"\n  Summary:")
        print(f"    N pairs:          {len(pairs)}")
        print(f"    Mean |r|:         {pairs['abs_r'].mean():.4f}")
        print(f"    Min |r|:          {pairs['abs_r'].min():.4f}")
        print(f"    Mean distance:    {pairs['distance'].mean():.4f}")
        print(f"    Max distance:     {pairs['distance'].max():.4f}")

        threshold = 0.05
        all_close = (pairs["distance"] < threshold).all()
        print(f"    All distances < {threshold}: {'YES' if all_close else 'NO'}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
