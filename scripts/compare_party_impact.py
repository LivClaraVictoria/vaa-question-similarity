"""Compare two party impact Phase 2 result sets across all parties.

Usage:
    python scripts/compare_party_impact.py <dir_a> <dir_b> [--out <output_dir>]

Example:
    python scripts/compare_party_impact.py \
        experiment_results/party_impact/phase2/pipeline_e5_instruct_ZH_a03 \
        experiment_results/party_impact/phase2/pipeline_e5_instruct_ZH_a04

    python scripts/compare_party_impact.py \
        experiment_results/party_impact/phase2/pipeline_e5_instruct_ZH_a04 \
        experiment_results/party_impact/phase2/pipeline_qwen3_ZH
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PARTIES = ["SP", "Green", "GLP", "Centre", "FDP", "SVP"]


def load_party_csvs(result_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all per-party Phase 2 CSVs from a result directory."""
    dfs = {}
    for party in PARTIES:
        party_dir = result_dir / party
        if not party_dir.exists():
            continue
        csvs = sorted(party_dir.glob("*_phase2.csv"))
        if not csvs:
            continue
        dfs[party] = pd.read_csv(csvs[-1])  # most recent
    return dfs


def compute_metrics(df: pd.DataFrame, target_party: str) -> dict:
    """Extract key metrics for the target party from a Phase 2 CSV."""
    row = df[df["party"] == target_party].iloc[0]
    orig = row["original"]
    orig_crw = row["original_crw"]

    metrics = {"original": orig, "drift": orig_crw - orig}

    for scenario in ["worst_case", "realistic"]:
        attacked = row[scenario]
        corrected = row[f"{scenario}_crw"]
        delta_attack = attacked - orig
        delta_crw = corrected - orig
        reduction = 1 - (delta_crw / delta_attack) if abs(delta_attack) > 1e-9 else 0

        metrics[f"{scenario}_attack_pp"] = delta_attack * 100
        metrics[f"{scenario}_crw_pp"] = delta_crw * 100
        metrics[f"{scenario}_reduction"] = reduction * 100
        metrics[f"{scenario}_correction_pp"] = (attacked - corrected) * 100

    return metrics


def compare(dir_a: Path, dir_b: Path, output_dir: Path | None = None):
    name_a = dir_a.name
    name_b = dir_b.name

    dfs_a = load_party_csvs(dir_a)
    dfs_b = load_party_csvs(dir_b)

    common_parties = [p for p in PARTIES if p in dfs_a and p in dfs_b]
    if not common_parties:
        print("ERROR: No common parties found between the two result sets.")
        sys.exit(1)

    rows = []
    for party in common_parties:
        # In each result set, get the metrics for *this party as target*
        m_a = compute_metrics(dfs_a[party], party)
        m_b = compute_metrics(dfs_b[party], party)

        row = {"target_party": party}
        for scenario in ["worst_case", "realistic"]:
            row[f"{scenario}_attack_pp"] = m_a[f"{scenario}_attack_pp"]
            row[f"{scenario}_red_A"] = m_a[f"{scenario}_reduction"]
            row[f"{scenario}_red_B"] = m_b[f"{scenario}_reduction"]
            row[f"{scenario}_red_diff"] = m_b[f"{scenario}_reduction"] - m_a[f"{scenario}_reduction"]
            row[f"{scenario}_corr_A_pp"] = m_a[f"{scenario}_correction_pp"]
            row[f"{scenario}_corr_B_pp"] = m_b[f"{scenario}_correction_pp"]
        rows.append(row)

    comp_df = pd.DataFrame(rows)

    # Print comparison tables
    print(f"\n{'=' * 90}")
    print(f"PARTY IMPACT COMPARISON")
    print(f"  A: {name_a}")
    print(f"  B: {name_b}")
    print(f"{'=' * 90}")

    for scenario, label in [("worst_case", "WORST-CASE (4 mixed clones)"),
                             ("realistic", "REALISTIC (1 easy paraphrase)")]:
        print(f"\n--- {label} ---")
        print(f"{'Target':>8}  {'Attack':>8}  {'Red. A':>8}  {'Red. B':>8}  {'Δ Red.':>8}  {'Corr. A':>9}  {'Corr. B':>9}")
        print("-" * 75)
        for _, r in comp_df.iterrows():
            print(
                f"{r['target_party']:>8}  "
                f"{r[f'{scenario}_attack_pp']:>+7.2f}pp  "
                f"{r[f'{scenario}_red_A']:>7.1f}%  "
                f"{r[f'{scenario}_red_B']:>7.1f}%  "
                f"{r[f'{scenario}_red_diff']:>+7.1f}%  "
                f"{r[f'{scenario}_corr_A_pp']:>8.2f}pp  "
                f"{r[f'{scenario}_corr_B_pp']:>8.2f}pp"
            )

        # Averages
        avg_red_a = comp_df[f"{scenario}_red_A"].mean()
        avg_red_b = comp_df[f"{scenario}_red_B"].mean()
        avg_diff = comp_df[f"{scenario}_red_diff"].mean()
        avg_corr_a = comp_df[f"{scenario}_corr_A_pp"].mean()
        avg_corr_b = comp_df[f"{scenario}_corr_B_pp"].mean()
        print("-" * 75)
        print(
            f"{'AVG':>8}  {'':>9}  "
            f"{avg_red_a:>7.1f}%  "
            f"{avg_red_b:>7.1f}%  "
            f"{avg_diff:>+7.1f}%  "
            f"{avg_corr_a:>8.2f}pp  "
            f"{avg_corr_b:>8.2f}pp"
        )

    print(f"\n{'=' * 90}")
    print("Red. = CRW reduction % (higher = better CRW correction)")
    print("Corr. = absolute CRW correction in pp (attacked → corrected)")
    print(f"Δ Red. = B - A (positive = B corrects more)")
    print(f"{'=' * 90}\n")

    # Save CSV
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = dir_b.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"comparison_{name_a}_vs_{name_b}.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two party impact result sets")
    parser.add_argument("dir_a", type=Path, help="First result directory (phase2/<config>/)")
    parser.add_argument("dir_b", type=Path, help="Second result directory (phase2/<config>/)")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for CSV")
    args = parser.parse_args()

    compare(args.dir_a, args.dir_b, args.out)
