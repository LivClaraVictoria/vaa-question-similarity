"""Compile all Gen2 alpha sweep CSVs into a single merged CSV.

Reads individual alpha sweep results (10 models x 6 clone types x 21 alphas)
and outputs a single CSV for external plotting (e.g., Overleaf, Jupyter).

Usage:
    python scripts/compile_merged_alpha_sweep_csv.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("experiment_results/exp1/model_alpha_sweep/top5impact")
OUTPUT_PATH = Path("experiment_results/exp1/model_alpha_sweep/compiled/alpha_sweep_all_models_all_clones.csv")
GEN2_FILTER = "top5impact_n4"


def parse_dirname(dirname: str) -> tuple[str, str] | None:
    """Extract (model, clone_type) from a Gen2 alpha sweep directory name."""
    if GEN2_FILTER not in dirname:
        return None
    parts = dirname.split("_vs_", 1)
    if len(parts) != 2:
        return None
    config_a, config_b = parts
    model = config_a.replace("alpha_sweep_pipeline_", "").removesuffix("_ZH")
    suffix = f"_top5impact_n4_{model}_ZH"
    if not config_b.endswith(suffix):
        return None
    clone_type = config_b[: -len(suffix)]
    return model, clone_type


def main():
    if not RESULTS_DIR.is_dir():
        print(f"ERROR: {RESULTS_DIR} not found")
        sys.exit(1)

    frames = []
    for subdir in sorted(RESULTS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = parse_dirname(subdir.name)
        if parsed is None:
            continue
        model, clone_type = parsed

        csvs = sorted(
            f for f in subdir.glob("alpha_sweep_*.csv")
            if "worker" not in f.name
        )
        if not csvs:
            continue

        csv_path = csvs[-1]  # latest by timestamp
        df = pd.read_csv(csv_path)
        df["model"] = model
        df["clone_type"] = clone_type

        match = re.search(r"_(\d{4})_(\d{4})_[a-f0-9]+\.csv$", csv_path.name)
        df["_ts"] = match.group(1) + match.group(2) if match else "0000"
        frames.append(df)

    if not frames:
        print(f"ERROR: No Gen2 CSVs found in {RESULTS_DIR}")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("_ts").drop_duplicates(
        subset=["model", "clone_type", "alpha"], keep="last"
    )
    combined = combined.drop(columns=["_ts"])

    # Reorder columns
    col_order = ["model", "clone_type", "alpha",
                 "crw_jaccard_mean", "crw_jaccard_median", "crw_jaccard_p10",
                 "crw_spearman_mean", "crw_kendall_mean"]
    combined = combined[col_order].sort_values(
        ["clone_type", "model", "alpha"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    n_models = combined["model"].nunique()
    n_clones = combined["clone_type"].nunique()
    n_alphas = combined["alpha"].nunique()
    print(f"Saved {len(combined)} rows ({n_models} models x {n_clones} clone types x {n_alphas} alphas)")
    print(f"  Models: {sorted(combined['model'].unique())}")
    print(f"  Clone types: {sorted(combined['clone_type'].unique())}")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
