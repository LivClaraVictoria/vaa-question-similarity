import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path


class CrossRunSaver:
    def __init__(self, output_dir: str = "experiment_results/comparator_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(self, df: pd.DataFrame, meta_a: dict, meta_b: dict, n_used: int):
        """
        Saves the comparison results with a deterministic hash and expanded prefix.
        """
        # 1. Generate Hash
        combined_config = {"run_a": meta_a, "run_b": meta_b, "comparison_n": n_used}
        config_str = json.dumps(combined_config, sort_keys=True)
        run_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

        # 2. Construct Detailed Prefix (Name + Overrides)
        prefix_a = self._get_name_with_overrides(meta_a)
        prefix_b = self._get_name_with_overrides(meta_b)

        filename = f"compare_{prefix_a}_vs_{prefix_b}_{run_hash}"

        # 3. Check existence
        parquet_path = self.output_dir / f"{filename}.parquet"
        if parquet_path.exists():
            print(f"--- [Skip Save] Comparison result already exists: ---")
            print(f"    -> {parquet_path.name}")
            self._print_existing_summary(parquet_path)
            return

        # 4. Save Data
        df.to_parquet(parquet_path)
        print(f"Saved comparison data to:\n  -> {parquet_path}")

        # 5. Generate & Save Text Summary
        summary = self._generate_summary(df, meta_a, meta_b, n_used)
        txt_path = parquet_path.with_suffix(".txt")
        txt_path.write_text(summary, encoding="utf-8")
        print(f"  -> {txt_path}")

        # 6. Save Metadata
        json_path = parquet_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(combined_config, f, indent=4)

        print(summary)

    def _get_name_with_overrides(self, meta: dict) -> str:
        """Extracts config name and appends overrides if present."""
        base = meta.get("config_name", "unknown")
        overrides = meta.get("overrides", [])
        if overrides and isinstance(overrides, list):
            # Clean up overrides string (replace special chars like ~ with _)
            clean_overrides = "_".join(overrides).replace("~", "").replace("=", "")
            return f"{base}_{clean_overrides}"
        return base

    def _generate_summary(self, df, meta_a, meta_b, n) -> str:
        """
        Generates a summary matching the EXACT format of recommendation_saver.py,
        then appends cross-run specifics with detailed distribution stats.
        """
        # --- CALCULATE AGGREGATES ---
        # 1. Jaccard Stats
        avg_jaccard = df["base_jaccard"].mean()
        min_jaccard = df["base_jaccard"].min()

        # 2. Candidate Swaps (derived from Jaccard)
        jaccard_vals = df["base_jaccard"].values
        # Avoid division by zero if jaccard is -1 (unlikely but possible in math)
        with np.errstate(divide="ignore", invalid="ignore"):
            intersections = (2 * n * jaccard_vals) / (1 + jaccard_vals)
            swaps = n - intersections
            # Handle NaNs from 0/0 or similar edge cases
            swaps = np.nan_to_num(swaps, nan=0.0)

        avg_swaps = np.mean(swaps)
        max_swaps = np.max(swaps) if len(swaps) > 0 else 0

        # 3. Rank Stability (Full List)
        total_voters = len(df)
        voters_with_change = (df["base_changed_count"] > 0).mean() * 100
        avg_changed_cands = df["base_changed_count"].mean()

        list_len = df["base_list_len"].iloc[0] if total_voters > 0 else 0
        avg_pct_list_changed = (
            (avg_changed_cands / list_len * 100) if list_len > 0 else 0
        )

        grand_total_shift = df["base_total_shift"].sum()
        total_candidates_across_all_voters = df["base_list_len"].sum()
        avg_shift = (
            (grand_total_shift / total_candidates_across_all_voters)
            if total_candidates_across_all_voters > 0
            else 0
        )

        max_shift = df["base_max_shift"].max() if total_voters > 0 else 0

        # 4. Correlation Distributions (New!)
        def get_dist_stats(series):
            return {
                "mean": series.mean(),
                "min": series.min(),
                "max": series.max(),
                "std": series.std(),
                "p05": series.quantile(0.05),  # 5th percentile (worst 5%)
                "high_corr_pct": (series > 0.99).mean() * 100,  # % of "perfect" matches
            }

        base_s = get_dist_stats(df["base_spearman"])
        base_k = get_dist_stats(df["base_kendall"])
        crw_s = get_dist_stats(df["crw_spearman"])
        crw_k = get_dist_stats(df["crw_kendall"])

        # --- FORMATTING ---
        year_str = meta_a.get("data_year", "Unknown")

        summary = (
            f"\n--- Stats Summary {year_str} (Cross-Run) ---\n"
            f"Scope: Evaluating top {n} recommendations for Set Similarity, and all {list_len} slots for Rank Stability.\n\n"
            f"Set Similarity (Top {n}):\n"
            f"  - Average Jaccard Similarity:       {avg_jaccard:.4f} (1.0 = identical sets)\n"
            f"  - Minimum Jaccard Similarity:       {min_jaccard:.4f}\n"
            f"  - Avg. Candidates Swapped In/Out:   {avg_swaps:.2f} candidates per voter\n"
            f"  - Max Candidates Swapped In/Out:    {int(max_swaps)} candidates\n\n"
            f"Rank Stability (All {list_len}):\n"
            f"  - Voters w/ at least 1 rank change: {voters_with_change:.2f}%\n"
            f"  - Avg. candidates changed per voter:{avg_changed_cands:>6.2f} candidates ({avg_pct_list_changed:.2f}% of list)\n"
            f"  - Avg. positions a candidate moved: {avg_shift:>6.2f} ranks\n"
            f"  - Max positions a candidate moved:  {int(max_shift):>6} ranks\n"
            f"----------------------------\n\n"
            f"--- Additional Correlation Metrics ---\n"
            f"1. Standard Recommendations (Spearman Rho):\n"
            f"   Mean: {base_s['mean']:.4f} | Min: {base_s['min']:.4f} | Max: {base_s['max']:.4f} | StdDev: {base_s['std']:.4f}\n"
            f"   (worst 5% are below {base_s['p05']:.4f}, {base_s['high_corr_pct']:.1f}% are > 0.99)\n\n"
            f"2. Standard Recommendations (Kendall Tau):\n"
            f"   Mean: {base_k['mean']:.4f} | Min: {base_k['min']:.4f} | Max: {base_k['max']:.4f} | StdDev: {base_k['std']:.4f}\n\n"
            f"3. CRW Recommendations (Spearman Rho):\n"
            f"   Mean: {crw_s['mean']:.4f} | Min: {crw_s['min']:.4f} | Max: {crw_s['max']:.4f} | StdDev: {crw_s['std']:.4f}\n"
            f"   (worst 5% are below {crw_s['p05']:.4f}, {crw_s['high_corr_pct']:.1f}% are > 0.99)\n"
        )
        return summary

    def _print_existing_summary(self, path: Path):
        txt_path = path.with_suffix(".txt")
        if txt_path.exists():
            print(txt_path.read_text(encoding="utf-8"))
