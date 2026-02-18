import argparse
import pandas as pd
import numpy as np
import json
import re
import warnings
from pathlib import Path
from scipy.stats import spearmanr, kendalltau


class CrossRunAnalyzer:
    def __init__(self, run_a_path: str, run_b_path: str, n_override: int | None = None):
        self.run_a_path = Path(run_a_path)
        self.run_b_path = Path(run_b_path)
        self.n_override = n_override

        # 1. Load Metadata
        self.meta_a = self._load_metadata(self.run_a_path)
        self.meta_b = self._load_metadata(self.run_b_path)

        # 2. Print Comparison (Purely informational)
        self._compare_configs()

        # 3. Resolve 'n' (Logic decision)
        self.n = self._resolve_n()

        # 4. Load Data
        print(f"Loading data (this may take a moment)...")
        self.df_a = self._load_data(self.run_a_path)
        self.df_b = self._load_data(self.run_b_path)

    def _load_metadata(self, parquet_path: Path) -> dict:
        """Looks for a .json file with the same name as the parquet file."""
        json_path = parquet_path.with_suffix(".json")
        if not json_path.exists():
            print(f"⚠️ Warning: No metadata found at {json_path}. Defaulting to n=30.")
            return {"n_jaccard": 30}

        with open(json_path, "r") as f:
            return json.load(f)

    def _compare_configs(self):
        """Prints differences between the two run configurations for user awareness."""
        print("\n--- Configuration Comparison ---")

        all_keys = set(self.meta_a.keys()) | set(self.meta_b.keys())
        diffs = []

        for k in sorted(all_keys):
            if k == "config_name":
                continue

            val_a = self.meta_a.get(k, "N/A")
            val_b = self.meta_b.get(k, "N/A")

            # Handle list/type mismatch in string comparison
            if str(val_a) != str(val_b):
                diffs.append(f"{k}: {val_a} vs {val_b}")

        if diffs:
            print("Differences detected between runs:")
            for d in diffs:
                print(f"  • {d}")
        else:
            print("✅ Configurations are identical.")

    def _resolve_n(self) -> int:
        """Decides which 'n' value to use for Jaccard calculations."""
        # Priority 1: CLI Override
        if self.n_override is not None:
            print(
                f"⚠️  CLI Override active: Using n={self.n_override} (ignoring metadata)."
            )
            return self.n_override

        # Priority 2: Metadata
        n_a = self.meta_a.get("n_jaccard", 30)
        n_b = self.meta_b.get("n_jaccard", 30)

        if n_a != n_b:
            resolved_n = min(n_a, n_b)
            print(f"⚠️  Mismatch in configured Jaccard 'n' (Run A={n_a}, Run B={n_b}).")
            print(f"   -> Using smaller n={resolved_n} for fair comparison.")
            return resolved_n

        print(f"   -> Using n={n_a} (derived from metadata).")
        return n_a

    def _load_data(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Could not find file at {path}")

        df = pd.read_parquet(path)

        # Grab columns in the exact order they appear in the file
        std_cols = [c for c in df.columns if re.match(r"^_matchID_\d+_L2_sv$", c)]
        crw_cols = [c for c in df.columns if re.match(r"^CRW__matchID_\d+_L2_sv$", c)]

        # Collapse columns into lists
        if std_cols:
            df["ranked_standard"] = df[std_cols].values.tolist()
            df["ranked_standard"] = df["ranked_standard"].apply(
                lambda lst: [x for x in lst if pd.notna(x)]
            )

        if crw_cols:
            df["ranked_crw"] = df[crw_cols].values.tolist()
            df["ranked_crw"] = df["ranked_crw"].apply(
                lambda lst: [x for x in lst if pd.notna(x)]
            )

        return df

    def calculate_rank_correlations(self, list_a: list, list_b: list, voter_id: str):
        """Computes correlations using O(1) dictionary lookups and fast C-compiled SciPy math."""
        n = len(list_a)
        if n < 2:
            return np.nan, np.nan

        if len(set(list_a)) != n:
            warnings.warn(
                f"Voter {voter_id}: Duplicate candidates found. Returning NaN."
            )
            return np.nan, np.nan

        rank_dict_b = {candidate: rank for rank, candidate in enumerate(list_b)}

        try:
            ranks_b = [rank_dict_b[c] for c in list_a]
        except KeyError:
            return np.nan, np.nan

        ranks_a = np.arange(n)

        spearman_corr, _ = spearmanr(ranks_a, ranks_b)
        kendall_tau, _ = kendalltau(ranks_a, ranks_b)

        return spearman_corr, kendall_tau

    def calculate_jaccard_at_n(self, list_a: list, list_b: list, n: int) -> float:
        """Computes Jaccard similarity for top-n items."""
        top_n_a = set(list_a[:n])
        top_n_b = set(list_b[:n])

        if not top_n_a and not top_n_b:
            print(
                "⚠️ Warning: Both lists are empty. Returning Jaccard similarity of 1.0."
            )
            return 1.0

        intersection_size = len(top_n_a.intersection(top_n_b))
        union_size = len(top_n_a.union(top_n_b))

        return intersection_size / union_size

    def _can_compute_correlations(
        self, merged_df: pd.DataFrame, col_a: str, col_b: str
    ) -> bool:
        """Circuit breaker to prevent invalid math on mismatched sets."""
        for list_a, list_b in zip(merged_df[col_a], merged_df[col_b]):
            if len(list_a) != len(list_b):
                return False
            if set(list_a) != set(list_b):
                return False
        return True

    def analyze(self):
        """
        Executes the comparison using high-speed native Python zipping.
        """
        # --- STRICT DUPLICATE CHECK ---
        if self.df_a["voterID"].duplicated().any():
            dupes = self.df_a["voterID"].duplicated().sum()
            raise ValueError(f"❌ FATAL: Run A has {dupes} duplicate voterIDs!")
        if self.df_b["voterID"].duplicated().any():
            dupes = self.df_b["voterID"].duplicated().sum()
            raise ValueError(f"❌ FATAL: Run B has {dupes} duplicate voterIDs!")

        # --- STRICT VOTER CHECK ---
        voters_a = set(self.df_a["voterID"])
        voters_b = set(self.df_b["voterID"])

        if voters_a != voters_b:
            missing_in_b = len(voters_a - voters_b)
            missing_in_a = len(voters_b - voters_a)
            raise ValueError(
                f"❌ FATAL: Voter sets do not match! ({missing_in_b} missing in B, {missing_in_a} missing in A)."
            )

        # Inner merge is safe
        merged = pd.merge(
            self.df_a,
            self.df_b,
            on="voterID",
            suffixes=("_data", "_cloned"),
            how="inner",
        )

        # --- CIRCUIT BREAKERS ---
        can_do_base_corr = self._can_compute_correlations(
            merged, "ranked_standard_data", "ranked_standard_cloned"
        )
        can_do_crw_corr = self._can_compute_correlations(
            merged, "ranked_crw_data", "ranked_crw_cloned"
        )

        if not can_do_base_corr:
            print(
                "⚠️ Mismatched candidate sets detected in standard runs. Skipping standard rank correlations."
            )
        if not can_do_crw_corr:
            print(
                "⚠️ Mismatched candidate sets detected in CRW runs. Skipping CRW rank correlations."
            )

        results = []

        # --- THE SPEED LOOP ---
        voter_ids = merged["voterID"].tolist()
        base_data = merged["ranked_standard_data"].tolist()
        base_cloned = merged["ranked_standard_cloned"].tolist()
        crw_data = merged["ranked_crw_data"].tolist()
        crw_cloned = merged["ranked_crw_cloned"].tolist()

        n_val = self.n

        for voterID, b_data, b_cloned, c_data, c_cloned in zip(
            voter_ids, base_data, base_cloned, crw_data, crw_cloned
        ):

            # 1. Base comparisons
            base_jaccard = self.calculate_jaccard_at_n(b_data, b_cloned, n_val)
            if can_do_base_corr:
                base_spearman, base_kendall = self.calculate_rank_correlations(
                    b_data, b_cloned, voterID
                )
            else:
                base_spearman, base_kendall = np.nan, np.nan

            # 2. CRW comparisons
            crw_jaccard = self.calculate_jaccard_at_n(c_data, c_cloned, n_val)
            if can_do_crw_corr:
                crw_spearman, crw_kendall = self.calculate_rank_correlations(
                    c_data, c_cloned, voterID
                )
            else:
                crw_spearman, crw_kendall = np.nan, np.nan

            results.append(
                {
                    "voterID": voterID,
                    "base_jaccard": base_jaccard,
                    "base_spearman": base_spearman,
                    "base_kendall": base_kendall,
                    "crw_jaccard": crw_jaccard,
                    "crw_spearman": crw_spearman,
                    "crw_kendall": crw_kendall,
                }
            )

        return pd.DataFrame(results)


if __name__ == "__main__":
    """
    Basic usage (Automatically detects 'n' from metadata):
    python cross_run_analyzer.py path/to/run_A.parquet path/to/run_B.parquet

    Manual override (Force a specific 'n' for Jaccard similarity):
    python cross_run_analyzer.py path/to/run_A.parquet path/to/run_B.parquet -n 50
    """
    parser = argparse.ArgumentParser(description="Compare two VAA recommendation runs.")
    parser.add_argument(
        "run_a", type=str, help="Path to first parquet file (standard data)"
    )
    parser.add_argument(
        "run_b", type=str, help="Path to second parquet file (cloned data)"
    )
    parser.add_argument(
        "-n",
        "--n_value",
        type=int,
        default=None,
        help="Override the Jaccard 'n' value (ignores metadata)",
    )

    args = parser.parse_args()

    print(f"Loading runs:\n  A: {args.run_a}\n  B: {args.run_b}")

    try:
        analyzer = CrossRunAnalyzer(args.run_a, args.run_b, n_override=args.n_value)
        print("Running analysis...")
        results_df = analyzer.analyze()

        print("\n--- Mean Results Across All Voters ---")
        print(results_df.drop(columns=["voterID"]).mean())

    except Exception as e:
        print(f"\n❌ Error: {e}")
