import argparse
import warnings
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from pathlib import Path


"""
Important assumption: Assume rec dist = L2_sv. 
"""


class CrossRunAnalyzer:
    def __init__(self, run_a_path: str, run_b_path: str):
        self.run_a_path = Path(run_a_path)
        self.run_b_path = Path(run_b_path)

        # Load the data immediately upon initialization
        self.df_a = self._load_data(self.run_a_path)
        self.df_b = self._load_data(self.run_b_path)

    def _load_data(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Could not find file at {path}")

        df = pd.read_parquet(path)

        # Grab columns in the exact order they appear in the file
        std_cols = [c for c in df.columns if re.match(r"^_matchID_\d+_L2_sv$", c)]
        crw_cols = [c for c in df.columns if re.match(r"^CRW__matchID_\d+_L2_sv$", c)]

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

    def _can_compute_correlations(
        self, merged_df: pd.DataFrame, col_a: str, col_b: str
    ) -> bool:
        """
        Sweeps the dataset to ensure ALL voters have the exact same candidate sets
        in both runs. If even one voter has a mismatch, returns False.
        """
        for list_a, list_b in zip(merged_df[col_a], merged_df[col_b]):
            # Quick length check first (faster than set conversion)
            if len(list_a) != len(list_b):
                return False
            # Strict set check
            if set(list_a) != set(list_b):
                return False
        return True

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

        # THE O(1) FIX: Map candidates to their rank in List B instantly using a dictionary
        rank_dict_b = {candidate: rank for rank, candidate in enumerate(list_b)}

        try:
            # Look up the positions in List B based on List A's exact order
            ranks_b = [rank_dict_b[c] for c in list_a]
        except KeyError:
            # Failsafe just in case a candidate from A is totally missing in B
            return np.nan, np.nan

        # Since we evaluate based on List A's order, List A's ranks are simply 0, 1, 2, 3...
        ranks_a = np.arange(n)

        # SciPy processes these large arrays instantly in C
        spearman_corr, _ = spearmanr(ranks_a, ranks_b)
        kendall_tau, _ = kendalltau(ranks_a, ranks_b)

        return spearman_corr, kendall_tau

    def calculate_jaccard_at_n(self, list_a: list, list_b: list, n: int) -> float:
        """
        Computes the Jaccard similarity between the top-n items of two ranked lists.
        """
        # Isolate the top n candidates
        top_n_a = set(list_a[:n])
        top_n_b = set(list_b[:n])

        # Edge case: If n=0 or both lists are completely empty
        if not top_n_a and not top_n_b:
            return 1.0

        # Calculate intersection and union sizes
        intersection_size = len(top_n_a.intersection(top_n_b))
        union_size = len(top_n_a.union(top_n_b))

        return intersection_size / union_size

    def analyze(self, n: int):
        """
        Executes the comparison using high-speed native Python zipping.
        """
        # --- STRICT VOTER CHECK ---
        voters_a = set(self.df_a["voterID"])
        voters_b = set(self.df_b["voterID"])

        if voters_a != voters_b:
            missing_in_b = len(voters_a - voters_b)
            missing_in_a = len(voters_b - voters_a)
            raise ValueError(
                f"❌ FATAL: Voter sets do not match! "
                f"Run A has {len(voters_a)} voters, Run B has {len(voters_b)}. "
                f"({missing_in_b} missing in B, {missing_in_a} missing in A)."
            )

        # Since we know they match perfectly, an inner merge is safe and fast
        merged = pd.merge(
            self.df_a,
            self.df_b,
            on="voterID",
            suffixes=("_data", "_cloned"),
            how="inner",
        )

        # --- THE CIRCUIT BREAKERS ---
        can_do_base_corr = self._can_compute_correlations(
            merged, "ranked_standard_data", "ranked_standard_cloned"
        )
        can_do_crw_corr = self._can_compute_correlations(
            merged, "ranked_crw_data", "ranked_crw_cloned"
        )

        if not can_do_base_corr:
            print(
                "⚠️ Mismatched candidate sets detected in standard runs (likely top-k truncation). Skipping standard rank correlations."
            )
        if not can_do_crw_corr:
            print(
                "⚠️ Mismatched candidate sets detected in CRW runs (likely top-k truncation). Skipping CRW rank correlations."
            )

        results = []

        # Convert pandas columns to native Python lists for instant iteration
        voter_ids = merged["voterID"].tolist()
        base_data = merged["ranked_standard_data"].tolist()
        base_cloned = merged["ranked_standard_cloned"].tolist()
        crw_data = merged["ranked_crw_data"].tolist()
        crw_cloned = merged["ranked_crw_cloned"].tolist()

        # zip() runs in C natively, bypassing pandas object creation
        for voterID, b_data, b_cloned, c_data, c_cloned in zip(
            voter_ids, base_data, base_cloned, crw_data, crw_cloned
        ):

            # 1. Base comparisons
            base_jaccard = self.calculate_jaccard_at_n(b_data, b_cloned, n)
            if can_do_base_corr:
                base_spearman, base_kendall = self.calculate_rank_correlations(
                    b_data, b_cloned, voterID
                )
            else:
                base_spearman, base_kendall = np.nan, np.nan

            # 2. CRW comparisons
            crw_jaccard = self.calculate_jaccard_at_n(c_data, c_cloned, n)
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
    # CLI format:
    # python compare_runs.py <path_to_df_a> <path_to_df_b> [-n N_VALUE]

    parser = argparse.ArgumentParser(description="Compare two VAA recommendation runs.")

    # Positional arguments (required, supports tab-completion)
    parser.add_argument(
        "run_a", type=str, help="Path to the first parquet file (e.g., standard data)"
    )
    parser.add_argument(
        "run_b", type=str, help="Path to the second parquet file (e.g., cloned data)"
    )

    # Optional argument
    # For now, to be calculated later
    parser.add_argument(
        "-n",
        "--n_value",
        type=int,
        default=30,
        help="The 'n' value for Jaccard@n. (Defaults to 10 until metadata integration)",
    )

    args = parser.parse_args()

    print(f"Loading runs:\n  A: {args.run_a}\n  B: {args.run_b}")

    try:
        analyzer = CrossRunAnalyzer(args.run_a, args.run_b)
        print(f"Running analysis with n={args.n_value}...")

        results_df = analyzer.analyze(n=args.n_value)

        print("\n--- Mean Results Across All Voters ---")
        # Drop voterID before calculating the mean so pandas doesn't try to average strings
        print(results_df.drop(columns=["voterID"]).mean())

    except Exception as e:
        print(f"\n❌ Error: {e}")
