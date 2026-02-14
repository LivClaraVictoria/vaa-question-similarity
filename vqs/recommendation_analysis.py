import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from vqs.result_management import ResultManager

"""
Important Assumption: 
We assume equal number of recommendations for all voters. Achieved either by setting n_recommendations to a fixed number, or by filtering voters and candidates by a certain canton. 
Any NaN values will raise an error.
"""


class RecommendationAnalyzer:
    def __init__(self, config, important_params_list):
        self.config = config
        self.method = config.rec_dist_method
        self.important_params_list = important_params_list
        self.output_dir = Path(config.RECOMMENDATION_RESULTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Initialized RecommendationAnalyzer with important parameters: {self.important_params_list}"
        )

    def analyze(
        self, df_recommendations: pd.DataFrame, df_weights: pd.DataFrame
    ) -> pd.DataFrame:
        df = df_recommendations.copy()

        # 0. Check for cached files
        prefix = self._get_prefix()
        rm = ResultManager(
            config=self.config,
            dir=self.output_dir,
            params_list=self.important_params_list,
            prefix=prefix,
        )

        exists = rm.exists()
        if exists:
            print(f"--- [Skip Save] Result with hash {rm.hash} already exists: ---")
            print(f"    -> {exists.name}")
            # TODO: print summary stats from text file if exists
            return df

        # If no existing file, proceed with analysis
        print("No cached analysis found. Running new analysis...")
        # 1. Identify match columns
        # These look like _matchID_1_L2_sv ... _matchID_36_L2_sv
        base_match_cols = [
            c
            for c in df.columns
            if c.startswith("_matchID_") and c.endswith(f"_{self.method}")
        ]
        crw_match_cols = [
            c
            for c in df.columns
            if c.startswith("CRW__matchID_") and c.endswith(f"_{self.method}")
        ]

        # 2. Safety checks
        self._safety_checks(df=df, base_cols=base_match_cols, crw_cols=crw_match_cols)
        n_recs = len(base_match_cols)

        print(f"Analyzing {n_recs} recommendation slots for {len(df)} voters...")

        # 3. Compute Metrics
        stats = {}
        df, stats = self._calculate_jaccard(
            df=df,
            base_cols=base_match_cols,
            crw_cols=crw_match_cols,
            n=n_recs,
            stats=stats,
        )

        df, stats = self._calculate_rank_metrics(
            df=df,
            base_cols=base_match_cols,
            crw_cols=crw_match_cols,
            stats=stats,
            n=n_recs,
        )

        # 5. Print and Save Stats
        self._print_and_save_stats(stats)
        self._visualize_changes(df)

        return df

    def _calculate_jaccard(self, df, base_cols, crw_cols, n, stats):
        # convert to numpy values once and iterate using zip
        base_matrix = df[base_cols].values
        crw_matrix = df[crw_cols].values

        jaccard_scores = []
        candidate_changes = []
        for b, c in zip(base_matrix, crw_matrix):
            # Convert to sets, stripping NaNs if the rankings are of uneven length
            s1 = set(b[~pd.isna(b)])
            s2 = set(c[~pd.isna(c)])

            intersection = len(s1 & s2)
            union = len(s1 | s2)

            # Handle empty sets (union=0) safely
            jaccard_scores.append(intersection / union if union > 0 else 1.0)
            candidate_changes.append(n - intersection)

        df["jaccard_similarity"] = jaccard_scores
        df["candidate_changes"] = candidate_changes

        # 3. Calculate and Save Stats
        stats.update(
            {
                "avg_jaccard": np.mean(jaccard_scores),
                "min_jaccard": np.min(jaccard_scores),
                "max_jaccard": np.max(jaccard_scores),
                "pct_changed": (np.array(jaccard_scores) < 1.0).sum() / len(df) * 100,
                "avg_candidate_changes": np.mean(candidate_changes),
                "max_candidate_changes": np.max(candidate_changes),
            }
        )
        return df, stats

    def _calculate_rank_metrics(self, df, base_cols, crw_cols, stats, n):
        # Convert to numpy for speed
        base_matrix = df[base_cols].values
        crw_matrix = df[crw_cols].values

        # A rank is the same only if ID at index i is the same for both matrices.
        matches = base_matrix == crw_matrix

        # Percent of positions that stayed the same per voter
        # (Number of True values / Total columns)
        stable_candidates_per_voter = matches.sum(axis=1)

        # Percentage of positions that changed per voter
        df["pct_rank_change"] = (n - stable_candidates_per_voter) / n * 100

        # Global stat: What % of voters had ANY rank change at all?
        # Note: If a candidate is swapped out, their rank changed.
        # If they just moved from 1 to 2, their rank also changed.
        stats.update(
            {
                "pct_voters_with_rank_change": (df["pct_rank_change"] > 0).mean() * 100,
                "avg_pct_rank_change": df["pct_rank_change"].mean(),
                "max_pct_rank_change": df["pct_rank_change"].max(),
            }
        )

        # print(
        #     f"Percentage of voters with any rank shuffle: {any_rank_change_voters:.2f}%"
        #     f"\nAverage percentage of rank change per voter: {avg_rank_change:.2f}%"
        #     f"\nMax percentage of rank change for any voter: {max_rank_change:.2f}%"
        # )

        return df, stats

    def _print_and_save_stats(self, stats):
        summary_line = (
            f"\n--- Recommendation Change Analysis Summary {self.config.data_year} {self.config.dist} Canton:{self.config.district if self.config.filter_districts else 'all'} ---\n"
            f"Average Jaccard Similarity: {stats['avg_jaccard']:.4f}\n"
            f"Min Jaccard Similarity:     {stats['min_jaccard']:.4f}\n"
            f"Max Jaccard Similarity:     {stats['max_jaccard']:.4f}\n"
            f"Average Candidate Changes:  {stats['avg_candidate_changes']:.2f}\n"
            f"Max Candidate Changes:      {stats['max_candidate_changes']}\n"
            f"Voters with different candidates in top recommendations:   {stats['pct_changed']:.2f}%\n"
            f"Voters with any rank change:                {stats['pct_voters_with_rank_change']:.2f}%\n"
            f"Average percentage of rank change per voter: {stats['avg_pct_rank_change']:.2f}%\n"
            f"Max percentage of rank change for any voter: {stats['max_pct_rank_change']:.2f}%\n"
            f"--------------------------------------\n"
        )
        print(summary_line)

        # Save to text file
        # TODO: hashing!
        d = self.config.district if self.config.filter_districts else "all"
        sub = (
            f"subset={self.config.subset_n}_"
            if self.config.subset_n is not None
            else ""
        )
        with open(
            self.output_dir
            / f"Jaccard_{self.config.data_year}_{self.config.dist}_{d}_{sub}.txt",
            "w",
        ) as f:
            f.write(summary_line)

    def _visualize_changes(self, df):
        plt.figure(figsize=(10, 6))
        sns.histplot(df["jaccard_similarity"], bins=20, color="skyblue")
        plt.title(f"Distribution of Jaccard Similarity (Baseline vs CRW)")
        plt.xlabel("Jaccard Similarity (1.0 = No Change in Set)")
        plt.ylabel("Number of Voters")
        plt.grid(axis="y", alpha=0.3)

        plot_path = self.output_dir / f"jaccard_dist_{self.method}.png"
        plt.savefig(plot_path)
        print(f"Visualization saved to {plot_path}")
        plt.close()

    def _safety_checks(self, df, base_cols, crw_cols):
        if len(base_cols) != len(crw_cols):
            raise ValueError(
                f"⚠️ Mismatch in number of match columns: {len(base_cols)} baseline vs {len(crw_cols)} CRW. Check column naming conventions."
            )
        if len(base_cols) == 0:
            raise ValueError(
                "⚠️ No match columns found! Check that your recommendation DataFrame has the expected column naming pattern."
            )
        if df[base_cols + crw_cols].isna().any().any():
            raise ValueError(
                "⚠️ Critical Error: NaN values detected in recommendation matches."
            )

    def _get_prefix(self):
        dist = self.config.dist
        data_year = self.config.data_year
        district = self.config.district if self.config.filter_districts else "all"
        alpha = self.config.alpha
        paper = self.config.crw_paper_choice
        subset = self.config.subset_n if self.config.subset_n is not None else None
        prefix = f"analysis_{data_year}_{dist}_a{alpha}_{paper}_canton={district}"
        if subset is not None:
            prefix += f"_subset={subset}"
        return prefix
