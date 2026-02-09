import pandas as pd
from dependencies import add_candidate_voting_recommendations


class RecommendationEngine:
    def __init__(self, config, data_map):
        self.config = config
        self.df_voters = data_map["voters"]
        self.df_candidates = data_map["candidates"]
        self.dist_method = config.rec_dist_method

    def run_baseline(self):
        """Calculates recommendations using standard 1.0/2.0 weights."""
        print(f"Running Baseline ({self.dist_method})...")
        return add_candidate_voting_recommendations(
            df_voters=self.df_voters.copy(),
            df_candidates=self.df_candidates,
            distance_method=self.dist_method,
            n_recommendations=self.config.n_recommendations,
        )

    def run_crw(self, df_weights):
        """Injects CRW weights, then calculates recommendations."""
        print(f"Running CRW ({self.dist_method})...")
        voters_modified = self.df_voters.copy()
        weight_lookup = df_weights.set_index("ID_question")["Weight"].to_dict()

        # INJECTION LOGIC
        for q_id, crw_val in weight_lookup.items():
            target_col = f"weight_{q_id}"
            if target_col in voters_modified.columns:
                voters_modified[target_col] *= crw_val
            else:
                print(
                    f"⚠️ Warning: Expected column '{target_col}' not found in voters DataFrame. Skipping weight injection for question ID {q_id}."
                )

        return add_candidate_voting_recommendations(
            df_voters=voters_modified,
            df_candidates=self.df_candidates,
            distance_method=self.dist_method,
            n_recommendations=self.config.n_recommendations,
        )

    def evaluate_pipeline(self, df_weights) -> pd.DataFrame:
        baseline_recs_df: pd.DataFrame = self.run_baseline()  # type: ignore
        crw_recs_df: pd.DataFrame = self.run_crw(df_weights)  # type: ignore

        match_cols = [c for c in crw_recs_df.columns if "match" in c or "Dist" in c]
        crw_subset_prefixed = crw_recs_df[match_cols].add_prefix("CRW_")
        recommendation_df = baseline_recs_df.join(crw_subset_prefixed)
        print(
            "SUCCESS: Baseline and CRW recommendations calculated and combined into a single DataFrame."
        )

        return recommendation_df
