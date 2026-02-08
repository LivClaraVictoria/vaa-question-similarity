import pandas as pd
from dependencies import add_candidate_voting_recommendations


class RecommendationEngine:
    def __init__(self, config, data_map):
        self.config = config
        self.df_voters = data_map["voters"]
        self.df_candidates = data_map["candidates"]

    def run_baseline(self, dist_method="L2_sv"):
        """Calculates recommendations using standard 1.0/2.0 weights."""
        print(f"Running Baseline ({dist_method})...")
        return add_candidate_voting_recommendations(
            df_voters=self.df_voters.copy(),
            df_candidates=self.df_candidates,
            distance_method=dist_method,
            n_recommendations=self.config.n_recommendations,
        )

    def run_crw(self, df_weights, dist_method="L2_sv"):
        """Injects CRW weights, then calculates recommendations."""
        print(f"Running CRW ({dist_method})...")
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
            distance_method=dist_method,
            n_recommendations=self.config.n_recommendations,
        )

    def evaluate_pipeline(self, df_weights, dist_method="L2_sv") -> pd.DataFrame:
        baseline_recs: pd.DataFrame = self.run_baseline(dist_method=dist_method)  # type: ignore
        crw_recs_df: pd.DataFrame = self.run_crw(df_weights, dist_method=dist_method)  # type: ignore
        crw_prefixed_df: pd.DataFrame = crw_recs_df.add_prefix("CRW")
        recommendation_df: pd.DataFrame = baseline_recs.join(crw_prefixed_df)

        return recommendation_df
