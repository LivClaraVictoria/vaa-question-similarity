import pandas as pd
from dependencies import add_candidate_voting_recommendations
from vqs.cache_management import CacheManager


class RecommendationEngine:
    def __init__(self, config, data_map):
        self.config = config
        self.dist_method = config.rec_dist_method
        self.df_candidates = data_map["candidates"].copy()
        self.df_voters = data_map["voters"].copy()

        # Set all weights to 1 (comment out if you want original weighting)
        weight_cols = self.df_voters.filter(like="weight_").columns
        self.df_voters[weight_cols] = 1

        # Parameters that affect the recommendationcalculations and should be included in the cache hash
        self.important_params_list = [
            "data_year",
            "dist",
            "alpha",
            "crw_paper_choice",
            "rec_dist_method",
            "n_recommendations",
            "subset_n",
            "filter_districts",
        ] + (["district"] if config.filter_districts else [])

    def run_baseline(self):
        """Calculates recommendations using standard 1.0/2.0 weights."""
        print(f"Running Baseline ({self.dist_method})...")
        # possibly need to set progress_bar=False to run on cluster
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
        # possibly need to set progress_bar=False to run on cluster
        return add_candidate_voting_recommendations(
            df_voters=voters_modified,
            df_candidates=self.df_candidates,
            distance_method=self.dist_method,
            n_recommendations=self.config.n_recommendations,
        )

    def evaluate_pipeline(self, df_weights) -> pd.DataFrame:
        # 1. Initialize Cache
        prefix = f"recs_{self.config.data_year}_{self.config.dist}_a{self.config.alpha}"
        prefix += f"_{self.config.district}" if self.config.filter_districts else ""
        cacher = CacheManager(
            config=self.config,
            cache_dir=self.config.RECOMMENDATION_CACHE_DIR,
            prefix=prefix,
            params_list=self.important_params_list,
        )

        # 2. Check Cache
        cached_df = cacher.load_if_exists()
        if cached_df is not None:
            return cached_df

        # 3. Compute if no cache found
        print("No cache found for recommendations. Computing pipeline...")
        baseline_recs_df: pd.DataFrame = self.run_baseline()  # type: ignore
        crw_recs_df: pd.DataFrame = self.run_crw(df_weights)  # type: ignore

        match_cols = [c for c in crw_recs_df.columns if "match" in c or "Dist" in c]
        crw_subset_prefixed = crw_recs_df[match_cols].add_prefix("CRW_")
        recommendation_df = baseline_recs_df.join(crw_subset_prefixed)
        print(
            "SUCCESS: Baseline and CRW recommendations calculated and combined into a single DataFrame."
        )

        print(recommendation_df.head(5))
        # 4. Save to Cache & return
        cacher.save(recommendation_df)
        return recommendation_df
