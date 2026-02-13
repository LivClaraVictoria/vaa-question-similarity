import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # for computations on the cluster
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import hashlib
import time


class RecChangeAnalyzer:
    def __init__(self, config):
        self.config = config
        self.output_dir = getattr(config, f"{config.crw_paper_choice}_WEIGHT_DIR")
        # Create a specific cache folder
        self.cache_dir = self.config.RECOMMENDATION_ANALYSIS_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n = config.n_recommendations
        self.p = config.p_rbo

    def get_data_hash(self, df_weights):
        """Creates a unique hash based on ACTUAL weight values and key config params."""
        # 1. Content-based: The actual weight values
        # We sort by ID so the hash is stable even if the rows move around
        weight_content = (
            df_weights.sort_values("ID_question")["Weight"].astype(str).values.tobytes()
        )

        # 2. Context-based: Parameters that change the result logic
        config_str = (
            f"alpha{self.config.alpha}_"
            f"n{self.n}_"
            f"p{self.p}_"
            f"{self.config.rec_dist_method}"
        ).encode()

        # Combine both to ensure the cache is invalidated if data OR params change
        return hashlib.md5(weight_content + config_str).hexdigest()[:10]

    def calculate_rbo(self, list1, list2):
        """
        Calculates Rank-Biased Overlap (RBO).
        p=0.9 means the first 10 items carry ~86% of the weight.
        """
        s1 = set()
        s2 = set()
        rbo_sum = 0.0
        for i, (item1, item2) in enumerate(zip(list1, list2)):
            i += 1
            s1.add(item1)
            s2.add(item2)
            overlap = len(s1.intersection(s2))
            rbo_sum += (overlap / i) * (self.p ** (i - 1))
        return (1 - self.p) * rbo_sum

    def analyze(self, df_recommendations, df_weights):
        """Performs analysis and saves with hash-based versioning."""

        c
        data_hash = self.get_data_hash(df_weights)
        timestamp = time.strftime("%Y%m%d-%H%M")

        file_name = (
            f"rec_analysis_{self.config.crw_paper_choice}_"
            f"{self.config.rec_dist_method}_h{data_hash}.csv"
        )
        cache_path = self.cache_dir / file_name

        # --- CACHING LOGIC ---
        if cache_path.exists():
            print(f"Loading cached analysis from: {file_name}")
            return pd.read_csv(cache_path)

        print(f"No cache found. Calculating shifts (Hash: {data_hash})...")

        print(f"Analyzing shifts in top {self.n} recommendations...")

        # Base columns: _matchID_1_L2_sv, _matchID_2_L2_sv... (the original top-n candidates)
        base_cols = [
            f"_matchID_{i}_{self.config.rec_dist_method}" for i in range(1, self.n + 1)
        ]

        # CRW columns: CRW__matchID_1_L2_sv, CRW__matchID_2_L2_sv...
        crw_cols = [
            f"CRW__matchID_{i}_{self.config.rec_dist_method}"
            for i in range(1, self.n + 1)
        ]

        # 1. Jaccard Similarity (Unordered Set Overlap)
        df_recommendations["jaccard"] = df_recommendations.apply(
            lambda row: len(set(row[base_cols]) & set(row[crw_cols]))
            / len(set(row[base_cols]) | set(row[crw_cols])),
            axis=1,
        )

        # 2. RBO (Ordered Similarity)
        df_recommendations["rbo"] = df_recommendations.apply(
            lambda row: self.calculate_rbo(row[base_cols].values, row[crw_cols].values),
            axis=1,
        )

        # 3. Top-1 Displacement
        # How far down the CRW list did the original Baseline #1 go?
        def get_dist(row):
            target = row[
                f"_matchID_1_{self.config.rec_dist_method}"
            ]  # Original top-1 candidate
            crw_list = row[crw_cols].values
            indices = np.where(crw_list == target)[0]
            return (
                indices[0] if indices.size > 0 else self.n
            )  # Penalty if it disappeared

        df_recommendations["top_1_displacement"] = df_recommendations.apply(
            get_dist, axis=1
        )

        # Summary Statistics
        top_1_changed = (
            df_recommendations[f"_matchID_1_{self.config.rec_dist_method}"]
            != df_recommendations[f"CRW__matchID_1_{self.config.rec_dist_method}"]
        ).sum()
        print(f"--- Analysis Summary (n={self.n}) ---")
        print(
            f"Voters with new #1 Candidate: {top_1_changed} ({top_1_changed/len(df_recommendations):.2%})"
        )
        print(f"Mean RBO: {df_recommendations['rbo'].mean():.4f}")
        print(f"Mean Jaccard: {df_recommendations['jaccard'].mean():.4f}")

        # Add metadata columns for easier tracking later
        df_recommendations["analysis_hash"] = data_hash
        df_recommendations["analysis_at"] = timestamp

        # Save to cache
        df_recommendations.to_csv(cache_path, index=False)
        print(f"✅ Analysis cached at: {cache_path}")

        self.plot_stability(
            df=df_recommendations, data_hash=data_hash, timestamp=timestamp
        )
        return df_recommendations

    def plot_stability(self, df, data_hash, timestamp):
        """Plots with timestamp and hash info."""
        plt.figure(figsize=(10, 5))
        sns.kdeplot(df["jaccard"], label="Jaccard (Unordered)", fill=True)
        sns.kdeplot(df["rbo"], label="RBO (Ordered)", fill=True)

        # Add hash and timestamp to title or text box
        title = f"Stability: {self.config.crw_paper_choice} (n={self.n})\nHash: {data_hash} | {timestamp}"
        plt.title(title, fontsize=10)
        plt.xlabel("Similarity Score (1.0 = Identical)")
        plt.legend()

        # Save with hash to avoid overwriting different experiments
        plot_path = self.output_dir / f"similarity_dist_{data_hash}.png"
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.close()
