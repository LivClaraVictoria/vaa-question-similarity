import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class RecChangeAnalyzer:
    def __init__(self, config):
        self.config = config
        self.output_dir = getattr(config, f"{config.method_choice}_WEIGHT_DIR")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_and_save(self, df: pd.DataFrame, dist_method: str):
        """Main entry point for auditing results."""
        # 1. Calculate basic shift metrics
        id_col = f"_matchID_1_{dist_method}"
        df["Shifted"] = df[id_col] != df[f"CRW{id_col}"]

        shift_pct = df["Shifted"].mean() * 100
        print(f"📊 Overall Shift: {shift_pct:.2f}% of voters saw a new #1 candidate.")

        # 2. Store the Mega-DataFrame
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        file_path = (
            self.output_dir
            / f"recommendation_{self.config.method_choice}_{dist_method}_a{self.config.alpha}_{self.config.data_year}_{timestamp}.parquet"
        )
        df.to_parquet(file_path)

        # 3. Visualize
        self._plot_shift_summary(df, dist_method)

    def _plot_shift_summary(self, df: pd.DataFrame, dist_method: str):
        """Generates a simple visualization of match distance changes."""
        dist_col = f"_matchDist_1_{dist_method}"

        plt.figure(figsize=(10, 6))
        sns.kdeplot(df[dist_col], label="Baseline", fill=True)  # type: ignore
        sns.kdeplot(df[f"CRW{dist_col}"], label="CRW", fill=True)  # type: ignore
        plt.title("Distribution of Top Match Distances")
        plt.xlabel("Distance (Lower = Better Match)")
        plt.legend()

        plt.savefig(
            self.output_dir
            / f"distance_distribution_{self.config.method_choice}_{dist_method}_a{self.config.alpha}_{self.config.data_year}.png"
        )
        plt.close()
