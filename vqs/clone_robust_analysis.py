import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from datetime import datetime
from pathlib import Path


def get_weighting_filename(config, method_key: str) -> str:
    """
    Generates a descriptive filename with a high-precision timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    year = getattr(config, "data_year", "fake")
    dist_method = config.dist
    alpha = getattr(config, "alpha", "N-A")

    return f"{method_key}_{dist_method}_a{alpha}_{year}_{timestamp}.{config.results_file_type}"


def save_reweighting_results(
    df: pd.DataFrame, config, method_key: str = "CRW"
) -> pd.DataFrame:
    """
    Saves question weights and generates a distribution plot.
    """
    df.sort_values(by="Weight", ascending=True, inplace=True)

    # Use the specific config path you mentioned
    method_dir = config.QU_WEIGHT_DIR / method_key
    method_dir.mkdir(parents=True, exist_ok=True)

    filename = get_weighting_filename(config, method_key)
    file_path = method_dir / filename

    if config.results_file_type == "parquet":
        df.to_parquet(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)

    _plot_weight_distribution(df, config, method_key, file_path)

    print(f"\n[WeightManager] Success! Results saved to: {file_path}")
    return df


def _plot_weight_distribution(
    df: pd.DataFrame, config, method_key: str, file_path: Path
) -> None:
    """
    VAA-optimized plot with greedy margins and right-aligned wrapped text.
    """
    display_df = df.copy()

    # 1. Wide wrap for 'greedy' extension to the left
    wrap_width = 90
    display_df["Question_Wrapped"] = display_df["Question"].apply(
        lambda x: "\n".join(textwrap.wrap(str(x), width=wrap_width))
    )

    plot_height = max(10, len(display_df) * 0.7)
    plt.figure(figsize=(18, plot_height))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(
        data=display_df,
        x="Weight",
        y="Question_Wrapped",
        palette="viridis",
        hue="Question_Wrapped",
        legend=False,
    )

    # 2. Force right alignment to anchor text to the bars
    ax.set_yticklabels(ax.get_yticklabels(), ha="right", va="center", fontsize=10)
    ax.tick_params(axis="y", which="major", pad=10)

    # 3. 1.0 Baseline
    plt.axvline(x=1.0, color="firebrick", linestyle="--", label="Baseline (Unique)")

    # 4. Greedy margin (55% for text)
    plt.subplots_adjust(left=0.40, top=0.92, right=0.95, bottom=0.05)

    plt.suptitle(
        f"{method_key} Weights: {config.dist} (alpha={getattr(config, 'alpha', 'N/A')})",
        fontsize=16,
    )
    plt.xlabel("Weight (Lower = More Redundant)")

    plot_path = file_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
