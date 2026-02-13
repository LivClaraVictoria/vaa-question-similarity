import pandas as pd
import matplotlib
import json
import hashlib

matplotlib.use("Agg")  # for cluster
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from datetime import datetime
from pathlib import Path


def get_hash(config, params_list) -> str:
    """
    Generates a short hash based on important parameters from the config.
    Identical to the version used in distance results.
    """
    sorted_params_list = sorted(params_list)
    params = {k: getattr(config, k) for k in sorted_params_list}
    param_str = json.dumps(params, sort_keys=True, default=str)
    full_hash = hashlib.md5(param_str.encode()).hexdigest()
    return full_hash[:8]


# TODO: filename format
def get_filename(config, timestamp, hash) -> str:
    """
    Generates a descriptive but unique filename.
    Format: {dist}_{data}_{canton}_{MMDD_HHMM}_{short_hash}.{ext}
    """
    dist_method = config.dist
    alpha = config.alpha
    data_year = config.data_year if config.data_choice != "fake" else "fake"
    paper_choice = config.crw_paper_choice

    filename = f"{data_year}_{dist_method}_a{alpha}_{paper_choice}__{timestamp}_{hash}.{config.results_file_type}"
    return filename


def save_reweighting_results(
    df: pd.DataFrame, config, important_params_list: list[str]
) -> pd.DataFrame:
    """
    Saves question weights and generates a distribution plot.
    """
    # 1. Sort by weight
    df.sort_values(by="Weight", ascending=True, inplace=True)

    # 2. For quick test runs
    if config.save_results is False:
        print("No reweighting results will be saved.")
        return df

    # 3. Define output directory
    method_dir = config.QU_WEIGHT_DIR / config.crw_paper_choice
    method_dir.mkdir(parents=True, exist_ok=True)

    # 4. Get hash and check for existing file to avoid duplicates
    hash = get_hash(config=config, params_list=important_params_list)
    existing_files = list(method_dir.glob(f"*{hash}.*"))

    if existing_files:
        print(f"--- [Skip Save] Result with hash {hash} already exists: ---")
        print(f"    -> {existing_files[0].name}")
        return df

    # 5. If no existing file, save new results
    timestamp = datetime.now().strftime("%m%d_%H%M")
    filename = get_filename(config=config, timestamp=timestamp, hash=hash)
    file_path = method_dir / filename

    if config.results_file_type == "parquet":
        df.to_parquet(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)

    _plot_weight_distribution(df, config, config.crw_paper_choice, file_path)

    print(f"\n[WeightManager] Success! Results saved to: {file_path}")
    return df


def _plot_weight_distribution(
    df: pd.DataFrame, config, p_choice: str, file_path: Path
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
        f"{p_choice} Weights: {config.dist} (alpha={getattr(config, 'alpha', 'N/A')})",
        fontsize=16,
    )
    plt.xlabel("Weight (Lower = More Redundant)")

    plot_path = file_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
