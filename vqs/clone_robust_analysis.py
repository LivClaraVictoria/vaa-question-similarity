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

from vqs.result_management import ResultManager


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
def get_prefix(config) -> str:
    """
    Generates a descriptive but unique filename.
    Final Format: {dist}_{data}_{MMDD_HHMM}_{short_hash}.{ext}
    """
    dist_method = config.dist
    alpha = config.alpha
    data_year = config.data_year if config.data_choice != "fake" else "fake"
    paper_choice = config.crw_paper_choice

    filename = f"{data_year}_{dist_method}_a{alpha}_{paper_choice}"
    return filename


def save_reweighting_results(
    df: pd.DataFrame, config, important_params_list: list[str]
) -> pd.DataFrame:
    """
    Saves question weights and generates a distribution plot.
    """

    # 1. Sort by weight
    df.sort_values(by="Weight", ascending=True, inplace=True)

    # 2. Check for cached files
    method_dir = config.QU_WEIGHT_DIR / config.crw_paper_choice
    method_dir.mkdir(parents=True, exist_ok=True)
    prefix = get_prefix(config)
    rm = ResultManager(config, method_dir, important_params_list, prefix=prefix)

    exists = rm.exists()
    if exists:
        print(f"--- [Skip Save] Result with hash {rm.hash} already exists: ---")
        print(f"    -> {exists.name}")
        return df

    # 3. If no existing file, save new results

    file_path = rm.save(df=df, readable=True)

    _plot_weight_distribution(df, config, config.crw_paper_choice, file_path)  # type: ignore

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

    plot_path = file_path.parent / (file_path.stem + "_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
