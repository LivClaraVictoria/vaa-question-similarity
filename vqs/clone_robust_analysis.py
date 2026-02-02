import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from datetime import datetime
from pathlib import Path


def get_weighting_filename(config, method_key: str) -> str:
    """
    Generates a descriptive filename with a high-precision timestamp.
    Example: CRW_SBERT_a0.5_2026_20260202_104522.csv
    """
    # Precision down to the second ensures unique files for every run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    year = config.data_year if config.data_choice != "fake" else "fake"
    dist_method = config.dist
    alpha = getattr(config, "alpha", "N-A")

    return f"{method_key}_{dist_method}_a{alpha}_{year}_{timestamp}.{config.results_file_type}"


def save_reweighting_results(
    df: pd.DataFrame, config, method_key: str = "P2"
) -> pd.DataFrame:
    """
    Saves question weights and generates a distribution plot.
    Sorted ascending to highlight 'redundancy discounts' at the top.
    """
    # 1. Sort by Weight (Ascending)
    df.sort_values(by="Weight", ascending=True, inplace=True)

    # 2. Setup Directory (Using config.QU_WEIGHT_DIR)
    method_dir = getattr(
        config, f"{method_key.upper()}_WEIGHT_DIR", config.QU_WEIGHT_DIR / method_key
    )
    method_dir.mkdir(parents=True, exist_ok=True)

    # 3. Generate Filename and Path
    filename = get_weighting_filename(config, method_key)
    file_path = method_dir / filename

    # 4. Save Data
    if config.results_file_type == "parquet":
        df.to_parquet(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)

    # 5. Visualize
    _plot_weight_distribution(df, config, method_key, file_path)

    print(f"\n[WeightManager] Success! Results saved to:")
    print(f"  -> {file_path}")
    print(f"  -> Method: {method_key} | Total Nodes: {len(df)}")

    return df


def _plot_weight_distribution(
    df: pd.DataFrame, config, method_key: str, file_path: Path
) -> None:
    """
    Creates a bar chart of weights to visualize the 'redundancy discount'.
    """
    # Dynamic height based on number of questions to keep it readable
    plot_height = max(8, len(df) * 0.25)
    plt.figure(figsize=(12, plot_height))
    sns.set_theme(style="whitegrid")

    # Horizontal bar chart
    sns.barplot(
        data=df,
        x="Weight",
        y="Question",
        palette="viridis",
        hue="Question",
        legend=False,
    )

    # Reference line at 1.0 (The 'Neutral' baseline for unique items)
    plt.axvline(x=1.0, color="firebrick", linestyle="--", label="Baseline (Unique)")

    # Titles and labels
    alpha_val = getattr(config, "alpha", "N/A")
    plt.suptitle(
        f"{method_key} Weights: {config.dist} (alpha={alpha_val})", fontsize=14
    )
    plt.xlabel("Weight (Lower = More Redundant)")
    plt.ylabel("Question")
    plt.legend(loc="lower right")

    # Instruction text handling
    if "instruct" in str(config.dist).lower():
        inst = getattr(config, "E5_instruction", "No instruction found")
        wrapped_inst = "\n".join(textwrap.wrap(f"Instruction: {inst}", width=80))
        plt.title(wrapped_inst, fontsize=9, style="italic", color="dimgrey", pad=10)

    plt.subplots_adjust(left=0.3, top=0.9, right=0.95, bottom=0.1)

    # Save plot
    plot_path = file_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
