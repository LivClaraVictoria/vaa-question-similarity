import textwrap
import pandas as pd
import matplotlib
import json
import hashlib

matplotlib.use("Agg")  # for computations on the cluster
import matplotlib.pyplot as plt
import seaborn as sns
import traceback  # For printing detailed error logs
from pathlib import Path  # If you use Path objects for directories
from datetime import datetime  # If you use datetime for timestamps


def get_experiment_filename(config, timestamp, params_list) -> str:
    """
    Generates a descriptive but unique filename.
    Format: {dist}_{data}_{short_hash}_{DD_HHMM}.{ext}
    """
    # 1. Build the parameter dict for hashing (Same logic as CacheManager)
    params = {k: getattr(config, k) for k in params_list}
    param_str = json.dumps(params, sort_keys=True, default=str)
    full_hash = hashlib.md5(param_str.encode()).hexdigest()
    short_hash = full_hash[:8]

    # 3. Construct readable parts
    dist_method = config.dist
    data_year = config.data_year if config.data_choice != "fake" else "fake"
    canton = config.district if config.filter_districts else "all"

    filename = f"{dist_method}_{data_year}_{canton}_{short_hash}_{timestamp}.{config.results_file_type}"
    return filename


def save_results(
    df: pd.DataFrame, config, important_params_list: list[str]
) -> pd.DataFrame:
    """
    Sorts similarities, and saves to CSV or parquet.
    Returns the DataFrame for further use if needed.
    """
    # 0. For quick test runs
    if config.save_results is False:
        print("No question distance results will be saved.")
        return df
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # 1. Sort data
    if "Similarity" in df.columns:
        df.sort_values(by="Similarity", ascending=False, inplace=True)
    elif "Distance" in df.columns:
        df.sort_values(by="Distance", ascending=True, inplace=True)
    else:
        print("⚠️WARNING⚠️: No 'Similarity' or 'Distance' column found for sorting.")

    # 2. Define data paths
    output_dir = (
        config.FAKE_RESULTS_DIR
        if config.data_choice == "fake"
        else config.CLEANED_RESULTS_DIR
    )
    output_dir.mkdir(exist_ok=True)

    # 3. Visualize data
    if config.data_choice == "fake":
        df = df[
            df["Cat1"] == "ANCHOR"
        ]  # filter for clarity: only comparison to ANCHOR matters

        _plot_fake_results(
            df,
            config=config,
            timestamp=timestamp,
            params_list=important_params_list,
            output_dir=output_dir,
        )

    # Generate filename
    filename = get_experiment_filename(
        config=config, timestamp=timestamp, params_list=important_params_list
    )
    file_path = output_dir / filename

    # Save results
    if config.results_file_type == "parquet":
        df.to_parquet(file_path, index=False)
    else:  # default to csv
        df.to_csv(file_path, index=False)

    print(f"\nSuccess! Results saved to:")
    print(f"  -> {file_path}")
    return df


def _plot_fake_results(
    df: pd.DataFrame, config, timestamp, params_list, output_dir
) -> None:
    # 1. Setup Plot Dimensions
    plt.figure(figsize=(12, 9))
    sns.set_theme(style="whitegrid")

    # 2. Determine Color Palette based on 'Cat2'
    # We use Cat2 because Cat1 is always 'ANCHOR' in this filtered view
    hue_col = None
    palette = None

    if "Cat2" in df.columns:
        hue_col = "Cat2"
        # Define semantic palette matching your CSV categories
        palette = {
            "ANCHOR": "black",
            "Negation": "teal",
            "Easy Paraphrase": "seagreen",
            "Easy Paraphrase Negation": "mediumaquamarine",
            "Hard Paraphrase": "mediumseagreen",
            "Hard Paraphrase Negation": "lightseagreen",
            "Antonym": "cadetblue",
            "Syntax Trap": "firebrick",
            "Keyword Trap": "indianred",
        }
    else:
        print("⚠️ Warning: No 'Cat2' column found. Plot will not be color-coded.")

    # 3. Handle Metric Type (Distance vs Similarity)
    if "Distance" in df.columns:
        metric_col = "Distance"
        title_suffix = "(Lower is Better)"
        # Distance threshold hint (approximate)
        threshold_val = 0.5
    else:
        metric_col = "Similarity"
        title_suffix = "(Higher is Better)"
        # Similarity threshold hint (approximate)
        threshold_val = 0.7

    # 4. Create the Bar Chart
    ax = sns.barplot(
        data=df,
        x=metric_col,
        y="Qu2",  # We plot the Comparison Question on the Y-Axis
        hue=hue_col,
        palette=palette,
        dodge=False,  # Keeps bars thick and aligned
        width=0.5,
    )

    # 5. Add Reference Line (The "Visual Cliff")
    plt.axvline(x=threshold_val, color="grey", linestyle="--", label="Likely Threshold")

    # 6. Formatting
    plt.suptitle(
        f"{config.dist} on {config.data_choice} data {title_suffix}", fontsize=14
    )
    plt.xlabel(metric_col)
    plt.ylabel("Comparison Question")

    if "instruct" in config.dist.lower():
        # Safely get instruction (defaults to string if missing)
        instruction_text = getattr(
            config, "E5_instruction", "Instruction not found in config"
        )

        # Wrap text so it doesn't run off the plot (e.g., width 80 chars)
        wrapped_inst = "\n".join(
            textwrap.wrap(f"Instruction: {instruction_text}", width=80)
        )

        # Add as a smaller subtitle
        plt.title(wrapped_inst, fontsize=10, style="italic", pad=10, color="dimgrey")

    # Place legend outside the plot area so it doesn't cover data
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Category")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # 7. Save Plot
    # Uses the same timestamp/name as the CSV for easy matching
    plot_filename = get_experiment_filename(config, timestamp, params_list).replace(
        f".{config.results_file_type}", "_visualization.png"
    )

    plot_path = output_dir / plot_filename

    plt.savefig(plot_path, dpi=300)
    plt.close()  # Close the memory buffer to prevent leaks

    print(f"[ResultsManager] Visualization saved to: {plot_path}")
