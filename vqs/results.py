import textwrap
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # for computations on the cluster
import matplotlib.pyplot as plt
import seaborn as sns
import traceback  # For printing detailed error logs
from pathlib import Path  # If you use Path objects for directories
from datetime import datetime  # If you use datetime for timestamps


def get_experiment_filename(config, timestamp) -> str:
    """
    Generates a descriptive filename based on config parameters.
    Example: "similarity_SBERT_2023.csv"
    """
    # Fallback if 'data_year' isn't set, e.g., for fake data
    if config.data_choice == "fake":
        year = timestamp
    else:
        year = config.data_year

    dist_method = config.dist

    return f"{dist_method}_{year}.{config.results_file_type}"


def save_results(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Sorts similarities, and saves to CSV or parquet.
    Returns the DataFrame for further use if needed.
    """
    # For quick test runs
    if config.save_results is False:
        print("No results will be saved.")
        return df
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # 1. Sort data
    if "Similarity" in df.columns:
        df.sort_values(by="Similarity", ascending=False, inplace=True)
    elif "Distance" in df.columns:
        df.sort_values(by="Distance", ascending=True, inplace=True)
    else:
        print("⚠️WARNING⚠️: No 'Similarity' or 'Distance' column found for sorting.")

    # 2. Visualize data
    if config.data_choice == "fake":
        df = df[
            df["Cat1"] == "ANCHOR"
        ]  # filter for clarity: only comparison to ANCHOR matters

        _plot_fake_results(df, config, timestamp)

    # 3. Save data
    output_dir = (
        config.FAKE_RESULTS_DIR
        if config.data_choice == "fake"
        else config.CLEANED_RESULTS_DIR
    )
    output_dir.mkdir(exist_ok=True)

    # Generate filename
    filename = get_experiment_filename(config=config, timestamp=timestamp)
    file_path = output_dir / filename

    # Save results
    if config.results_file_type == "parquet":
        df.to_parquet(file_path, index=False)
    else:  # default to csv
        df.to_csv(file_path, index=False)

    print(f"\nSuccess! Results saved to:")
    print(f"  -> {file_path}")
    print(f"  -> Total Pairs: {len(df)}")

    return df


def _plot_fake_results(df: pd.DataFrame, config, timestamp) -> None:
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
    plot_filename = get_experiment_filename(config, timestamp).replace(
        f".{config.results_file_type}", "_visualization.png"
    )

    # copied, bad style TODO fix: clean up fake and cleaned separation
    output_dir = (
        config.FAKE_RESULTS_DIR if config.data_choice == "fake" else config.RESULTS_DIR
    )
    output_dir.mkdir(exist_ok=True)

    plot_path = output_dir / plot_filename

    plt.savefig(plot_path, dpi=300)
    plt.close()  # Close the memory buffer to prevent leaks

    print(f"[ResultsManager] Visualization saved to: {plot_path}")
