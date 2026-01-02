import pandas as pd
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
        config.FAKE_RESULTS_DIR if config.data_choice == "fake" else config.RESULTS_DIR
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
    plt.figure(figsize=(12, 8))
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
            # Semantically Identical (Greens) -> Should be Close / High Similarity
            "Easy Paraphrase": "green",
            "Hard Paraphrase": "green",
            # Logically Connected / Opposites (Blues/Teals) -> Should be Relatively Close
            "Negation": "green",
            "Easy Paraphrase Negation": "green",
            "Hard Paraphrase Negation": "green",
            # Different Topics / Traps (Reds) -> Should be Far / Low Similarity
            "Trap": "red",
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
    )

    # 5. Add Reference Line (The "Visual Cliff")
    plt.axvline(x=threshold_val, color="grey", linestyle="--", label="Likely Threshold")

    # 6. Formatting
    plt.title(f"Metric Test: {metric_col} to Anchor {title_suffix}", fontsize=14)
    plt.xlabel(metric_col)
    plt.ylabel("Comparison Question")

    # Place legend outside the plot area so it doesn't cover data
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Category")
    plt.tight_layout()

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
