import pandas as pd
from pathlib import Path
import time


def get_experiment_filename(config) -> str:
    """
    Generates a descriptive filename based on config parameters.
    Example: "similarity_SBERT_2023.csv"
    """
    # Fallback if 'data_year' isn't set, e.g., for fake data
    year = config.data_year
    dist_method = config.dist

    return f"{dist_method}_{year}.{config.results_file_type}"


def save_results(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Sorts similarities, and saves to CSV or parquet.
    Returns the DataFrame for further use if needed.
    """
    df.sort_values(by="Similarity", ascending=False, inplace=True)
    output_dir = config.RESULTS_DIR
    output_dir.mkdir(exist_ok=True)

    # Generate filename
    filename = get_experiment_filename(config)
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
