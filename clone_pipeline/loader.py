"""
Clone pipeline data loader: reads preprocessed SmartVote parquets from data/cleaned/ as the
clean source for clone generation (separate from the main pipeline's data_loader.py).
"""

import pandas as pd
from pathlib import Path


def _find_parquet(directory: Path, prefix: str) -> Path:
    files = list(directory.glob(f"{prefix}*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet file with prefix '{prefix}' in {directory}"
        )
    return files[0]


def load_clean_data(
    cleaned_dir: Path, cand_prefix: str, voters_prefix: str, questions_path: Path
) -> dict[str, pd.DataFrame]:
    """
    Loads the three base dataframes as independent copies.
    Always reads from the real cleaned dir, never from a cloned dir.
    """
    print(f"Loading source data from: {cleaned_dir}")
    return {
        "candidates": pd.read_parquet(_find_parquet(cleaned_dir, cand_prefix)),
        "voters": pd.read_parquet(_find_parquet(cleaned_dir, voters_prefix)),
        "questions": pd.read_parquet(questions_path),
    }
