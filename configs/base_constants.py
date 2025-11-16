# Base configuration for VAA Question Similarity Analysis
from pathlib import Path

# --- FILE PATHS ---
try:
    # get_ipython() only exists in IPython/Jupyter
    get_ipython()  # type: ignore

    # If that line didn't crash, we are in a notebook or IPython shell.
    # Use Path.cwd() to find the root.
    PROJECT_ROOT = Path.cwd()
    print("Loaded 'base_constants' in notebook mode.")

except NameError:
    # Standard script mode (.py file).
    # We can safely use __file__.
    PROJECT_ROOT = Path(__file__).parent.parent
    print("Loaded 'base_constants' in script mode.")


DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "experiment_results"

# Specific data files
FAKE_DATA_PATH = DATA_DIR / "fake" / "questions.csv"
RAW_CAND_2023_PATH = RAW_DIR / "sv23_ETHZ" / "23_ch_nr_candidates_de_2024_03_06.csv"
RAW_VOTERS_2023_PATH = CLEANED_DIR / "df_voters_topmatch.parquet"
RAW_CAND_2019_PATH = RAW_DIR / "smart vote data" / "smartvote_2019_Candidates_NR.csv"
RAW_VOTERS_2019_PATH = RAW_DIR / "smart vote data" / "sv_Voter_1xNR_V1_0_ethz.csv"

VOTERS_19_PREFIX = "df_voters19"
VOTERS_PREFIX = "df_voters"
CANDIDATES_19_PREFIX = "df_candidates19"
CANDIDATES_PREFIX = "df_candidates"

# --- ANALYSIS PARAMETERS ---
# Distance/similarity metric to use
"""
Options:
- "SBERT": Sentence-BERT embeddings with cosine similarity
"""
dist = "SBERT"

# Data source to use
data_choice = "fake"  # Options: "fake", "cleaned", "raw"

# Model parameters
learning_rate = 0.01
batch_size = 32

# SBERT specific parameters (if using SBERT)
sbert_model_name = "all-MiniLM-L6-v2"
