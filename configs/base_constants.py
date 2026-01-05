# Base configuration for VAA Question Similarity Analysis
import os
from pathlib import Path

# --- FILE PATHS ---
# root directory
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

# data directory (changes if on cluster)
cluster_path_env = os.getenv("CLUSTER_DATA_PATH")

if cluster_path_env:
    DATA_DIR = Path(cluster_path_env)
    print(f"Cluster environment detected! Using data path: {DATA_DIR}")
else:
    DATA_DIR = PROJECT_ROOT / "data"
    print(f"Local environment detected. Using data path: {DATA_DIR}")

# additional data paths
CLEANED_DIR = DATA_DIR / "cleaned"
RAW_DIR = DATA_DIR / "raw"
FAKE_DIR = DATA_DIR / "fake"
DATA_2023_DIR = RAW_DIR / "sv23_ETHZ"
DATA_2019_DIR = RAW_DIR / "smart vote data"

# experiment results path
RESULTS_DIR = PROJECT_ROOT / "experiment_results"
FAKE_RESULTS_DIR = RESULTS_DIR / "fake_results"


# Specific data files
FAKE_DATA_FILE = FAKE_DIR / "fake_questions.csv"
TIMESTAMP_FILE = DATA_2023_DIR / "sv23 Voters-NR_time_recDATE.csv"

# raw data file paths
RAW_CAND_2023_PATH = DATA_2023_DIR / "23_ch_nr_candidates_de_2024_03_06.csv"
RAW_VOTERS_2023_PATH = CLEANED_DIR / "df_voters_topmatch.parquet"
RAW_CAND_2019_PATH = DATA_2019_DIR / "smartvote_2019_candidates_NR.csv"
RAW_VOTERS_2019_PATH = DATA_2019_DIR / "sv_Voter_1xNR_V1_0_ethz.csv"

# cleaned data paths
VOTERS_19_PREFIX = "df_voters19"
VOTERS_PREFIX = "df_voters"
CANDIDATES_19_PREFIX = "df_candidates19"
CANDIDATES_PREFIX = "df_candidates"

# questions
QUESTIONS_2023_PATH = CLEANED_DIR / "df_questions.parquet"
QUESTIONS_2019_PATH = CLEANED_DIR / "df_questions19.parquet"
# --------------------------------------------------------------------------------------

# --- ANALYSIS PARAMETERS ---
# Distance/similarity metric to use
# Create new config files for different metrics if needed, or just override in command line
"""
Options:
- "SBERT": Sentence-BERT embeddings with cosine similarity
- "SBERT_EUCLIDEAN": Sentence-BERT embeddings with Euclidean distance (on normalized embeddings): equivalent to sqrt(2 - 2*cosine_similarity)
- "E5": E5 model embeddings with euclidean distance on normalized embeddings
- "E5-asymmetric": E5 model retrieval-style (query/passage) with eucdlidean distance on normalized embeddings
- "E5-instruct": E5 model with instructions, euclidean distance on normalized embeddings
"""
dist = "SBERT"
E5_instruction: str = "Political question"  # TODO: actual instruction handling

# Data source to use
data_choice = "cleaned"  # Options: "fake", "cleaned", "raw"
data_year = "2023"  # Options: "2019", "2023"

load_voters = False  # false by default, set true for methods where you want to look into correlation
load_candidates = False  # false by default, set true for methods where you want to look into correlation
results_file_type = "csv"  # "parquet" for using voter dataset

# Model parameters
learning_rate = 0.01
batch_size = 32

# SBERT specific parameters (if using SBERT)
sbert_model_name = "all-MiniLM-L6-v2"
