# Base configuration for VAA Question Similarity Analysis
import os
from pathlib import Path
import sys

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

# --- CONSTANTS -----
# additional data paths
CLEANED_DIR = DATA_DIR / "cleaned"
RAW_DIR = DATA_DIR / "raw"
FAKE_DIR = DATA_DIR / "fake"
DATA_2023_DIR = RAW_DIR / "sv23_ETHZ"
DATA_2019_DIR = RAW_DIR / "smart vote data"

# experiment results path
RESULTS_DIR = PROJECT_ROOT / "experiment_results"
DIST_RESULTS_DIR = RESULTS_DIR / "distance_metric"
FAKE_RESULTS_DIR = DIST_RESULTS_DIR / "fake_results"
CLEANED_RESULTS_DIR = DIST_RESULTS_DIR / "cleaned_results"
QU_WEIGHT_DIR = RESULTS_DIR / "question_weighting_results"
P1_WEIGHT_DIR = QU_WEIGHT_DIR / "P1"
P2_WEIGHT_DIR = QU_WEIGHT_DIR / "P2"
RECOMMENDATION_RESULTS_DIR = RESULTS_DIR / "recommendation_results"
COMPARISON_RESULTS_DIR = RESULTS_DIR / "comparator_results"


# Specific data files
FAKE_DATA_FILE = FAKE_DIR / "fake_questions.csv"
TIMESTAMP_FILE = DATA_2023_DIR / "sv23 Voters-NR_time_recDATE.csv"

# raw data file paths
RAW_CAND_2023_PATH = DATA_2023_DIR / "23_ch_nr_candidates_de_2024_03_06.csv"
RAW_VOTERS_2023_PATH = CLEANED_DIR / "df_voters_topmatch.parquet"
RAW_CAND_2019_PATH = DATA_2019_DIR / "smartvote_2019_candidates_NR.csv"
RAW_VOTERS_2019_PATH = DATA_2019_DIR / "sv_Voter_1xNR_V1_0_ethz.csv"

# cleaned data paths
VOTERS_19_PREFIX = "df_voters19-"
VOTERS_PREFIX = "df_voters-"
CANDIDATES_19_PREFIX = "df_candidates19-"
CANDIDATES_PREFIX = "df_candidates-"

# questions
QUESTIONS_2023_PATH = CLEANED_DIR / "df_questions.parquet"
QUESTIONS_2019_PATH = CLEANED_DIR / "df_questions19.parquet"

# cache paths
CACHE_DIR = PROJECT_ROOT / "cache"
DISTANCE_CACHE_DIR = CACHE_DIR / "distance_calculations"
RECOMMENDATION_CACHE_DIR = CACHE_DIR / "recommendations"


# --- FROM DUSTIN'S BASE CONSTANTS.PY ---
# District to ID mapping (taken from constants.py in Dustin's repo)
# Important note on column names in the dataframes: in 2023 dataset, it's "ID_district" for candidates, "districtID" for voters, in 2019 dataset, it's "ID_district" for both
DISTRICT2ID = {
    "AG": 927,
    "AR": 928,
    "AI": 929,
    "BL": 930,
    "BS": 931,
    "BE": 932,
    "FR": 933,
    "GE": 934,
    "GL": 935,
    "GR": 936,
    "JU": 937,
    "LU": 938,
    "NE": 939,
    "NW": 940,
    "OW": 941,
    "SH": 942,
    "SZ": 943,
    "SO": 944,
    "SG": 945,
    "TI": 946,
    "TG": 947,
    "UR": 948,
    "VD": 949,
    "VS": 950,
    "ZG": 951,
    "ZH": 952,
}
DISTRICT2ID19 = {
    "AG": 1,
    "AR": 2,
    "AI": 3,
    "BL": 4,
    "BS": 5,
    "BE": 6,
    "FR": 7,
    "GE": 8,
    "GL": 9,
    "GR": 10,
    "JU": 11,
    "LU": 12,
    "NE": 13,
    "NW": 14,
    "OW": 15,
    "SH": 16,
    "SZ": 17,
    "SO": 18,
    "SG": 19,
    "TI": 20,
    "TG": 21,
    "UR": 22,
    "VD": 23,
    "VS": 24,
    "ZH": 25,
    "ZG": 26,
}

SEATS_PER_CANTON = {
    "ZH": 36,
    "BE": 24,
    "LU": 9,
    "UR": 1,
    "SZ": 4,
    "OW": 1,
    "NW": 1,
    "GL": 1,
    "ZG": 3,
    "FR": 7,
    "SO": 6,
    "BS": 4,
    "BL": 7,
    "SH": 2,
    "AR": 1,
    "AI": 1,
    "SG": 12,
    "GR": 5,
    "AG": 16,
    "TG": 6,
    "TI": 8,
    "VD": 19,
    "VS": 8,
    "NE": 4,
    "GE": 12,
    "JU": 2,
}

SEATS_PER_CANTON19 = {
    "ZH": 35,
    "BE": 24,
    "LU": 9,
    "UR": 1,
    "SZ": 4,
    "OW": 1,
    "NW": 1,
    "GL": 1,
    "ZG": 3,
    "FR": 7,
    "SO": 6,
    "BS": 5,
    "BL": 7,
    "SH": 2,
    "AR": 1,
    "AI": 1,
    "SG": 12,
    "GR": 5,
    "AG": 16,
    "TG": 6,
    "TI": 8,
    "VD": 19,
    "VS": 8,
    "NE": 4,
    "GE": 12,
    "JU": 2,
}
# --------------------------------------------------------------------------------------

# --- GENERAL PARAMETERS ---

# Data source to use
data_choice = "cleaned"  # Options: "fake", "cleaned", "raw"
data_year = 2023  # Options: 2019, 2023

load_voters = False  # false by default, set true for methods where you want to look into correlation
load_candidates = False  # false by default, set true for methods where you want to look into correlation
results_file_type = "parquet"  # "csv" to read file in vscode (but slower)

# For debug runs (set False for quick testing without saving)
save_results = True

# Canton Filtering
filter_districts = False
district = "ZH"  # e.g. "ZH", "BE", etc. (only relevant if filter_districts=True)

# Subsetting
subset_n = None  # for quick testing: set to an integer to subset the data, or None to use full data

# Method Choice for clone robust weighting
crw_paper_choice = "P2"  # Options: "P1", "P2"

# Keep track of CLI overrides
overrides: list = []


# --- DISTANCE/SIMILARITY PARAMETERS ---
# Distance/similarity metric to use
# Create new config files for different metrics if needed, or just override in command line

# terminal config instruction: python main.py --config configs/base_constants.py dist=SBERT_EUCLIDEAN for instance
"""
Options for dist:
- "SBERT": Sentence-BERT embeddings with cosine similarity
- "SBERT_EUCLIDEAN": Sentence-BERT embeddings with Euclidean distance (on normalized embeddings): equivalent to sqrt(2 - 2*cosine_similarity)
- "E5": E5 model embeddings with euclidean distance on normalized embeddings
- "E5-asymmetric": E5 model retrieval-style (query/passage) with eucdlidean distance on normalized embeddings
- "E5-instruct": E5 model (symmetric: query/query) with instructions, euclidean distance on normalized embeddings
- "E5-asymmetric-instruct": E5 model retrieval-style with instructions, euclidean distance on normalized embeddings
"""
dist = "SBERT"
E5_instruction: str | None = (
    None  # "Retrieve political questions that deal with the same topic."  # "Retrieve semantically similar political questions."
)


# --- CLONE-ROBUST WEIGHTING PARAMETERS ---
apply_clone_robust_weighting = True  # whether to apply the method at all
alpha: float = 0.6  # locality parameter, r in [0, alpha]

# --- RECOMMENDATION ENGINE PARAMETERS ---
# whether to use original weights (1.0 or 2.0) or set all weights to 1.0.
use_OG_weights = False

n_recommendations: str | int | None = None  # Options: "all", int or None.
# how many candidates to recommend per voter. If None, n_recommendation reflects size of list of that canton.

# Seats per canton 2019:
#     "ZH": 35,
#     "BE": 24,
#     "LU": 9,
#     "UR": 1,
#     "SZ": 4,
#     "OW": 1,
#     "NW": 1,
#     "GL": 1,
#     "ZG": 3,
#     "FR": 7,
#     "SO": 6,
#     "BS": 5,
#     "BL": 7,
#     "SH": 2,
#     "AR": 1,
#     "AI": 1,
#     "SG": 12,
#     "GR": 5,
#     "AG": 16,
#     "TG": 6,
#     "TI": 8,
#     "VD": 19,
#     "VS": 8,
#     "NE": 4,
#     "GE": 12,
#     "JU": 2,

# Seats per canton 2023:
#     "ZH": 36,
#     "BE": 24,
#     "LU": 9,
#     "UR": 1,
#     "SZ": 4,
#     "OW": 1,
#     "NW": 1,
#     "GL": 1,
#     "ZG": 3,
#     "FR": 7,
#     "SO": 6,
#     "BS": 4,
#     "BL": 7,
#     "SH": 2,
#     "AR": 1,
#     "AI": 1,
#     "SG": 12,
#     "GR": 5,
#     "AG": 16,
#     "TG": 6,
#     "TI": 8,
#     "VD": 19,
#     "VS": 8,
#     "NE": 4,
#     "GE": 12,
#     "JU": 2,

# Taken from constants.py in Dustin's repo, which in turn is based on the official seat distribution for the Swiss National Council elections. Note that the number of seats per canton can change slightly from election to election based on population changes, so these numbers are specific to the 2019 and 2023 elections.


"""
(see https://gitlab.ethz.ch/disco-students/fs24/recommender-systems-for-politics or the submodule for more info)
Options for rec_dist_method:
"L2",
"L2_sv",
"L1",
"AC",
"angular_unweighted",
"angular",
"mahalanobis_unweighted",
"DM_L1",
"DM_L1_BONUS",
"DM_L2",
"DM_HYBRID",
"DM_DIRECTIONAL"
"""
rec_dist_method = "L2_sv"  # which distance method to use for recommendations

# --- RECOMMENDATION ANALYSIS PARAMETERS ---
p_rbo = 0.9  # RBO parameter: how steeply to discount lower ranks (0.9 means top 10 items get ~86% of the weight)
