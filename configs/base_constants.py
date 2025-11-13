# Base configuration for VAA Question Similarity Analysis
from pathlib import Path

# --- FILE PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "experiment_results"

# Specific data files
FAKE_DATA_PATH = DATA_DIR / "fake" / "questions.csv"
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
