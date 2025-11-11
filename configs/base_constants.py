# paths.py
from pathlib import Path

# --- DEFAULT FILE PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiment_results"

# Specific data files
fake_data_path = DATA_DIR / "fake" / "questions.csv"
smartvote_data_path = DATA_DIR / "raw" / "smart vote data" / "df_Questions_2019.pk1"

# --- DEFAULT PARAMETERS ---
dist: str = "SBERT"
data_choice: str = "fake"
learning_rate: float = 0.01
batch_size: int = 32
