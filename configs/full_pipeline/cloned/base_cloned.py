from configs.base_constants import *

CLEANED_DIR = CLONED_DIR / "identical_q32214_n10"
QUESTIONS_2023_PATH = CLEANED_DIR / "df_questions.parquet"
QUESTIONS_2019_PATH = CLEANED_DIR / "df_questions19.parquet"
RAW_VOTERS_2023_PATH = CLEANED_DIR / "df_voters_topmatch.parquet"

data_choice = "cloned"
clone_id = "identical_q32214_n10"

load_candidates = True
load_voters = True
