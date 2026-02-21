from configs.full_pipeline.cloned.base_cloned import *

clone_id = "identical_combinedvar_n10_e5_ZH"

CLEANED_DIR = CLONED_DIR / clone_id
QUESTIONS_2023_PATH = CLEANED_DIR / "df_questions.parquet"
QUESTIONS_2019_PATH = CLEANED_DIR / "df_questions19.parquet"
RAW_VOTERS_2023_PATH = CLEANED_DIR / "df_voters_topmatch.parquet"

data_choice = "cloned"


load_candidates = True
load_voters = True
