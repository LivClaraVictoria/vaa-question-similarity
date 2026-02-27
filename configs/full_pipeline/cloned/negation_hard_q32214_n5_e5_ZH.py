# corresponds to config "configs/create_clones/negation_hard_q32214_n5.py"
from configs.full_pipeline.cloned.base_cloned import *

clone_id = "negation_hard_q32214_n5"

CLEANED_DIR = CLONED_DIR / clone_id
QUESTIONS_2023_PATH = CLEANED_DIR / "df_questions.parquet"
QUESTIONS_2019_PATH = CLEANED_DIR / "df_questions19.parquet"
RAW_VOTERS_2023_PATH = CLEANED_DIR / "df_voters_topmatch.parquet"

dist = "E5"
district = "ZH"

n_recommendations = "all"
