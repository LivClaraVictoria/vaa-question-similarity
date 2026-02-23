from configs.full_pipeline.cloned.base_cloned import *

clone_id = "easy_paraphrase_combinedvar_n10"

CLEANED_DIR = CLONED_DIR / clone_id
QUESTIONS_2023_PATH = CLEANED_DIR / "df_questions.parquet"
QUESTIONS_2019_PATH = CLEANED_DIR / "df_questions19.parquet"
RAW_VOTERS_2023_PATH = CLEANED_DIR / "df_voters_topmatch.parquet"

dist = "SBERT"
district = "ZH"
n_recommendations = "all"
