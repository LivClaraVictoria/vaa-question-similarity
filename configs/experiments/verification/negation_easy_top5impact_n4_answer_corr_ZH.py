from configs.base_pipeline.base_cloned import *

clone_id = "negation_easy_top5impact_n4"

CLEANED_DIR = CLONED_DIR / clone_id
QUESTIONS_2023_PATH = CLEANED_DIR / "df_questions.parquet"
QUESTIONS_2019_PATH = CLEANED_DIR / "df_questions19.parquet"
RAW_VOTERS_2023_PATH = CLEANED_DIR / "df_voters_topmatch.parquet"

dist = "ANSWER-CORRELATION"
correlation_answer_source = "voters"
district = "ZH"

n_recommendations = "all"
