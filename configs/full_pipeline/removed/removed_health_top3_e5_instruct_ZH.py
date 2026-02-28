# Auto-generated config for reduced dataset: removed_health_top3
# Generated: 2026-02-28T18:50:46.475178
from configs.full_pipeline.cloned.base_cloned import *

clone_id = "removed_health_top3"

CLEANED_DIR = REMOVED_DIR / clone_id
QUESTIONS_2023_PATH = CLEANED_DIR / "df_questions.parquet"
QUESTIONS_2019_PATH = CLEANED_DIR / "df_questions19.parquet"
RAW_VOTERS_2023_PATH = CLEANED_DIR / "df_voters_topmatch.parquet"

dist = "E5-INSTRUCT"
embedding_instruction = "Identify the political topic discussed in the following question, regardless of the stance or position taken."
district = "ZH"

n_recommendations = "all"
