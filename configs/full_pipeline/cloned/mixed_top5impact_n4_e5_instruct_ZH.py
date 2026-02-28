# corresponds to config "configs/create_clones/mixed_top5impact_n4.py"
from configs.full_pipeline.cloned.base_cloned import *

clone_id = "mixed_top5impact_n4"

CLEANED_DIR = CLONED_DIR / clone_id
QUESTIONS_2023_PATH = CLEANED_DIR / "df_questions.parquet"
QUESTIONS_2019_PATH = CLEANED_DIR / "df_questions19.parquet"
RAW_VOTERS_2023_PATH = CLEANED_DIR / "df_voters_topmatch.parquet"

dist = "E5-INSTRUCT"
embedding_instruction = "Identify the political topic discussed in the following question, regardless of the stance or position taken."
district = "ZH"

n_recommendations = "all"
