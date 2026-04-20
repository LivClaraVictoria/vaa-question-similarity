# Distance-only config: QWEN3 on cleaned ZH data (topic classification instruction)
from configs.base_constants import *

data_choice = "cleaned"
dist = "QWEN3"
embedding_instruction = "Classify the following political question by its topic or policy area, ignoring whether the question is for or against the policy."
district = "ZH"
