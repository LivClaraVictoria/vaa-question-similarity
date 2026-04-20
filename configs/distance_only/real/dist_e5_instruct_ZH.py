# Distance-only config: E5-INSTRUCT on cleaned ZH data (instruction-tuned, topic-similarity prompt)
from configs.base_constants import *

data_choice = "cleaned"
dist = "E5-INSTRUCT"
embedding_instruction = "Identify the political topic discussed in the following question, regardless of the stance or position taken."
district = "ZH"
