# Distance-only config: E5-ASYMMETRIC-INSTRUCT on cleaned ZH data (asymmetric + instruction-tuned)
from configs.base_constants import *

data_choice = "cleaned"
dist = "E5-ASYMMETRIC-INSTRUCT"
embedding_instruction = "Identify the political topic discussed in the following question, regardless of the stance or position taken."
district = "ZH"
