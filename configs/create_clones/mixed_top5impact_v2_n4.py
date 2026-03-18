from configs.create_clones.base_clone_creator import *

data_year = 2023
selector_type = "manual"
selector_params = {"q_ids": [32214, 32228, 32234, 32240, 32261]}

# Mixed: 2 easy paraphrases + 2 negated easy paraphrases per question = 4 clones total
clone_specs_config = [
    {"clone_type": "easy_paraphrase", "n_clones": 2, "flip_answers": False},
    {"clone_type": "negation_easy", "n_clones": 2, "flip_answers": True},
]
