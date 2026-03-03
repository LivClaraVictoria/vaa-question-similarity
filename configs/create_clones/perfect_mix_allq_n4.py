from configs.create_clones.base_clone_creator import *

data_year = 2023
selector_type = "all"
selector_params = {}

# Perfect mix: 1 of each non-identical type per question = 4 clones total
clone_specs_config = [
    {"clone_type": "easy_paraphrase", "n_clones": 1, "flip_answers": False},
    {"clone_type": "hard_paraphrase", "n_clones": 1, "flip_answers": False},
    {"clone_type": "negation_easy", "n_clones": 1, "flip_answers": True},
    {"clone_type": "negation_hard", "n_clones": 1, "flip_answers": True},
]
