from configs.create_clones.base_clone_creator import *

data_year = 2023
selector_type = "manual"
selector_params = {"q_ids": [32214, 32261, 32228, 32234, 32222]}

clone_specs_config = [
    {"clone_type": "hard_paraphrase", "n_clones": 4, "flip_answers": False},
]
