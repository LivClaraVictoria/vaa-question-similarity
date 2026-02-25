from configs.create_clones.base_clone_creator import *

data_year = 2023
selector_type = "high_voter_variance"
selector_params = {"n": 2}
clone_specs_config = [
    {"clone_type": "hard_paraphrase", "n_clones": 5, "flip_answers": False},
]
