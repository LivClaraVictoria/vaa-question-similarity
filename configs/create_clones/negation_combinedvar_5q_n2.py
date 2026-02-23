from configs.create_clones.base_clone_creator import *

data_year = 2023
selector_type = "combined_variance"
selector_params = {"n": 5}
clone_specs_config = [
    {"clone_type": "negation", "n_clones": 2, "flip_answers": True},
]
