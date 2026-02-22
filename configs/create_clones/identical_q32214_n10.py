from configs.create_clones.base_clone_creator import *

data_year = 2023
selector_type = "manual"
selector_params = {"q_ids": [32214]}

clone_specs_config = [
    {"clone_type": "identical", "n_clones": 10, "flip_answers": False},
]
