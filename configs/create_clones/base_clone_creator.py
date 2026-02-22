from configs.base_constants import *

data_year = 2023

selector_type = "manual"  # Options: "manual", "random", "high_candidate_variance", "combined_variance"
selector_params = {}

# Default: 1 to 1 identical clones of all selected questions, cloned 10 times each
clone_specs_config = [
    {"clone_type": "identical", "n_clones": 10, "flip_answers": False},
]
