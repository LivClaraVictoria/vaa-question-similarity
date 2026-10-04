from configs.base_constants import *

data_year = 2023

selector_type = "manual"  # Options: "manual", "random", "high_candidate_variance", "combined_variance"
selector_params = {}

# Canton whose respondents the data-driven selectors (variance-based) see; "all" = national.
# Clones are always written for all respondents. Kept at "all" so existing clone sets reproduce;
# set to VALIDATION_DISTRICT for selections that must not see test-canton data.
district = "all"

# Default: 1 to 1 identical clones of all selected questions, cloned 10 times each
clone_specs_config = [
    {"clone_type": "identical", "n_clones": 10, "flip_answers": False},
]
