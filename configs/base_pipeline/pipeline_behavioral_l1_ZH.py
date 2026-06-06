from configs.base_constants import *

# Behavioral L1 answer-distance metric (data-driven), Zurich.
# Distance is estimated from voters + candidates; for the deployment simulation the voter
# side is restricted to a `train_voter_fraction` sample (see deployment_simulation.py).

load_candidates = True
load_voters = True

data_choice = "cleaned"
dist = "BEHAVIORAL-L1"
correlation_answer_source = "both"
district = "ZH"

use_OG_weights = False
n_recommendations = "all"

# Out-of-sample deployment defaults (overridden per-seed by the deployment harness).
train_voter_fraction = 0.05
split_seed = 0
