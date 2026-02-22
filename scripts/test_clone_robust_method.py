import sys
import os

# Adds the parent directory of 'scripts' to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from vqs.clone_robust_weighting import CloneRobustReweighter

# Define a simple scenario: 1 unique point, 2 clones
data = {
    "Qu1": ["Unique", "Clone1", "Unique"],
    "Qu2": ["Clone1", "Clone2", "Clone2"],
    "Distance": [1.0, 0.0, 1.0],  # Clone1 and Clone2 are identical
}
df_test = pd.DataFrame(data)


class Config:
    alpha = 0.5


reweighter = CloneRobustReweighter(Config())
results = reweighter.reweight(df_test)

print(results)
print(f"Total Weight Sum: {results['Weight'].sum()}")  # Should be 3.0
