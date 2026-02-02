import pandas as pd
import numpy as np
import warnings
from typing import Any, Sequence


class CloneRobustReweighter:
    """
    Implements General Clone-Robust Weighting Functions from (TODO: Add Citation).
    Balances importance by integrating graph-based weights over distance thresholds.
    """

    def __init__(self, config: Any):
        # Using getattr to support config objects with safe defaults
        self.alpha = getattr(config, "alpha", 1.0)

        # The probability density function v(r). Default is Uniform: 1/alpha.
        self.v_func = getattr(
            config, "v_func", lambda r, alpha: 1.0 / alpha
        )  # for now assume uniform

        # Swappable graph weighting strategy w. Default is Class-Uniform.
        self.weighting_func = getattr(
            config, "weighting_func", self._class_uniform_weighting_fn
        )

    def _class_uniform_weighting_fn(self, adj: np.ndarray) -> np.ndarray:
        """
        Calculates w_CU: Each equivalence class accounts for the same
        """
        # A signature is the node's closed neighborhood N_G[x].
        adj_rows = [tuple(row) for row in adj]  # make immutable so it can be a dict key
        class_counts = {}

        # Pass 1: Group signatures into Equivalence Classes
        for row in adj_rows:
            class_counts[row] = class_counts.get(row, 0) + 1

        num_classes = len(class_counts)  # |V/≡G|
        # Pass 2: Calculate weights: 1 / (|V/≡G| * |[x]G|)
        weights = np.array(
            [1.0 / (num_classes * class_counts[row]) for row in adj_rows]
        )
        return weights

    def _get_distance_matrix(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, Sequence[str]]:
        """Extracts distances and handles Similarity-to-Distance conversion."""

        # Strict check for required columns
        if "Distance" in df.columns:
            val_col = "Distance"
            is_similarity = False
        elif "Similarity" in df.columns:
            val_col = "Similarity"
            is_similarity = True
        else:
            warnings.warn("⚠️ WARNING ⚠️: No 'Similarity' or 'Distance' column found.")
            val_col = df.columns[-1]
            is_similarity = False

        # Identify unique nodes (the set S)
        nodes = sorted(list(set(df["Qu1"]).union(set(df["Qu2"]))))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        dist_matrix = np.zeros((n, n))

        for _, row in df.iterrows():
            u, v = node_to_idx[row["Qu1"]], node_to_idx[row["Qu2"]]
            val = row[val_col]

            if is_similarity:
                # Euclidean distance on normalized vectors: sqrt(2(1-s))
                # Ensures triangle inequality is satisfied for metric space axioms.
                val = np.sqrt(max(0, 2 * (1 - val)))

            dist_matrix[u, v] = val
            dist_matrix[v, u] = val  # Symmetry: d(x,y) = d(y,x) [cite: 131]

        return dist_matrix, nodes

    def reweight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates final weights f_v,w by integrating over radius intervals.
        Final weights sum to the number of points (N).
        """
        dist_matrix, nodes = self._get_distance_matrix(df)
        n = len(nodes)

        # Collect unique distances (critical radii) <= alpha
        unique_dists = np.unique(dist_matrix[dist_matrix <= self.alpha])
        unique_dists = np.sort(unique_dists)

        # Ensure integration range [0, alpha] is fully covered
        if 0 not in unique_dists:
            unique_dists = np.insert(unique_dists, 0, 0)
        if unique_dists[-1] < self.alpha:
            unique_dists = np.append(unique_dists, self.alpha)

        # Initialize running integral vector
        running_integral = np.zeros(n)

        # Iterative Integration over piecewise-constant intervals
        for i in range(len(unique_dists) - 1):
            r_curr = unique_dists[i]
            r_next = unique_dists[i + 1]

            # Graph topology G_r is constant in the interval [r_curr, r_next)
            adj = dist_matrix <= r_curr  # creates boolean adjacency matrix

            # Calculate graph weights w_r(x)
            w_g = self.weighting_func(adj)

            # Apply integration factors: (width) * (density v(r)) * (graph weight)
            # Evaluation of v at the interval start.
            density = self.v_func(r_curr, self.alpha)
            width = r_next - r_curr

            running_integral += width * density * w_g

        # Rescale the probability distribution (sum=1) to sum to N
        # This makes '1.0' the baseline weight for a distinct point.
        final_weights = running_integral * n

        return pd.DataFrame({"Question": nodes, "Weight": final_weights}).sort_values(
            by="Weight", ascending=False
        )
