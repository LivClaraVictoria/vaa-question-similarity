import pandas as pd
import numpy as np
import warnings
from typing import Any, Sequence


class CloneRobustReweighter:
    """
    Implements General Clone-Robust Weighting.
    Balances importance by integrating graph-based weights over distance thresholds
    while maintaining Question IDs for downstream mapping.
    """

    def __init__(self, config: Any):
        self.alpha = getattr(config, "alpha", 1.0)
        self.v_func = getattr(config, "v_func", lambda r, alpha: 1.0 / alpha)
        self.weighting_func = getattr(
            config, "weighting_func", self._class_uniform_weighting_fn
        )
        self.important_params_list = list(config.CRW_HASH_PARAMS)

        print(
            f"Initialized CloneRobustReweighter with alpha={self.alpha} and important parameters: {self.important_params_list}"
        )
        # Add more if needed for hashing

    def _class_uniform_weighting_fn(self, adj: np.ndarray) -> np.ndarray:
        adj_rows = [tuple(row) for row in adj]
        class_counts = {}
        for row in adj_rows:
            class_counts[row] = class_counts.get(row, 0) + 1

        num_classes = len(class_counts)
        weights = np.array(
            [1.0 / (num_classes * class_counts[row]) for row in adj_rows]
        )
        return weights

    def _get_distance_matrix(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, list[int], dict[int, str]]:
        """
        Returns:
            dist_matrix: The NxN matrix
            ids: Sorted list of IDs
            id_to_text: Dictionary mapping ID -> Question Text for visualization
        """

        if "Distance" in df.columns:
            val_col = "Distance"
            is_similarity = False
        elif "Similarity" in df.columns:
            val_col = "Similarity"
            is_similarity = True
        else:
            print(
                "⚠️ Error: DataFrame does not contain 'Distance' or 'Similarity' column."
            )

        # 1. Create a mapping of ID -> Text using both columns
        # This ensures we don't lose the text for any question (just in case some IDs only appear in one column)
        id_to_text = (
            pd.concat(
                [
                    df[["ID1", "Qu1"]].rename(columns={"ID1": "ID", "Qu1": "Text"}),
                    df[["ID2", "Qu2"]].rename(columns={"ID2": "ID", "Qu2": "Text"}),
                ]
            )
            .drop_duplicates("ID")
            .set_index("ID")["Text"]
            .to_dict()
        )

        # 2. Sort IDs and create index mapping
        ids = sorted(list(id_to_text.keys()))
        id_to_idx = {q_id: i for i, q_id in enumerate(ids)}

        n = len(ids)
        dist_matrix = np.zeros(
            (n, n)
        )  # ensures diagonals = 0, since 2 equal questions not contained in dataframe

        for _, row in df.iterrows():  # row: contains all data of that specific row
            u, v = id_to_idx[row["ID1"]], id_to_idx[row["ID2"]]
            val = row[val_col]

            if is_similarity:
                val = np.sqrt(max(0, 2 * (1 - val)))

            dist_matrix[u, v] = val
            dist_matrix[v, u] = val

        return dist_matrix, ids, id_to_text

    def reweight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the matrix, the sorted IDs, and the text map.
        """
        dist_matrix, ids, id_to_text = self._get_distance_matrix(df)
        n = len(ids)

        unique_dists = np.unique(dist_matrix[dist_matrix <= self.alpha])
        unique_dists = np.sort(unique_dists)

        if 0 not in unique_dists:
            unique_dists = np.insert(unique_dists, 0, 0)
        if unique_dists[-1] < self.alpha:
            unique_dists = np.append(unique_dists, self.alpha)

        running_integral = np.zeros(n)

        for i in range(len(unique_dists) - 1):
            r_curr = unique_dists[i]
            r_next = unique_dists[i + 1]
            adj = dist_matrix <= r_curr
            w_g = self.weighting_func(adj)
            density = self.v_func(r_curr, self.alpha)
            width = r_next - r_curr
            running_integral += width * density * w_g

        final_weights = running_integral * n

        results_df = pd.DataFrame({"ID_question": ids, "Weight": final_weights})

        # Add the text back in by mapping the IDs --> pd contains columns ID_question, Question, Weight
        results_df["Question"] = results_df["ID_question"].map(id_to_text)

        return results_df
