import pandas as pd
import numpy as np
import re
import os
import json
import psutil
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

from cross_run_analysis.computation_cache import ComputationCache


class CrossRunAnalyzer:
    def __init__(self, run_a_path: str, run_b_path: str, n_override: int | None = None):
        self.run_a_path = Path(run_a_path)
        self.run_b_path = Path(run_b_path)
        self.n_override = n_override

        self.meta_a = self._load_metadata(self.run_a_path)
        self.meta_b = self._load_metadata(self.run_b_path)
        self.n = self._resolve_n()

    def analyze(self, cache: ComputationCache | None = None) -> pd.DataFrame:
        """Main entry point. Returns per-voter results DataFrame."""
        if cache is not None:
            cached = cache.load(self.meta_a, self.meta_b, self.n)
            if cached is not None:
                return cached

        print(f"\nLoading data...")
        df_a = self._load_data(self.run_a_path)
        df_b = self._load_data(self.run_b_path)

        # Inner join on voterID — only compare voters present in both runs
        merged = df_a.join(df_b, lsuffix="_a", rsuffix="_b", how="inner")
        print(f"Voters in both runs: {len(merged)}")

        results = self._run_analysis_loop(merged)

        if cache is not None:
            cache.save(results, self.meta_a, self.meta_b, self.n)

        return results

    def _run_analysis_loop(self, df: pd.DataFrame) -> pd.DataFrame:
        total = len(df)
        print(f"--- Analysis started: {total} voters ---")

        results = []
        process = psutil.Process(os.getpid())

        # Zip optimization: fastest way to iterate rows for custom Python logic
        zipped = zip(
            df.index,
            df["ranked_standard_a"],
            df["ranked_standard_b"],
            df["ranked_crw_a"],
            df["ranked_crw_b"],
        )

        for i, (vid, std_a, std_b, crw_a, crw_b) in enumerate(zipped):
            if i % 2000 == 0 and i > 0:
                mem = process.memory_info().rss / (1024**2)
                print(f"  > {i}/{total} ({i/total*100:.1f}%) | RAM: {mem:.1f} MB")

            b_jac = self._jaccard(std_a, std_b, self.n)
            b_rank = self._rank_stats(std_a, std_b)
            c_jac = self._jaccard(crw_a, crw_b, self.n)
            c_rank = self._rank_stats(crw_a, crw_b)

            results.append(
                {
                    "voterID": vid,
                    "base_jaccard": b_jac,
                    "base_spearman": b_rank.get("spearman", np.nan),
                    "base_kendall": b_rank.get("kendall", np.nan),
                    "base_swaps": self._swaps(b_jac),
                    "crw_jaccard": c_jac,
                    "crw_spearman": c_rank.get("spearman", np.nan),
                    "crw_kendall": c_rank.get("kendall", np.nan),
                    "crw_swaps": self._swaps(c_jac),
                }
            )

        print("--- Analysis complete ---")
        return pd.DataFrame(results)

    def _swaps(self, jac: float | None) -> float:
        if jac is None:
            return np.nan
        # Approximate number of position swaps implied by Jaccard similarity
        return self.n - (2 * self.n * jac) / (1 + jac)

    def _load_data(self, path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if df.index.name != "voterID":
            df.set_index("voterID", inplace=True)

        std_cols = [c for c in df.columns if re.match(r"^_matchID_\d+_L2_sv$", c)]
        crw_cols = [c for c in df.columns if re.match(r"^CRW__matchID_\d+_L2_sv$", c)]

        df["ranked_standard"] = df[std_cols].values.tolist()
        df["ranked_crw"] = df[crw_cols].values.tolist()

        # Drop NaNs that can appear at the tail of shorter canton lists
        for col in ["ranked_standard", "ranked_crw"]:
            if df[col].apply(lambda x: any(pd.isna(v) for v in x)).any():
                df[col] = df[col].apply(lambda lst: [x for x in lst if pd.notna(x)])

        return df[["ranked_standard", "ranked_crw"]]

    def _rank_stats(self, list_a: list, list_b: list) -> dict:
        common = set(list_a) & set(list_b)
        if len(common) < 2:
            return {}

        # rank_dict_b maps candidate ID -> position in list_b
        # ranks_a is just 0,1,2,... (list_a is the reference ordering)
        # This correctly measures how much list_b disagrees with list_a's order
        rank_dict_b = {c: i for i, c in enumerate(list_b)}
        ranks_b = np.array([rank_dict_b[c] for c in list_a if c in rank_dict_b])
        ranks_a = np.arange(len(ranks_b))

        s, _ = spearmanr(ranks_a, ranks_b)
        k, _ = kendalltau(ranks_a, ranks_b)
        return {"spearman": s, "kendall": k}

    def _jaccard(self, list_a: list, list_b: list, n: int) -> float:
        s1, s2 = set(list_a[:n]), set(list_b[:n])
        if not s1 and not s2:
            return 1.0
        return len(s1 & s2) / len(s1 | s2)

    def _load_metadata(self, parquet_path: Path) -> dict:
        json_path = parquet_path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(
                f"Metadata JSON not found for {parquet_path.name}. "
                f"Expected: {json_path}"
            )
        with open(json_path, "r") as f:
            return json.load(f)

    def _resolve_n(self) -> int:
        if self.n_override is not None:
            return self.n_override
        n_a = self.meta_a.get("n_jaccard")
        n_b = self.meta_b.get("n_jaccard")
        if n_a is None or n_b is None:
            raise ValueError(
                "Could not resolve n_jaccard from metadata. "
                "Use --n_value to override explicitly."
            )
        return min(n_a, n_b)
