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
            b_pos = self._position_stats(std_a, std_b)
            c_jac = self._jaccard(crw_a, crw_b, self.n)
            c_rank = self._rank_stats(crw_a, crw_b)
            c_pos = self._position_stats(crw_a, crw_b)

            results.append(
                {
                    "voterID": vid,
                    # Set similarity (top-k)
                    "base_jaccard": b_jac,
                    "base_swaps": self._swaps(b_jac),
                    # Rank correlation (full list)
                    "base_spearman": b_rank.get("spearman", np.nan),
                    "base_kendall": b_rank.get("kendall", np.nan),
                    # Position movement (full list)
                    "base_any_rank_change": b_pos.get("any_change", np.nan),
                    "base_n_changed": b_pos.get("n_changed", np.nan),
                    "base_avg_pos_moved": b_pos.get("avg_pos_moved", np.nan),
                    "base_max_pos_moved": b_pos.get("max_pos_moved", np.nan),
                    # CRW equivalents
                    "crw_jaccard": c_jac,
                    "crw_swaps": self._swaps(c_jac),
                    "crw_spearman": c_rank.get("spearman", np.nan),
                    "crw_kendall": c_rank.get("kendall", np.nan),
                    "crw_any_rank_change": c_pos.get("any_change", np.nan),
                    "crw_n_changed": c_pos.get("n_changed", np.nan),
                    "crw_avg_pos_moved": c_pos.get("avg_pos_moved", np.nan),
                    "crw_max_pos_moved": c_pos.get("max_pos_moved", np.nan),
                }
            )

        print("--- Analysis complete ---")
        return pd.DataFrame(results)

    def _swaps(self, jac: float | None) -> float:
        if jac is None:
            return np.nan
        # Approximate number of position swaps implied by Jaccard similarity
        return self.n - (2 * self.n * jac) / (1 + jac)

    def _position_stats(self, list_a: list, list_b: list) -> dict:
        """
        Computes per-voter position movement stats over the full ranked list.
        For each candidate present in both lists, measures |rank_a - rank_b|.
        Candidates only in one list are excluded (they have no comparable rank).
        """
        rank_a = {c: i for i, c in enumerate(list_a)}
        rank_b = {c: i for i, c in enumerate(list_b)}
        common = set(rank_a) & set(rank_b)

        if not common:
            return {}

        deltas = np.array([abs(rank_a[c] - rank_b[c]) for c in common])
        n_changed = int(np.sum(deltas > 0))

        return {
            "any_change": float(
                n_changed > 0
            ),  # 1.0 or 0.0, for easy averaging across voters
            "n_changed": n_changed,
            "avg_pos_moved": float(deltas.mean()),
            "max_pos_moved": int(deltas.max()),
        }

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

    def _load_data(self, path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        return self._extract_rankings(df)

    @staticmethod
    def _extract_rankings(df: pd.DataFrame) -> pd.DataFrame:
        """Extract ranked lists from a recommendation DataFrame (works in-memory or from parquet)."""
        if df.index.name != "voterID":
            df = df.set_index("voterID")

        std_cols = [c for c in df.columns if re.match(r"^_matchID_\d+_L2_sv$", c)]
        crw_cols = [c for c in df.columns if re.match(r"^CRW__matchID_\d+_L2_sv$", c)]

        df = df.copy()
        df["ranked_standard"] = df[std_cols].values.tolist()
        df["ranked_crw"] = df[crw_cols].values.tolist()

        # Drop NaNs that can appear at the tail of shorter canton lists
        for col in ["ranked_standard", "ranked_crw"]:
            if df[col].apply(lambda x: any(pd.isna(v) for v in x)).any():
                df[col] = df[col].apply(lambda lst: [x for x in lst if pd.notna(x)])

        return df[["ranked_standard", "ranked_crw"]]

    @classmethod
    def from_n(cls, n: int) -> "CrossRunAnalyzer":
        """Lightweight constructor for programmatic use (no file paths or metadata needed)."""
        instance = object.__new__(cls)
        instance.n = n
        instance.n_override = n
        instance.run_a_path = None
        instance.run_b_path = None
        instance.meta_a = None
        instance.meta_b = None
        return instance

    def analyze_from_dfs(
        self, rec_df_a: pd.DataFrame, rec_df_b: pd.DataFrame
    ) -> pd.DataFrame:
        """Analyze two in-memory recommendation DataFrames. Reuses all metric computation."""
        df_a = self._extract_rankings(rec_df_a)
        df_b = self._extract_rankings(rec_df_b)
        merged = df_a.join(df_b, lsuffix="_a", rsuffix="_b", how="inner")
        return self._run_analysis_loop(merged)

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

        if n_a != n_b:
            print(
                f"⚠️ Warning: n_jaccard differs between runs (A: {n_a}, B: {n_b}). "
                f"Using the smaller value for analysis."
            )
        return min(n_a, n_b)
