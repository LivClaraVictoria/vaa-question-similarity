import pandas as pd
import numpy as np
import json
import re
import warnings
from pathlib import Path
from scipy.stats import spearmanr, kendalltau


class CrossRunAnalyzer:
    def __init__(self, run_a_path: str, run_b_path: str, n_override: int | None = None):
        self.run_a_path = Path(run_a_path)
        self.run_b_path = Path(run_b_path)
        self.n_override = n_override

        # 1. Load Metadata
        self.meta_a = self._load_metadata(self.run_a_path)
        self.meta_b = self._load_metadata(self.run_b_path)

        # 2. Resolve N
        self.n = self._resolve_n()

    def analyze(self) -> pd.DataFrame:
        """Main execution method."""
        print(f"Loading data (this may take a moment)...")
        self.df_a = self._load_data(self.run_a_path)
        self.df_b = self._load_data(self.run_b_path)

        return self._run_analysis_loop()

    def _load_metadata(self, parquet_path: Path) -> dict:
        json_path = parquet_path.with_suffix(".json")
        if not json_path.exists():
            return {"n_jaccard": 30}
        with open(json_path, "r") as f:
            return json.load(f)

    def _resolve_n(self) -> int:
        if self.n_override is not None:
            print(f"⚠️  CLI Override active: Using n={self.n_override}")
            return self.n_override
        n_a = self.meta_a.get("n_jaccard", 30)
        n_b = self.meta_b.get("n_jaccard", 30)
        if n_a != n_b:
            print(
                f"⚠️  Mismatch in 'n' (A={n_a}, B={n_b}). Using smaller n={min(n_a, n_b)}."
            )
            return min(n_a, n_b)
        return n_a

    def _load_data(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Could not find file at {path}")
        df = pd.read_parquet(path)

        std_cols = [c for c in df.columns if re.match(r"^_matchID_\d+_L2_sv$", c)]
        crw_cols = [c for c in df.columns if re.match(r"^CRW__matchID_\d+_L2_sv$", c)]

        if std_cols:
            df["ranked_standard"] = df[std_cols].values.tolist()
            df["ranked_standard"] = df["ranked_standard"].apply(
                lambda lst: [x for x in lst if pd.notna(x)]
            )
        if crw_cols:
            df["ranked_crw"] = df[crw_cols].values.tolist()
            df["ranked_crw"] = df["ranked_crw"].apply(
                lambda lst: [x for x in lst if pd.notna(x)]
            )
        return df

    def _run_analysis_loop(self):
        # 1. Strict Validation
        if (
            self.df_a["voterID"].duplicated().any()
            or self.df_b["voterID"].duplicated().any()
        ):
            raise ValueError(
                "❌ FATAL: Duplicate voterIDs detected in one of the runs."
            )

        voters_a = set(self.df_a["voterID"])
        voters_b = set(self.df_b["voterID"])
        if voters_a != voters_b:
            raise ValueError(
                f"❌ FATAL: Voter sets do not match! ({len(voters_a)} vs {len(voters_b)})"
            )

        # 2. Merge
        merged = pd.merge(
            self.df_a,
            self.df_b,
            on="voterID",
            suffixes=("_data", "_cloned"),
            how="inner",
        )

        # 3. Circuit Breakers
        can_do_base = self._can_compute_correlations(
            merged, "ranked_standard_data", "ranked_standard_cloned"
        )
        can_do_crw = self._can_compute_correlations(
            merged, "ranked_crw_data", "ranked_crw_cloned"
        )

        if not can_do_base:
            print("⚠️ Standard ranks mismatch. Skipping standard stats.")
        if not can_do_crw:
            print("⚠️ CRW ranks mismatch. Skipping CRW stats.")

        # 4. The Loop
        results = []
        n_val = self.n

        iter_data = zip(
            merged["voterID"],
            merged["ranked_standard_data"],
            merged["ranked_standard_cloned"],
            merged["ranked_crw_data"],
            merged["ranked_crw_cloned"],
        )

        for vid, b_dat, b_clo, c_dat, c_clo in iter_data:
            # Base Stats
            b_jac = self._calculate_jaccard(b_dat, b_clo, n_val)
            b_stats = self._calculate_rank_stats(b_dat, b_clo) if can_do_base else {}

            # CRW Stats
            c_jac = self._calculate_jaccard(c_dat, c_clo, n_val)
            c_stats = self._calculate_rank_stats(c_dat, c_clo) if can_do_crw else {}

            # Combine into one dict
            row = {
                "voterID": vid,
                "base_jaccard": b_jac,
                "base_spearman": b_stats.get("spearman", np.nan),
                "base_kendall": b_stats.get("kendall", np.nan),
                "base_changed_count": b_stats.get("changed_count", 0),
                "base_total_shift": b_stats.get("total_shift", 0),
                "base_max_shift": b_stats.get("max_shift", 0),
                "base_list_len": b_stats.get("list_len", 0),
                "crw_jaccard": c_jac,
                "crw_spearman": c_stats.get("spearman", np.nan),
                "crw_kendall": c_stats.get("kendall", np.nan),
                "crw_changed_count": c_stats.get("changed_count", 0),
                "crw_total_shift": c_stats.get("total_shift", 0),
                "crw_max_shift": c_stats.get("max_shift", 0),
                "crw_list_len": c_stats.get("list_len", 0),
            }
            results.append(row)

        return pd.DataFrame(results)

    def _calculate_rank_stats(self, list_a, list_b):
        """Computes both Correlations AND Shift Metrics efficiently."""
        n = len(list_a)
        if n < 1:
            return {}

        # O(1) Dictionary Mapping
        rank_dict_b = {c: i for i, c in enumerate(list_b)}
        try:
            ranks_b = np.array([rank_dict_b[c] for c in list_a])
        except KeyError:
            return {}

        ranks_a = np.arange(n)

        # 1. Shift Metrics (Pure Math)
        diffs = np.abs(ranks_a - ranks_b)

        # 2. Correlations (SciPy)
        if n >= 2:
            s, _ = spearmanr(ranks_a, ranks_b)
            k, _ = kendalltau(ranks_a, ranks_b)
        else:
            s, k = np.nan, np.nan

        return {
            "spearman": s,
            "kendall": k,
            "changed_count": np.sum(ranks_a != ranks_b),
            "total_shift": np.sum(diffs),
            "max_shift": np.max(diffs) if n > 0 else 0,
            "list_len": n,
        }

    def _calculate_jaccard(self, list_a, list_b, n):
        s1, s2 = set(list_a[:n]), set(list_b[:n])
        if not s1 and not s2:
            return 1.0
        return len(s1 & s2) / len(s1 | s2)

    def _can_compute_correlations(self, df, col_a, col_b):
        for a, b in zip(df[col_a], df[col_b]):
            if len(a) != len(b) or set(a) != set(b):
                return False
        return True
