import json
import hashlib
import pandas as pd
from pathlib import Path

# Fields used to uniquely identify a computation.
# Must be stable across runs (no timestamps, no file paths).
_HASH_FIELDS = [
    "dist",
    "data_year",
    "overrides",
    "alpha",
    "crw_paper_choice",
    "rec_dist_method",
    "n_recommendations",
    "filter_districts",
    "use_OG_weights",
    "district",
    "E5_instruction",
    "subset_n",
]


def _stable_fields(meta: dict) -> dict:
    return {k: meta[k] for k in _HASH_FIELDS if k in meta}


def _get_hash(meta_a: dict, meta_b: dict, n: int) -> str:
    """
    Deterministic hash over stable metadata fields + n.
    Order of A/B is preserved (A vs B != B vs A).
    """
    payload = {
        "run_a": _stable_fields(meta_a),
        "run_b": _stable_fields(meta_b),
        "n_jaccard": n,
    }
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


class ComputationCache:
    """
    Manages storage of raw per-voter computation results.
    NOT for human reading — filenames are hash-only, no timestamps.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, meta_a: dict, meta_b: dict, n: int) -> Path:
        h = _get_hash(meta_a, meta_b, n)
        return self.cache_dir / f"calc_{h}.parquet"

    def load(self, meta_a: dict, meta_b: dict, n: int) -> pd.DataFrame | None:
        path = self._get_path(meta_a, meta_b, n)
        if path.exists():
            print(f"[CACHE HIT] Loading pre-computed results: {path.name}")
            return pd.read_parquet(path)
        return None

    def save(self, df: pd.DataFrame, meta_a: dict, meta_b: dict, n: int) -> None:
        path = self._get_path(meta_a, meta_b, n)
        df.to_parquet(path, index=False)
        print(f"[CACHE SAVE] Computation cached: {path.name}")
