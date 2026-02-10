import hashlib
import json
import pandas as pd
from pathlib import Path


class CacheManager:
    def __init__(
        self, config, cache_dir: Path, prefix: str, params_list: list[str]
    ):  # prefix to make it human-readable, params to generate a unique hash
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # build params dictionary from config
        params = self.build_params(params_list)

        # generate the filename once during initialization
        self.filename = f"{prefix}_{self._generate_hash(params)}.parquet"
        self.path = self.cache_dir / self.filename

    def _generate_hash(self, params: dict) -> str:
        # convert to string for hashing, sort keys for consistency
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()

    def save(self, df: pd.DataFrame):
        df.to_parquet(self.path)
        print(f"--- Result Cached: {self.filename} ---")

    def load_if_exists(self) -> pd.DataFrame | None:
        if self.path.exists():
            print(f"--- Cache Hit: {self.filename} ---")
            return pd.read_parquet(self.path)
        return None

    def build_params(self, keys: list) -> dict:
        """
        Helper to extract specific keys from the config object.
        """
        return {k: getattr(self.config, k, None) for k in keys}
