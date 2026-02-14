import json
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime


class ResultManager:
    def __init__(self, config, dir: Path, params_list: list[str], prefix: str = ""):
        self.config = config
        self.output_dir = Path(dir)
        self.params_list = params_list
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate hash once at initialization
        self.hash = self._generate_hash()

    def _generate_hash(self) -> str:
        params = {k: getattr(self.config, k, None) for k in self.params_list}
        param_str = json.dumps(
            params, sort_keys=True, default=str
        )  # sorted for hash consistency
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    def get_path(self, readable=False, extension=None) -> Path:
        ext = extension or self.config.results_file_type

        if readable:
            # For experiment_results: readable names + hash
            timestamp = datetime.now().strftime("%m%d_%H%M")
            name = f"{self.prefix}_{timestamp}_{self.hash}.{ext}"
        else:
            # For functional cache: strict hash-based name
            name = f"{self.prefix}_{self.hash}.{ext}"

        return self.output_dir / name

    def exists(self) -> Path | None:
        """Checks if a file with this hash already exists in the directory."""
        matches = list(self.output_dir.glob(f"*{self.hash}.*"))
        return matches[0] if matches else None

    def load(self):
        path = self.exists()
        if not path:
            return None
        print(f"--- Cache Hit: Loading results from {path.name} ---")
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    def save(self, df: pd.DataFrame, readable=False):
        if getattr(self.config, "save_results", True) is False:
            print("---No results saved.---")
            return None

        path = self.get_path(readable=readable)
        if self.config.results_file_type == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

        print(f"\nSuccess! Results saved to:")
        print(f"  -> {path}")
        return path
