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
        if ext.startswith("."):  # skips leading dot if provided
            ext = ext[1:]

        if readable:
            # For experiment_results: readable names + hash
            timestamp = datetime.now().strftime("%m%d_%H%M")
            name = f"{self.prefix}_{timestamp}_{self.hash}.{ext}"
        else:
            # For functional cache: strict hash-based name
            name = f"{self.prefix}_{self.hash}.{ext}"

        return self.output_dir / name

    def exists(self, extension=None) -> Path | None:
        """Checks if a file with this hash and specific extension exists."""
        ext = extension or getattr(self.config, "results_file_type", "*")
        if ext.startswith("."):
            ext = ext[1:]

        matches = list(self.output_dir.glob(f"*{self.hash}.{ext}"))
        return matches[0] if matches else None

    def load(self, extension=None):
        path = self.exists(extension=extension)
        if not path:
            return None

        print(f"--- Cache Hit: Loading results from {path.name} ---")
        ext = path.suffix.lower()

        if ext == ".parquet":
            return pd.read_parquet(path)
        elif ext == ".csv":
            return pd.read_csv(path)
        elif ext in [".txt", ".md"]:
            return path.read_text(encoding="utf-8")
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return path  # For unsupported types, just return the path (ex matplotlib)

    def save(self, data=None, readable=False, extension=None) -> Path | None:
        if getattr(self.config, "save_results", True) is False:
            print("---No results saved.---")
            return None

        path = self.get_path(readable=readable, extension=extension)
        ext = path.suffix.lower()

        if data is not None:
            if isinstance(data, pd.DataFrame):
                if ext == ".parquet":
                    data.to_parquet(path, index=False)
                else:
                    data.to_csv(path, index=False)
            elif ext in [".txt", ".md"] and isinstance(data, str):
                path.write_text(data, encoding="utf-8")
            elif ext == ".json":
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, default=str)
            else:
                print(
                    f"Warning: ResultManager auto-save not configured for type {type(data)} to {ext}."
                )

        print(f"\nSuccess! File saved to:")
        print(f"  -> {path}")
        return path
