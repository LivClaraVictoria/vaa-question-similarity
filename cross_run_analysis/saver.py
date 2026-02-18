import pandas as pd
from pathlib import Path
from datetime import datetime

from cross_run_analysis.computation_cache import _stable_fields, _get_hash


class CrossRunSaver:
    """
    Saves human-readable results for a cross-run comparison.
    Produces:
      - compare_<A>_vs_<B>_<timestamp>_<hash>.txt     — summary report
      - compare_<A>_vs_<B>_<timestamp>_<hash>.parquet — raw per-voter results
    Deduplication: if a .txt with the same hash already exists, saving is skipped.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(
        self,
        df: pd.DataFrame,
        meta_a: dict,
        meta_b: dict,
        n_used: int,
    ) -> None:
        h = _get_hash(meta_a, meta_b, n_used)
        prefix = self._get_prefix(meta_a, meta_b)

        # Deduplication check
        existing = list(self.output_dir.glob(f"*{h}*.txt"))
        if existing:
            print(
                f"[SKIP SAVE] Results with hash {h} already exist: {existing[0].name}"
            )
            return

        timestamp = datetime.now().strftime("%m%d_%H%M")
        base = f"{prefix}_{timestamp}_{h}"

        txt_path = self.output_dir / f"{base}.txt"
        parquet_path = self.output_dir / f"{base}.parquet"

        summary = self._generate_summary(df, meta_a, meta_b, n_used)
        txt_path.write_text(summary, encoding="utf-8")
        df.to_parquet(parquet_path, index=False)

        print(f"\n[Results Saved]")
        print(f"  -> Report: {txt_path.name}")
        print(f"  -> Data:   {parquet_path.name}")
        print(f"\n{summary}")

    def _get_prefix(self, meta_a: dict, meta_b: dict) -> str:
        return f"compare_{self._clean_name(meta_a)}_vs_{self._clean_name(meta_b)}"

    def _clean_name(self, meta: dict) -> str:
        base = meta.get("config_name", "unknown")
        overrides = meta.get("overrides", [])
        if overrides:
            suffix = "_".join(overrides).replace("~", "").replace("=", "")
            return f"{base}_{suffix}"
        return base

    def _generate_summary(
        self, df: pd.DataFrame, meta_a: dict, meta_b: dict, n: int
    ) -> str:
        def stats(prefix):
            return {
                "j_mean": df[f"{prefix}_jaccard"].mean(),
                "j_min": df[f"{prefix}_jaccard"].min(),
                "s_mean": df[f"{prefix}_spearman"].mean(),
                "s_std": df[f"{prefix}_spearman"].std(),
                "k_mean": df[f"{prefix}_kendall"].mean(),
                "sw_mean": df[f"{prefix}_swaps"].mean(),
                "sw_max": df[f"{prefix}_swaps"].max(),
            }

        b = stats("base")
        c = stats("crw")

        lines = [
            "=== Cross-Run Comparison Report ===",
            f"A: {self._clean_name(meta_a)}",
            f"B: {self._clean_name(meta_b)}",
            f"N (Jaccard top-k): {n}",
            f"Voters compared: {len(df)}",
            "",
            "--- STANDARD ---",
            f"  Jaccard:  mean={b['j_mean']:.4f}  min={b['j_min']:.4f}",
            f"  Spearman: mean={b['s_mean']:.4f}  std={b['s_std']:.4f}",
            f"  Kendall:  mean={b['k_mean']:.4f}",
            f"  Swaps:    mean={b['sw_mean']:.2f}  max={int(b['sw_max'])}",
            "",
            "--- CRW ---",
            f"  Jaccard:  mean={c['j_mean']:.4f}  min={c['j_min']:.4f}",
            f"  Spearman: mean={c['s_mean']:.4f}  std={c['s_std']:.4f}",
            f"  Kendall:  mean={c['k_mean']:.4f}",
            f"  Swaps:    mean={c['sw_mean']:.2f}  max={int(c['sw_max'])}",
        ]
        return "\n".join(lines)
