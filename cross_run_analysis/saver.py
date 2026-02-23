import pandas as pd
from pathlib import Path
from datetime import datetime

from cross_run_analysis.computation_cache import _stable_fields, _get_hash
from cross_run_analysis.plotter import CrossRunPlotter

# Width of the label column for aligned formatting
_LABEL_W = 42


def _row(label: str, value: str) -> str:
    return f"  {label:<{_LABEL_W}}{value}"


class CrossRunSaver:
    """
    Saves human-readable results for a cross-run comparison.
    Produces:
      - compare_<A>_vs_<B>_<timestamp>_<hash>.txt     — formatted summary report
      - compare_<A>_vs_<B>_<timestamp>_<hash>.parquet — raw per-voter results
      - compare_<A>_vs_<B>_<timestamp>_<hash>.csv     — flat summary stats for downstream analysis
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
        csv_path = self.output_dir / f"{base}_summary.csv"

        bs, br, cs, cr = self._compute_stats(df, meta_a, n_used)

        summary = self._generate_summary(
            meta_a, meta_b, n_used, len(df), bs, br, cs, cr
        )
        txt_path.write_text(summary, encoding="utf-8")

        df.to_parquet(parquet_path, index=False)

        self._save_summary_csv(csv_path, meta_a, meta_b, n_used, bs, br, cs, cr)

        print(f"\n[Results Saved]")
        print(f"  -> Report:  {txt_path.name}")
        print(f"  -> Data:    {parquet_path.name}")
        print(f"  -> Summary: {csv_path.name}")

        plotter = CrossRunPlotter(self.output_dir)
        plotter.save_plots(df, base, meta_a, meta_b)

        print(f"\n{summary}")

    # ------------------------------------------------------------------
    # Stats computation
    # ------------------------------------------------------------------

    def _compute_stats(self, df: pd.DataFrame, meta_a: dict, n: int):
        total = len(df)
        list_len = meta_a.get("n_jaccard", None)

        def set_stats(prefix):
            jac = df[f"{prefix}_jaccard"]
            swaps = df[f"{prefix}_swaps"]
            return {
                "j_mean": jac.mean(),
                "j_median": jac.median(),
                "j_min": jac.min(),
                "j_perfect": (jac == 1.0).sum() / total * 100,
                "sw_mean": swaps.mean(),
                "sw_max": swaps.max(),
            }

        def rank_stats(prefix):
            n_changed = df[f"{prefix}_n_changed"]
            pct_changed = (
                (n_changed.mean() / list_len * 100) if list_len else float("nan")
            )
            return {
                "list_len": list_len,
                "spearman": df[f"{prefix}_spearman"].mean(),
                "kendall": df[f"{prefix}_kendall"].mean(),
                "pct_any_change": df[f"{prefix}_any_rank_change"].mean() * 100,
                "avg_n_changed": n_changed.mean(),
                "pct_changed": pct_changed,
                "avg_pos_moved": df[f"{prefix}_avg_pos_moved"].mean(),
                "max_pos_moved": df[f"{prefix}_max_pos_moved"].max(),
            }

        return (
            set_stats("base"),
            rank_stats("base"),
            set_stats("crw"),
            rank_stats("crw"),
        )

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    def _generate_summary(self, meta_a, meta_b, n, total, bs, br, cs, cr) -> str:
        sep = "-" * 60
        ll = br["list_len"] or "?"  # wrong, TODO: find real list length

        def set_block(s, r, label):
            return [
                f"{label}",
                f"  Set Similarity (Top-{n}):",
                _row("Avg Jaccard Similarity:", f"{s['j_mean']:.4f}"),
                _row("Median Jaccard Similarity:", f"{s['j_median']:.4f}"),
                _row("Min Jaccard Similarity:", f"{s['j_min']:.4f}"),
                _row(
                    "Perfect matches (Jaccard=1.0):", f"{s['j_perfect']:.1f}% of voters"
                ),
                _row(
                    "Avg candidates swapped in/out:",
                    f"{s['sw_mean']:.2f} candidates per voter",
                ),
                _row(
                    "Max candidates swapped in/out:", f"{int(s['sw_max'])} candidates"
                ),
                f"  Rank Stability (All):",
                _row("Spearman Rho:", f"{r['spearman']:.4f}"),
                _row("Kendall Tau:", f"{r['kendall']:.4f}"),
                _row(
                    "Voters w/ at least 1 rank change:", f"{r['pct_any_change']:.2f}%"
                ),
                _row(
                    "Avg candidates changed per voter:",
                    f"{r['avg_n_changed']:.1f} ({r['pct_changed']:.2f}% of list)",
                ),
                _row(
                    "Avg positions a candidate moved:",
                    f"{r['avg_pos_moved']:.2f} ranks",
                ),
                _row(
                    "Max positions a candidate moved:",
                    f"{int(r['max_pos_moved'])} ranks",
                ),
            ]

        lines = [
            f"=== Cross-Run Comparison Report ===",
            f"A: {self._clean_name(meta_a)}",
            f"B: {self._clean_name(meta_b)}",
            f"Voters compared: {total}",
            sep,
            *set_block(bs, br, "--- STANDARD ---"),
            sep,
            *set_block(cs, cr, "--- CRW ---"),
            sep,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CSV summary (one row per run-comparison, all stats as columns)
    # ------------------------------------------------------------------

    def _save_summary_csv(self, path: Path, meta_a, meta_b, n, bs, br, cs, cr):
        row = {
            "run_a": self._clean_name(meta_a),
            "run_b": self._clean_name(meta_b),
            "n_jaccard": n,
            "data_year": meta_a.get("data_year"),
            "dist": meta_a.get("dist"),
            # Standard set similarity
            "base_jaccard_mean": bs["j_mean"],
            "base_jaccard_median": bs["j_median"],
            "base_jaccard_min": bs["j_min"],
            "base_jaccard_perfect_pct": bs["j_perfect"],
            "base_swaps_mean": bs["sw_mean"],
            "base_swaps_max": bs["sw_max"],
            # Standard rank stability
            "base_spearman": br["spearman"],
            "base_kendall": br["kendall"],
            "base_pct_any_change": br["pct_any_change"],
            "base_avg_n_changed": br["avg_n_changed"],
            "base_pct_changed": br["pct_changed"],
            "base_avg_pos_moved": br["avg_pos_moved"],
            "base_max_pos_moved": br["max_pos_moved"],
            # CRW set similarity
            "crw_jaccard_mean": cs["j_mean"],
            "crw_jaccard_median": cs["j_median"],
            "crw_jaccard_min": cs["j_min"],
            "crw_jaccard_perfect_pct": cs["j_perfect"],
            "crw_swaps_mean": cs["sw_mean"],
            "crw_swaps_max": cs["sw_max"],
            # CRW rank stability
            "crw_spearman": cr["spearman"],
            "crw_kendall": cr["kendall"],
            "crw_pct_any_change": cr["pct_any_change"],
            "crw_avg_n_changed": cr["avg_n_changed"],
            "crw_pct_changed": cr["pct_changed"],
            "crw_avg_pos_moved": cr["avg_pos_moved"],
            "crw_max_pos_moved": cr["max_pos_moved"],
        }
        pd.DataFrame([row]).to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_prefix(self, meta_a: dict, meta_b: dict) -> str:
        return f"compare_{self._clean_name(meta_a)}_vs_{self._clean_name(meta_b)}"

    def _clean_name(self, meta: dict) -> str:
        base = meta.get("config_name", "unknown")
        overrides = meta.get("overrides", [])
        if overrides:
            suffix = "_".join(overrides).replace("~", "").replace("=", "")
            return f"{base}_{suffix}"
        return base
