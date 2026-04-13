"""
Forbidden interval coverage analysis for CRW mini/maxi question pairs.

For each (x_mini, y_maxi) pair (30 x 45 = 1,350 total), computes the fraction
of [0, alpha] covered by the union of "forbidden intervals". The forbidden
interval for a third question z is [min(d_xz, d_yz), max(d_xz, d_yz)): the set
of radii r at which z is a neighbor of exactly one of x or y, meaning CRW
cannot treat the pair as equivalent at those radii.

High forbidden-interval coverage explains why CRW fails to correct natural topic
overrepresentation in the mini vs full questionnaire setting: the pair's
neighborhoods differ too much across [0, alpha] for CRW to down-weight either.

Metrics: E5-INSTRUCT and ANSWER-CORRELATION-ARCCOS (ZH 2023, full 75 questions).

Usage:
    python -m experiments.exploratory.analysis.forbidden_interval_coverage
    python -m experiments.exploratory.analysis.forbidden_interval_coverage \\
        --metrics E5-INSTRUCT \\
        --top-k 5 \\
        --no-viz
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments._common import _get_question_text_col
from vqs.config_utils import load_config
from vqs.data_loader import load_dataset
from vqs.similarity_metrics import get_calculator

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "dependencies" / "rsfp"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_CONFIGS = {
    "E5-INSTRUCT": {
        "config_path": "configs/full_pipeline/base_data/pipeline_e5_instruct_ZH.py",
        "key_alphas": [0.3, 0.4],
    },
    "ANSWER-CORRELATION-ARCCOS": {
        "config_path": "configs/full_pipeline/base_data/pipeline_answer_corr_arccos_ZH.py",
        "key_alphas": [1.1, 1.5],
    },
}

ALPHA_GRID: list[float] = sorted(
    set([round(i * 0.1, 1) for i in range(1, 31)] + [0.3, 0.4, 1.1, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0])
)

BASE_RESULTS_DIR = Path("experiment_results/distance_analysis/forbidden_intervals")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forbidden interval coverage for mini/maxi question pairs"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="E5-INSTRUCT,ANSWER-CORRELATION-ARCCOS",
        help="Comma-separated metric keys to run (default: both)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of closest pairs to show in bar chart (default: 3)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip all matplotlib output",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core math utilities
# ---------------------------------------------------------------------------


def _alpha_col(a: float) -> str:
    return f"coverage_{a:.1f}"


def _measure_col(a: float) -> str:
    return f"covered_measure_{a:.1f}"


def _eff_col(a: float) -> str:
    return f"eff_usable_frac_{a:.1f}"


def _eff_measure_col(a: float) -> str:
    return f"eff_usable_measure_{a:.1f}"


def _find_precomputed_distances(metric_name: str, data_year: int = 2023) -> Path | None:
    """Return the most recent *_all_distances.csv for this metric, or None."""
    dist_dir = Path(f"experiment_results/distance_analysis/{metric_name}_{data_year}")
    if not dist_dir.exists():
        return None
    csvs = sorted(dist_dir.glob("*_all_distances.csv"))
    return csvs[-1] if csvs else None


def build_distance_matrix(
    dist_df: pd.DataFrame,
    id_to_text: dict[int, str] | None = None,
) -> tuple[np.ndarray, list[int], dict[int, str]]:
    """Convert long-format distance DataFrame to a symmetric NxN numpy array.

    Handles both 'Distance' and 'Similarity' columns. NaN values are filled
    with pi/2 (maximum arccos distance) as a defensive measure.

    id_to_text can be supplied externally (e.g. from the questions DataFrame)
    when the dist_df does not have Qu1/Qu2 columns.

    Returns:
        dist_matrix: NxN symmetric float64 array (diagonal = 0)
        ids: sorted list of question IDs (length N)
        id_to_text: dict mapping question ID -> question text
    """
    if "Distance" in dist_df.columns:
        val_col = "Distance"
        is_similarity = False
    elif "Similarity" in dist_df.columns:
        val_col = "Similarity"
        is_similarity = True
    else:
        raise ValueError("DataFrame must contain a 'Distance' or 'Similarity' column")

    if id_to_text is None:
        if "Qu1" in dist_df.columns and "Qu2" in dist_df.columns:
            id_to_text = (
                pd.concat(
                    [
                        dist_df[["ID1", "Qu1"]].rename(columns={"ID1": "ID", "Qu1": "Text"}),
                        dist_df[["ID2", "Qu2"]].rename(columns={"ID2": "ID", "Qu2": "Text"}),
                    ]
                )
                .drop_duplicates("ID")
                .set_index("ID")["Text"]
                .to_dict()
            )
        else:
            all_ids = set(dist_df["ID1"].tolist()) | set(dist_df["ID2"].tolist())
            id_to_text = {q_id: str(q_id) for q_id in all_ids}

    ids = sorted(id_to_text.keys())
    id_to_idx = {q_id: i for i, q_id in enumerate(ids)}
    n = len(ids)
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for _, row in dist_df.iterrows():
        u = id_to_idx[row["ID1"]]
        v = id_to_idx[row["ID2"]]
        val = float(row[val_col])
        if is_similarity:
            val = float(np.sqrt(max(0.0, 2.0 * (1.0 - val))))
        dist_matrix[u, v] = val
        dist_matrix[v, u] = val

    # Defensive: fill any remaining NaN with pi/2
    nan_mask = np.isnan(dist_matrix)
    if nan_mask.any():
        dist_matrix[nan_mask] = np.pi / 2

    return dist_matrix, ids, id_to_text


def get_mini_maxi_ids(
    dataset: dict,
) -> tuple[list[int], list[int]]:
    """Return (mini_ids_sorted, maxi_ids_sorted) from dataset questions.

    mini: rapide == 1 AND ID_question < 9_000_000
    maxi (full-only): rapide == 0 AND ID_question < 9_000_000
    """
    df_q = dataset["questions"]
    real_mask = df_q["ID_question"] < 9_000_000
    mini_ids = sorted(df_q.loc[real_mask & (df_q["rapide"] == 1), "ID_question"].tolist())
    maxi_ids = sorted(df_q.loc[real_mask & (df_q["rapide"] == 0), "ID_question"].tolist())
    return mini_ids, maxi_ids


def compute_forbidden_intervals(
    x_idx: int,
    y_idx: int,
    dist_matrix: np.ndarray,
) -> list[tuple[float, float]]:
    """Compute raw (unmerged) forbidden intervals for the pair (x, y).

    For each z != x, y, the forbidden interval is [min(d_xz, d_yz), max(d_xz, d_yz)).
    z's where d_xz == d_yz contribute no interval and are skipped.

    Returns list of (lo, hi) tuples.
    """
    n = dist_matrix.shape[0]
    d_xz = dist_matrix[x_idx, :]  # shape (N,)
    d_yz = dist_matrix[y_idx, :]

    z_mask = np.ones(n, dtype=bool)
    z_mask[x_idx] = False
    z_mask[y_idx] = False

    valid = z_mask & (d_xz != d_yz)
    lo = np.minimum(d_xz, d_yz)[valid]
    hi = np.maximum(d_xz, d_yz)[valid]

    return list(zip(lo.tolist(), hi.tolist()))


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sort and merge overlapping/touching intervals into a minimal set."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        cur_start, cur_end = merged[-1]
        if start <= cur_end:
            merged[-1] = (cur_start, max(cur_end, end))
        else:
            merged.append((start, end))
    return merged


def covered_measure_at_alpha(
    merged_intervals: list[tuple[float, float]],
    alpha: float,
) -> float:
    """Return |union(merged_intervals) ∩ [0, alpha]| (absolute measure)."""
    if alpha <= 0.0:
        return 0.0
    total = 0.0
    for start, end in merged_intervals:
        if start >= alpha:
            break
        total += min(end, alpha) - start
    return total


def coverage_fraction_at_alpha(
    merged_intervals: list[tuple[float, float]],
    alpha: float,
) -> float:
    """Return fraction of [0, alpha] covered by merged_intervals."""
    if alpha <= 0.0:
        return 0.0
    return covered_measure_at_alpha(merged_intervals, alpha) / alpha


def effective_usable_measure(
    merged_intervals: list[tuple[float, float]],
    d_xy: float,
    alpha: float,
) -> float:
    """Measure of [d_xy, alpha] NOT covered by merged_intervals.

    This is the CRW-relevant usable range: radii where the pair is adjacent
    (r >= d_xy) AND their neighborhoods agree (r not forbidden).

    Returns 0.0 if d_xy >= alpha (adjacent window empty — CRW can never
    see this pair as similar at this alpha).
    """
    if d_xy >= alpha:
        return 0.0
    adjacent_length = alpha - d_xy
    forbidden_in_window = 0.0
    for start, end in merged_intervals:
        s = max(start, d_xy)
        e = min(end, alpha)
        if s < e:
            forbidden_in_window += e - s
    return adjacent_length - forbidden_in_window


# ---------------------------------------------------------------------------
# Per-metric analysis
# ---------------------------------------------------------------------------


def run_metric_analysis(
    metric_name: str,
    config_path: str,
    alpha_grid: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run forbidden interval analysis for one metric.

    Returns:
        intervals_df: one row per (pair, merged interval), sorted by d_xy
        coverage_df: one row per pair with coverage at every alpha, sorted by d_xy
        max_pair_dist: maximum pairwise distance in the full dataset — beyond this
                       the graph is fully connected (one class, all weights equal),
                       so alpha values above this are meaningless for CRW
    """
    print(f"  Loading config: {config_path}")
    config = load_config(Path(config_path))

    print("  Loading dataset (parquet, fast)...")
    dataset = load_dataset(config)

    # Build id_to_text from questions DataFrame — needed regardless of distance source
    text_col = _get_question_text_col(dataset["questions"])
    id_to_text: dict[int, str] = dict(
        zip(dataset["questions"]["ID_question"], dataset["questions"][text_col])
    )

    # Load distances: prefer pre-computed CSV to avoid model inference on home node
    precomputed_csv = _find_precomputed_distances(metric_name)
    if precomputed_csv is not None:
        print(f"  Loading pre-computed distances: {precomputed_csv}")
        dist_df = pd.read_csv(precomputed_csv)
    else:
        print("  No pre-computed distances found — computing (requires model)...")
        calculator = get_calculator(config)
        dist_df = calculator.calculate_distance(dataset, config)

    print("  Building distance matrix...")
    dist_matrix, ids, _ = build_distance_matrix(dist_df, id_to_text=id_to_text)
    id_to_idx = {q_id: i for i, q_id in enumerate(ids)}
    n = len(ids)
    upper = dist_matrix[np.triu_indices(n, k=1)]
    max_pair_dist = float(upper.max())
    print(f"  Distance matrix: {n}x{n} ({n*(n-1)//2} pairs), max pairwise dist={max_pair_dist:.4f}")

    mini_ids, maxi_ids = get_mini_maxi_ids(dataset)
    # Keep only IDs that are present in the distance matrix
    mini_ids = [q for q in mini_ids if q in id_to_idx]
    maxi_ids = [q for q in maxi_ids if q in id_to_idx]
    print(f"  Mini: {len(mini_ids)} questions, Maxi: {len(maxi_ids)} questions")

    total_pairs = len(mini_ids) * len(maxi_ids)
    print(f"  Computing forbidden intervals for {total_pairs} pairs...")

    intervals_rows: list[dict] = []
    coverage_rows: list[dict] = []

    pair_counter = 0
    for x_id in mini_ids:
        for y_id in maxi_ids:
            pair_counter += 1
            if pair_counter % 100 == 1:
                print(f"    {pair_counter}/{total_pairs} pairs processed...", end="\r")

            x_idx = id_to_idx[x_id]
            y_idx = id_to_idx[y_id]
            d_xy = float(dist_matrix[x_idx, y_idx])

            raw = compute_forbidden_intervals(x_idx, y_idx, dist_matrix)
            merged = merge_intervals(raw)
            total_measure = sum(e - s for s, e in merged)

            x_text = id_to_text.get(x_id, "")
            y_text = id_to_text.get(y_id, "")

            for start, end in merged:
                intervals_rows.append(
                    {
                        "q_mini_id": x_id,
                        "q_mini_text": x_text,
                        "q_maxi_id": y_id,
                        "q_maxi_text": y_text,
                        "d_xy": d_xy,
                        "interval_start": start,
                        "interval_end": end,
                        "metric": metric_name,
                    }
                )

            row: dict = {
                "q_mini_id": x_id,
                "q_mini_text": x_text,
                "q_maxi_id": y_id,
                "q_maxi_text": y_text,
                "d_xy": d_xy,
                "n_merged_intervals": len(merged),
                "total_interval_measure": total_measure,
            }
            for a in alpha_grid:
                measure = covered_measure_at_alpha(merged, a)
                fraction = measure / a if a > 0 else 0.0
                row[_measure_col(a)] = measure
                row[_alpha_col(a)] = fraction
                # Effective: restricted to [d_xy, alpha] (adjacent window only)
                eff_m = effective_usable_measure(merged, d_xy, a)
                adj_window = max(0.0, a - d_xy)
                row[_eff_measure_col(a)] = eff_m
                row[_eff_col(a)] = eff_m / a if a > 0.0 else 0.0
            row["metric"] = metric_name
            coverage_rows.append(row)

    print(f"    {total_pairs}/{total_pairs} pairs processed.   ")

    intervals_df = pd.DataFrame(intervals_rows).sort_values("d_xy").reset_index(drop=True)
    coverage_df = pd.DataFrame(coverage_rows).sort_values("d_xy").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Verification: print details for the closest pair so results can be
    # spot-checked against the raw distance CSV.
    # ------------------------------------------------------------------
    _print_verification(coverage_df, dist_matrix, id_to_idx, id_to_text, alpha_grid)

    return intervals_df, coverage_df, max_pair_dist


def _print_verification(
    coverage_df: pd.DataFrame,
    dist_matrix: np.ndarray,
    id_to_idx: dict[int, int],
    id_to_text: dict[int, str],
    alpha_grid: list[float],
) -> None:
    """Print a spot-check of the closest pair so results can be verified manually."""
    closest = coverage_df.iloc[0]
    x_id = int(closest["q_mini_id"])
    y_id = int(closest["q_maxi_id"])
    d_xy = float(closest["d_xy"])
    x_idx = id_to_idx[x_id]
    y_idx = id_to_idx[y_id]

    print(f"\n  --- Verification: closest pair ---")
    print(f"  MINI [{x_id}]: {id_to_text.get(x_id, '')[:80]}")
    print(f"  MAXI [{y_id}]: {id_to_text.get(y_id, '')[:80]}")
    print(f"  d(x,y) = {d_xy:.6f}")

    # Re-compute raw intervals with z identification
    d_xz = dist_matrix[x_idx, :]
    d_yz = dist_matrix[y_idx, :]
    all_ids_sorted = sorted(id_to_idx.keys())
    raw_with_z: list[tuple[float, float, int]] = []
    for z_id in all_ids_sorted:
        if z_id == x_id or z_id == y_id:
            continue
        zi = id_to_idx[z_id]
        lo, hi = min(d_xz[zi], d_yz[zi]), max(d_xz[zi], d_yz[zi])
        if hi > lo:
            raw_with_z.append((lo, hi, z_id))

    raw_with_z.sort(key=lambda t: t[1] - t[0], reverse=True)
    print(f"  Top-5 widest forbidden intervals (before merge):")
    for lo, hi, z_id in raw_with_z[:5]:
        zi = id_to_idx[z_id]
        print(f"    [{lo:.4f}, {hi:.4f}]  width={hi-lo:.4f}  z={z_id}  "
              f"d_xz={d_xz[zi]:.4f}  d_yz={d_yz[zi]:.4f}")

    merged = merge_intervals([(lo, hi) for lo, hi, _ in raw_with_z])
    print(f"  Merged into {len(merged)} intervals, total measure = "
          f"{sum(e-s for s,e in merged):.4f}")

    key_as = [a for a in alpha_grid if a in [0.3, 0.4, 1.1, 1.5]]
    print(f"  Coverage fractions at key alphas (full [0,α] and effective [d_xy,α]):")
    for a in key_as:
        cov = closest.get(_alpha_col(a), float("nan"))
        eff = closest.get(_eff_col(a), 0.0)
        adj = max(0.0, a - d_xy)
        print(f"    α={a:.1f}: coverage={cov:.3f}  eff_usable_frac(of α)={eff:.3f}  "
              f"adj_window={adj:.3f}")
    print(f"  -----------------------------------")


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


def save_outputs(
    intervals_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    metric_name: str,
    output_dir: Path,
    timestamp: str,
) -> tuple[Path, Path]:
    """Save intervals and coverage CSVs. Returns (intervals_path, coverage_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    int_path = output_dir / f"forbidden_intervals_{metric_name}_{timestamp}.csv"
    cov_path = output_dir / f"forbidden_coverage_{metric_name}_{timestamp}.csv"

    intervals_df.to_csv(int_path, index=False)
    coverage_df.to_csv(cov_path, index=False)

    print(f"  Saved: {int_path.name}  ({len(intervals_df)} rows)")
    print(f"  Saved: {cov_path.name}  ({len(coverage_df)} rows)")
    return int_path, cov_path


# ---------------------------------------------------------------------------
# Visualization 1 — horizontal bar chart for closest pairs
# ---------------------------------------------------------------------------


def _shorten(text: str, max_chars: int = 55) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def plot_bar_chart(
    coverage_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    metric_name: str,
    key_alphas: list[float],
    max_pair_dist: float,
    top_k: int,
    output_dir: Path,
    timestamp: str,
) -> Path:
    """Horizontal bar chart showing forbidden (red) vs usable (green) intervals
    for the top_k closest (x_mini, y_maxi) pairs.
    X-axis is capped at max_pair_dist — beyond that the graph is fully connected
    and CRW produces uniform weights with no correction."""
    max_alpha = max_pair_dist
    top_pairs = coverage_df.head(top_k)

    fig, axes = plt.subplots(
        top_k, 1, figsize=(12, 2.5 * top_k + 1.5), squeeze=False
    )

    for row_i, (_, pair_row) in enumerate(top_pairs.iterrows()):
        ax = axes[row_i, 0]
        x_id = pair_row["q_mini_id"]
        y_id = pair_row["q_maxi_id"]
        d_xy = pair_row["d_xy"]

        pair_intervals = intervals_df[
            (intervals_df["q_mini_id"] == x_id)
            & (intervals_df["q_maxi_id"] == y_id)
        ]

        d_xy_clipped = min(d_xy, max_alpha)

        # Green background for full range
        ax.barh(0, max_alpha, left=0, height=0.5, color="#4CAF50", alpha=0.7)

        # Gray overlay for [0, d_xy]: pair not adjacent — irrelevant for CRW
        if d_xy_clipped > 0:
            ax.barh(0, d_xy_clipped, left=0, height=0.5, color="#999999", alpha=0.6)

        # Red forbidden intervals
        for _, iv_row in pair_intervals.iterrows():
            s = iv_row["interval_start"]
            e = min(iv_row["interval_end"], max_alpha)
            if s >= max_alpha:
                continue
            ax.barh(
                0, e - s, left=s, height=0.5,
                color="#F44336", alpha=0.85,
            )

        # d(x,y) marker
        if d_xy < max_alpha:
            ax.axvline(x=d_xy, color="black", linestyle=":", linewidth=1.5)
            ax.text(d_xy, -0.3, f"d(x,y)={d_xy:.3f}", ha="center", va="top",
                    fontsize=7, color="black")

        # Key alpha markers
        for ka in key_alphas:
            if ka <= max_alpha:
                eff = pair_row.get(_eff_col(ka), float("nan"))
                ax.axvline(x=ka, color="navy", linestyle="--", linewidth=1.2)
                ax.text(
                    ka, 0.42, f"α={ka}\neff={eff:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color="navy", fontweight="bold",
                )

        ax.set_xlim(0, max_alpha)
        ax.set_ylim(-0.5, 0.8)
        ax.set_yticks([])

        mini_short = _shorten(pair_row["q_mini_text"])
        maxi_short = _shorten(pair_row["q_maxi_text"])
        ax.set_xlabel("Distance r" if row_i == top_k - 1 else "")
        ax.set_title(
            f"Pair {row_i + 1}  |  d(x,y)={d_xy:.4f}\n"
            f"MINI  [{x_id}]: {mini_short}\n"
            f"MAXI  [{y_id}]: {maxi_short}",
            fontsize=8.5,
            loc="left",
        )

        if row_i == 0:
            ax.legend(
                handles=[
                    mpatches.Patch(color="#4CAF50", alpha=0.7, label="Usable (adjacent)"),
                    mpatches.Patch(color="#F44336", alpha=0.85, label="Forbidden"),
                    mpatches.Patch(color="#999999", alpha=0.6, label="Not adjacent [0, d(x,y)]"),
                ],
                loc="upper right",
                fontsize=8,
            )

    fig.suptitle(
        f"Forbidden Interval Coverage — {metric_name}\n"
        f"Top {top_k} closest (mini, maxi) pairs",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    out_path = output_dir / f"forbidden_bar_{metric_name}_{timestamp}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Visualization 2 — per-pair usable fraction curves
# ---------------------------------------------------------------------------


def plot_usable_curves(
    coverage_df: pd.DataFrame,
    metric_name: str,
    key_alphas: list[float],
    alpha_grid: list[float],
    max_pair_dist: float,
    top_k: int,
    output_dir: Path,
    timestamp: str,
) -> Path:
    """Line plot of usable fraction (1 - coverage) vs alpha for the top_k closest pairs."""
    n_curves = min(top_k, len(coverage_df), 10)
    top_pairs = coverage_df.head(n_curves)

    cmap = plt.cm.plasma  # type: ignore[attr-defined]
    d_vals = np.asarray(top_pairs["d_xy"].values, dtype=float)
    d_min, d_max = float(np.nanmin(d_vals)), float(np.nanmax(d_vals))
    norm_d = (d_vals - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(d_vals)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (_, row) in enumerate(top_pairs.iterrows()):
        eff_plot = [row.get(_eff_col(a), 0.0) for a in alpha_grid]
        color = cmap(0.15 + 0.7 * float(norm_d[i]))
        label = f"[{row['q_mini_id']}↔{row['q_maxi_id']}]  d={row['d_xy']:.4f}"
        ax.plot(alpha_grid, eff_plot, color=color, linewidth=1.5, label=label)

    for ka in key_alphas:
        ax.axvline(x=ka, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(ka, 0.01, f"α={ka}", ha="center", va="bottom", fontsize=8, color="gray")

    # Mark max_pair_dist: beyond this the graph is fully connected (trivially usable)
    ax.axvline(x=max_pair_dist, color="black", linestyle="-", linewidth=1.2, alpha=0.5)
    ax.axvspan(max_pair_dist, max(alpha_grid), alpha=0.06, color="black")
    ax.text(max_pair_dist, 0.5, " fully\n connected", ha="left", va="center",
            fontsize=7.5, color="black", alpha=0.6)

    ax.set_xlim(min(alpha_grid), max(alpha_grid))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("α")
    ax.set_ylabel("Effective usable fraction of [d(x,y), α]")
    ax.set_title(
        f"Effective Usable Fraction vs α — {metric_name}\n"
        f"Top {n_curves} closest (mini, maxi) pairs  (0 where d(x,y) > α)"
    )
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"forbidden_curve_{metric_name}_{timestamp}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Visualization 3 — aggregate usable fraction distribution across all pairs
# ---------------------------------------------------------------------------


def plot_aggregate_usable(
    coverage_df: pd.DataFrame,
    metric_name: str,
    key_alphas: list[float],
    alpha_grid: list[float],
    max_pair_dist: float,
    output_dir: Path,
    timestamp: str,
) -> Path:
    """Distribution of usable fraction vs alpha across all mini/maxi pairs.

    Shows median, mean, IQR ribbon (25–75th pct), and 10–90th pct ribbon.
    """
    # Build matrix using effective usable fraction (NaN where d_xy > alpha)
    eff_matrix = np.array(
        [
            [row.get(_eff_col(a), float("nan")) for a in alpha_grid]
            for _, row in coverage_df.iterrows()
        ],
        dtype=float,
    )  # shape (n_pairs, n_alphas), NaN where pair not adjacent at that alpha

    median = np.median(eff_matrix, axis=0)
    mean = np.mean(eff_matrix, axis=0)
    p10 = np.percentile(eff_matrix, 10, axis=0)
    p25 = np.percentile(eff_matrix, 25, axis=0)
    p75 = np.percentile(eff_matrix, 75, axis=0)
    p90 = np.percentile(eff_matrix, 90, axis=0)
    # Pairs where d_xy < alpha (adjacent window non-empty)
    d_xy_vals = np.asarray(coverage_df["d_xy"].values, dtype=float)
    n_adjacent = np.array([int(np.sum(d_xy_vals < a)) for a in alpha_grid])

    alphas_arr = np.array(alpha_grid)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(alphas_arr, p10, p90, alpha=0.15, color="steelblue", label="10–90th pct")
    ax.fill_between(alphas_arr, p25, p75, alpha=0.30, color="steelblue", label="25–75th pct (IQR)")
    ax.plot(alphas_arr, median, color="steelblue", linewidth=2.0, label="Median")
    ax.plot(alphas_arr, mean, color="darkorange", linewidth=1.5, linestyle="--", label="Mean")

    for ka in key_alphas:
        ax.axvline(x=ka, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        # Compute fraction of pairs with usable > 0.5 at this alpha
        ka_matches = [i for i, a in enumerate(alpha_grid) if abs(a - ka) < 1e-9]
        ka_idx = ka_matches[0] if ka_matches else None
        if ka_idx is not None:
            col = eff_matrix[:, ka_idx]
            n_adj = int(n_adjacent[ka_idx])
            frac_above_half = float(np.mean(col > 0.5)) if n_adj > 0 else 0.0
            ax.text(
                ka, 1.02,
                f"α={ka}\nn_adj={n_adj}\n{frac_above_half:.0%} >50%",
                ha="center", va="bottom", fontsize=7.5, color="gray",
            )

    # Mark max_pair_dist: beyond this the graph is fully connected (trivially usable)
    ax.axvline(x=max_pair_dist, color="black", linestyle="-", linewidth=1.2, alpha=0.5)
    ax.axvspan(max_pair_dist, max(alpha_grid), alpha=0.06, color="black")
    ax.text(max_pair_dist, 0.55, " fully\n connected\n (trivial)", ha="left", va="center",
            fontsize=7.5, color="black", alpha=0.6)

    ax.set_xlim(min(alpha_grid), max(alpha_grid))
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("α")
    ax.set_ylabel("Effective usable fraction of [d(x,y), α]")
    ax.set_title(
        f"Aggregate Effective Usable Fraction vs α — {metric_name}\n"
        f"All {len(coverage_df)} mini/maxi pairs"
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"forbidden_aggregate_{metric_name}_{timestamp}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    metrics_to_run = [m.strip() for m in args.metrics.split(",")]
    timestamp = datetime.now().strftime("%m%d_%H%M")

    for metric_name in metrics_to_run:
        if metric_name not in METRIC_CONFIGS:
            print(f"Warning: unknown metric '{metric_name}'. Valid: {list(METRIC_CONFIGS)}")
            continue

        cfg = METRIC_CONFIGS[metric_name]
        output_dir = BASE_RESULTS_DIR / metric_name

        print(f"\n{'='*65}")
        print(f"  Metric : {metric_name}")
        print(f"  Output : {output_dir}/")
        print(f"{'='*65}")

        intervals_df, coverage_df, max_pair_dist = run_metric_analysis(
            metric_name=metric_name,
            config_path=cfg["config_path"],
            alpha_grid=ALPHA_GRID,
        )
        print(f"  Max pairwise distance (fully-connected threshold): {max_pair_dist:.4f}")

        save_outputs(intervals_df, coverage_df, metric_name, output_dir, timestamp)

        if not args.no_viz:
            plot_bar_chart(
                coverage_df,
                intervals_df,
                metric_name,
                cfg["key_alphas"],
                max_pair_dist,
                args.top_k,
                output_dir,
                timestamp,
            )
            plot_usable_curves(
                coverage_df,
                metric_name,
                cfg["key_alphas"],
                ALPHA_GRID,
                max_pair_dist,
                args.top_k,
                output_dir,
                timestamp,
            )
            plot_aggregate_usable(
                coverage_df,
                metric_name,
                cfg["key_alphas"],
                ALPHA_GRID,
                max_pair_dist,
                output_dir,
                timestamp,
            )

    print(f"\nDone. Results in {BASE_RESULTS_DIR}/")


if __name__ == "__main__":
    main()
