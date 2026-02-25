import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde


_COLOR_BASE = "#9E9E9E"
_COLOR_CRW = "#2196F3"
_COLOR_IMPROVEMENT = "#4CAF50"


class CrossRunPlotter:
    """
    Generates publication-ready plots for a cross-run comparison.
    Produces three PNG files per comparison run:
      - {base}_distributions.png  — 2x2 grid: Jaccard PMF + ECDF, Spearman KDE, Kendall KDE
      - {base}_metrics.png        — grouped bar chart of Jaccard, Spearman, Kendall
      - {base}_improvement.png    — 2x2 KDE grid of per-voter CRW improvement over baseline
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def save_plots(
        self,
        df: pd.DataFrame,
        base: str,
        meta_a: dict,
        meta_b: dict,
    ) -> list[Path]:
        sns.set_theme(style="whitegrid")
        name_a = self._clean_name(meta_a)
        name_b = self._clean_name(meta_b)

        n_a = meta_a.get("n_jaccard")
        n_b = meta_b.get("n_jaccard")
        n_jaccard = min(v for v in [n_a, n_b] if v is not None) if (n_a or n_b) else None

        dist_path = self.output_dir / f"{base}_distributions.png"
        metrics_path = self.output_dir / f"{base}_metrics.png"
        improvement_path = self.output_dir / f"{base}_improvement.png"

        self._plot_distributions(df, dist_path, name_a, name_b, n_jaccard)
        self._plot_metrics(df, metrics_path, name_a, name_b)
        self._plot_improvement(df, improvement_path, name_a, name_b)

        print(f"  -> Distributions plot: {dist_path.name}")
        print(f"  -> Metrics plot:       {metrics_path.name}")
        print(f"  -> Improvement plot:   {improvement_path.name}")
        return [dist_path, metrics_path, improvement_path]

    # ------------------------------------------------------------------
    # Plot 1: distributions (2x2 mosaic)
    # ------------------------------------------------------------------

    def _plot_distributions(
        self, df: pd.DataFrame, path: Path, name_a: str, name_b: str, n_jaccard: int | None
    ) -> None:
        # Jaccard row: PMF (left) + ECDF (right). Spearman + Kendall KDEs below.
        # avg_pos_moved is omitted here — it appears in the improvement plot.
        mosaic = [["jac_pmf", "jac_ecdf"], ["spearman", "kendall"]]
        fig, axes = plt.subplot_mosaic(mosaic, figsize=(14, 10))
        fig.suptitle(f"{name_a}  vs  {name_b}", fontsize=11, y=1.01)

        if n_jaccard is not None:
            self._jaccard_pmf_panel(axes["jac_pmf"], df["base_jaccard"], df["crw_jaccard"], n_jaccard)
            self._jaccard_ecdf_panel(axes["jac_ecdf"], df["base_jaccard"], df["crw_jaccard"], n_jaccard)
        else:
            # Fallback for old result files that lack n_jaccard in metadata
            self._kde_panel(axes["jac_pmf"], df["base_jaccard"], df["crw_jaccard"],
                            "Jaccard Similarity (Top-k)", "Jaccard Similarity", (0.0, 1.0))
            axes["jac_ecdf"].set_visible(False)

        self._kde_panel(axes["spearman"], df["base_spearman"], df["crw_spearman"],
                        "Spearman ρ", "Spearman Correlation", (-1.0, 1.0))
        self._kde_panel(axes["kendall"], df["base_kendall"], df["crw_kendall"],
                        "Kendall τ", "Kendall Correlation", (-1.0, 1.0))

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _jaccard_pmf_panel(
        self,
        ax: plt.Axes,
        base_vals: pd.Series,
        crw_vals: pd.Series,
        n: int,
    ) -> None:
        """Discrete PMF bar chart. Each bar corresponds to one possible Jaccard value."""
        # All possible Jaccard values for top-k = n: m / (2n - m), m = 0..n
        possible = np.array([m / (2 * n - m) for m in range(n + 1)])

        # Bin edges: midpoints between consecutive Jaccard values, extended at both ends
        gaps = np.diff(possible)
        edges = np.empty(len(possible) + 1)
        edges[0] = possible[0] - gaps[0] / 2
        edges[1:-1] = (possible[:-1] + possible[1:]) / 2
        edges[-1] = possible[-1] + gaps[-1] / 2

        base_arr = base_vals.dropna().values
        crw_arr = crw_vals.dropna().values

        base_counts, _ = np.histogram(base_arr, bins=edges)
        crw_counts, _ = np.histogram(crw_arr, bins=edges)

        base_freq = base_counts / base_counts.sum()
        crw_freq = crw_counts / crw_counts.sum()

        # Bar widths match each discrete cell (variable — honest about the uneven spacing)
        widths = np.diff(edges) * 0.85

        ax.bar(possible, base_freq, width=widths, color=_COLOR_BASE, alpha=0.6,
               label="Baseline", edgecolor=_COLOR_BASE, linewidth=0.3)
        ax.bar(possible, crw_freq, width=widths, color=_COLOR_CRW, alpha=0.6,
               label="CRW", edgecolor=_COLOR_CRW, linewidth=0.3)

        ax.axvline(base_vals.mean(), color=_COLOR_BASE, linestyle=":", linewidth=1.5, alpha=0.8)
        ax.axvline(crw_vals.mean(), color=_COLOR_CRW, linestyle=":", linewidth=1.5, alpha=0.8)

        ax.set_title(f"Jaccard Similarity — PMF (Top-{n})", fontsize=11)
        ax.set_xlabel("Jaccard Similarity")
        ax.set_ylabel("Proportion of Voters")
        ax.set_xlim(0.0, 1.0)
        ax.legend(frameon=True)

    def _jaccard_ecdf_panel(
        self,
        ax: plt.Axes,
        base_vals: pd.Series,
        crw_vals: pd.Series,
        n: int,
    ) -> None:
        """ECDF of Jaccard similarity. No smoothing; exact cumulative fractions."""
        for vals, color, label, ls in [
            (base_vals, _COLOR_BASE, "Baseline", "--"),
            (crw_vals, _COLOR_CRW, "CRW", "-"),
        ]:
            arr = np.sort(vals.dropna().values)
            total = len(arr)
            y = np.arange(1, total + 1) / total
            # Prepend (0, 0) so the step function starts at the left edge
            x_plot = np.concatenate([[0.0], arr])
            y_plot = np.concatenate([[0.0], y])
            ax.step(x_plot, y_plot, where="post", color=color, linestyle=ls,
                    linewidth=2, label=label)
            ax.axvline(vals.mean(), color=color, linestyle=":", linewidth=1.5, alpha=0.8)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_title(f"Jaccard Similarity — ECDF (Top-{n})", fontsize=11)
        ax.set_xlabel("Jaccard Similarity")
        ax.set_ylabel("Fraction of Voters ≤ x")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=True)

    def _kde_panel(
        self,
        ax: plt.Axes,
        base_vals: pd.Series,
        crw_vals: pd.Series,
        title: str,
        xlabel: str,
        xlim,
    ) -> None:

        def _plot_kde(vals, color, label, linestyle):
            vals = vals.dropna()
            kde = gaussian_kde(vals, bw_method="scott")
            x_min = vals.min()
            x_max = vals.max()
            x = np.linspace(x_min, x_max, 500)
            y = kde(x)
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=2, label=label)
            ax.fill_between(x, y, alpha=0.15, color=color)
            mean_val = vals.mean()
            ax.axvline(mean_val, color=color, linestyle=":", linewidth=1.5, alpha=0.8)

        _plot_kde(base_vals, _COLOR_BASE, "Baseline", linestyle="--")
        _plot_kde(crw_vals, _COLOR_CRW, "CRW", linestyle="-")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(left=0)
        ax.legend(frameon=True)

    # ------------------------------------------------------------------
    # Plot 2: Summary metrics bar chart
    # ------------------------------------------------------------------

    def _plot_metrics(
        self, df: pd.DataFrame, path: Path, name_a: str, name_b: str
    ) -> None:
        metrics = [
            ("Jaccard\n(mean)", df["base_jaccard"].mean(), df["crw_jaccard"].mean()),
            ("Spearman ρ", df["base_spearman"].mean(), df["crw_spearman"].mean()),
            ("Kendall τ", df["base_kendall"].mean(), df["crw_kendall"].mean()),
        ]
        labels = [m[0] for m in metrics]
        base_vals = [m[1] for m in metrics]
        crw_vals = [m[2] for m in metrics]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 5))
        bars_base = ax.bar(
            x - width / 2, base_vals, width, label="Baseline", color=_COLOR_BASE
        )
        bars_crw = ax.bar(
            x + width / 2, crw_vals, width, label="CRW", color=_COLOR_CRW
        )

        for bar in bars_base:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=_COLOR_BASE,
            )
        for bar in bars_crw:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=_COLOR_CRW,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(
            f"Recommendation Stability: Baseline vs CRW\n{name_a}  vs  {name_b}",
            fontsize=10,
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 3: CRW improvement over baseline (per-voter deltas as KDEs)
    # ------------------------------------------------------------------

    def _plot_improvement(
        self, df: pd.DataFrame, path: Path, name_a: str, name_b: str
    ) -> None:
        # Positive delta = CRW is better than baseline
        deltas = {
            "Δ Jaccard": df["crw_jaccard"] - df["base_jaccard"],
            "Δ Spearman ρ": df["crw_spearman"] - df["base_spearman"],
            "Δ Kendall τ": df["crw_kendall"] - df["base_kendall"],
            "Δ Avg. Positions Moved\n(baseline − CRW, positive = better)": (
                df["base_avg_pos_moved"] - df["crw_avg_pos_moved"]
            ),
        }

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"CRW Improvement over Baseline\n{name_a}  vs  {name_b}",
            fontsize=11,
            y=1.01,
        )

        for ax, (title, vals) in zip(axes.flat, deltas.items()):
            self._improvement_kde_panel(ax, vals, title)

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _improvement_kde_panel(
        self, ax: plt.Axes, vals: pd.Series, title: str
    ) -> None:
        vals = vals.dropna()
        kde = gaussian_kde(vals, bw_method="scott")
        x = np.linspace(vals.min(), vals.max(), 500)
        y = kde(x)

        ax.plot(x, y, color=_COLOR_IMPROVEMENT, linewidth=2)
        ax.fill_between(x, y, alpha=0.2, color=_COLOR_IMPROVEMENT)

        mean_val = vals.mean()
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.6, label="No change")
        ax.axvline(mean_val, color=_COLOR_IMPROVEMENT, linestyle=":", linewidth=1.5,
                   alpha=0.9, label=f"Mean: {mean_val:+.3f}")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Delta (CRW − Baseline)")
        ax.set_ylabel("Density")
        ax.legend(frameon=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_name(self, meta: dict) -> str:
        base = meta.get("config_name", "unknown")
        overrides = meta.get("overrides", [])
        if overrides:
            suffix = "_".join(overrides).replace("~", "").replace("=", "")
            return f"{base}_{suffix}"
        return base
