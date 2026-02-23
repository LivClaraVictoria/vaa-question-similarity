import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

_COLOR_BASE = "#9E9E9E"
_COLOR_CRW = "#2196F3"


class CrossRunPlotter:
    """
    Generates publication-ready plots for a cross-run comparison.
    Produces two PNG files per comparison run:
      - {base}_distributions.png  — KDE distributions of Jaccard and avg position moved
      - {base}_metrics.png        — grouped bar chart of Jaccard, Spearman, Kendall
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

        dist_path = self.output_dir / f"{base}_distributions.png"
        metrics_path = self.output_dir / f"{base}_metrics.png"

        self._plot_distributions(df, dist_path, name_a, name_b)
        self._plot_metrics(df, metrics_path, name_a, name_b)

        print(f"  -> Distributions plot: {dist_path.name}")
        print(f"  -> Metrics plot:       {metrics_path.name}")
        return [dist_path, metrics_path]

    # ------------------------------------------------------------------
    # Plot 1: KDE distributions
    # ------------------------------------------------------------------

    def _plot_distributions(
        self, df: pd.DataFrame, path: Path, name_a: str, name_b: str
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{name_a}  vs  {name_b}", fontsize=11, y=1.01)

        # Left: Jaccard
        self._kde_panel(
            ax=axes[0],
            base_vals=df["base_jaccard"],
            crw_vals=df["crw_jaccard"],
            title="Jaccard Similarity (Top-k)",
            xlabel="Jaccard Similarity",
            xlim=(0.0, 1.0),
        )

        # Right: Avg position moved
        self._kde_panel(
            ax=axes[1],
            base_vals=df["base_avg_pos_moved"],
            crw_vals=df["crw_avg_pos_moved"],
            title="Avg. Candidate Rank Positions Moved",
            xlabel="Avg. Positions Moved",
            xlim=None,  # auto
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _kde_panel(
        self,
        ax: plt.Axes,
        base_vals: pd.Series,
        crw_vals: pd.Series,
        title: str,
        xlabel: str,
        xlim,
    ) -> None:
        from scipy.stats import gaussian_kde

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

        # Value labels on top of each bar
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
    # Helpers
    # ------------------------------------------------------------------

    def _clean_name(self, meta: dict) -> str:
        base = meta.get("config_name", "unknown")
        overrides = meta.get("overrides", [])
        if overrides:
            suffix = "_".join(overrides).replace("~", "").replace("=", "")
            return f"{base}_{suffix}"
        return base
