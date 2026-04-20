"""
Shared CLI utilities used across all experiment scripts.
"""

from pathlib import Path

import pandas as pd

_alphas_low = [round(0.1 * i, 1) for i in range(1, 16)]   # 0.1 to 1.5
_alphas_high = [round(1.5 + (0.3 * i), 1) for i in range(1, 6)]  # 1.8 to 3.0
DEFAULT_ALPHAS = [0.01] + _alphas_low + _alphas_high
DEFAULT_ALPHA_REFERENCE = 0.6


def _get_clean_name(config) -> str:
    """Readable run name (matches CrossRunSaver._clean_name logic)."""
    base = Path(config.__file__).stem
    overrides = getattr(config, "overrides", [])
    if overrides:
        suffix = "_".join(overrides).replace("~", "").replace("=", "")
        return f"{base}_{suffix}"
    return base


def _resolve_n(config, n_override: int | None) -> int:
    """Determine Jaccard top-k from CLI override or config (mirrors recommendation_saver._get_jaccard_n)."""
    if n_override is not None:
        return n_override
    if config.n_recommendations == "all":
        if config.district != "all":
            seats = (
                config.SEATS_PER_CANTON.get(config.district)
                if config.data_year == 2023
                else config.SEATS_PER_CANTON19.get(config.district)
            )
            return seats if seats else 30
        return 30
    elif config.n_recommendations is not None:
        return config.n_recommendations
    return 30


def _get_question_text_col(df: pd.DataFrame) -> str:
    """Return the question text column name, handling 2019 vs 2023 naming variants."""
    for col in df.columns:
        if "question_en" in col.lower():
            return col
    raise ValueError("No question text column found")
