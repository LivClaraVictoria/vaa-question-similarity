"""
Shared building blocks for in-memory cloning experiments.

Clones are created on the fly from the loaded dataset (no cloned parquets on disk): build the
CloneSpecs, apply them, then compute distances / recommendations for the cloned side. Nothing here
writes caches, so one (question, alpha, clone type) task costs memory and CPU but no storage.
"""

from types import SimpleNamespace

import pandas as pd

from clone_pipeline.applicator import apply_specs
from clone_pipeline.spec import CloneSpec
from experiments._common import FLIP_TYPES, PERFECT_MIX_COMPONENTS
from experiments.perfect_clones.model_selection import _setup_side
from vqs.clone_robust_weighting import CloneRobustReweighter


def build_clone_specs(q_id: int, clone_type: str, n_clones: int) -> list[CloneSpec]:
    """CloneSpecs for cloning `q_id` `n_clones` times.

    perfect_mix splits n_clones equally over PERFECT_MIX_COMPONENTS (n_clones must be a multiple).
    """
    if clone_type == "perfect_mix":
        per_type = n_clones // len(PERFECT_MIX_COMPONENTS)
        return [
            CloneSpec(
                source_q_id=q_id, clone_type=ct, n_clones=per_type,
                flip_answers=(ct in FLIP_TYPES),
            )
            for ct in PERFECT_MIX_COMPONENTS
        ]
    return [CloneSpec(
        source_q_id=q_id, clone_type=clone_type, n_clones=n_clones,
        flip_answers=(clone_type in FLIP_TYPES),
    )]


def clone_dataset(dataset: dict, specs: list[CloneSpec], paraphrases: dict | None) -> dict:
    """Apply `specs` to the questions/voters/candidates of `dataset`; returns the cloned data map."""
    return apply_specs(
        specs=specs,
        dataframes={
            "questions": dataset["questions"],
            "voters": dataset["voters"],
            "candidates": dataset["candidates"],
        },
        paraphrases=paraphrases,
    )


def setup_cloned_side(config, dataset: dict, specs: list[CloneSpec], paraphrases: dict | None,
                      clone_id: str):
    """Clone `dataset` in memory and set up the cloned side (distances, rec engine, baseline).

    Returns (cloned_config, side) where `side` has the keys of model_selection._setup_side.
    `clone_id` must be unique per cloned dataset (it is part of the distance cache hash).
    """
    cloned_data = clone_dataset(dataset, specs, paraphrases)
    cloned_config = SimpleNamespace(**vars(config))
    cloned_config.clone_id = clone_id
    return cloned_config, _setup_side(cloned_config, dataset=cloned_data)


def combine_with_crw(baseline: pd.DataFrame, crw_df: pd.DataFrame) -> pd.DataFrame:
    """Join plain recommendations with the CRW ones (columns prefixed `CRW_`)."""
    match_cols = [c for c in crw_df.columns if "match" in c or "Dist" in c]
    return baseline.join(crw_df[match_cols].add_prefix("CRW_"))


def recs_at_alpha(config, side: dict, alpha: float, cached: bool = False) -> pd.DataFrame:
    """Baseline + CRW recommendations of one side at `alpha`.

    cached=True uses the on-disk base-side CRW cache (only for the un-cloned side, see
    RecommendationEngine.run_crw_cached). Sets config.alpha as a side effect, like the reweighter expects.
    """
    config.alpha = alpha
    weights = CloneRobustReweighter(config).reweight(side["dist_df"])
    engine = side["rec_engine"]
    crw = engine.run_crw_cached(weights) if cached else engine.run_crw(weights)
    return combine_with_crw(side["baseline"], crw)
