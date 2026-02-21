import pandas as pd
import numpy as np
import json
from pathlib import Path

from vqs.result_management import ResultManager

"""
Important Assumption: 
We assume equal number of recommendations for all voters. Achieved either by setting n_recommendations to a fixed number, or by filtering voters and candidates by a certain canton. 
Any NaN values will raise an error.
"""


def _get_prefix(config) -> str:
    base_name = Path(config.__file__).stem
    override_str = ("_" + "_".join(config.overrides)) if config.overrides else ""
    return f"recs_{base_name}{override_str}"


def _print_summary(file_path: Path | None = None, text: str | None = None):
    if text:
        print(text)
    elif file_path:
        txt_path = file_path.with_suffix(".txt")
        if txt_path.exists():
            print(txt_path.read_text(encoding="utf-8"))
        else:
            print("No summary available.")
    else:
        print("No summary available.")


def save_recommendation_results(
    df: pd.DataFrame, config, important_params_list: list[str]
):
    """
    Saves recommendation results, metadata, and prints a summary analysis of changes.
    """

    # 1. Check for cached files
    base_path = Path(config.RECOMMENDATION_RESULTS_DIR)

    output_dir = base_path / config.data_choice

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _get_prefix(config)
    rm = ResultManager(
        config=config,
        dir=output_dir,
        params_list=important_params_list,
        prefix=prefix,
    )

    path = rm.exists()
    if path:
        print(f"--- [Skip Save] Result with hash {rm.hash} already exists: ---")
        print(f"    -> {path.name}")
        _print_summary(path)
        return

    # 2. Save the Main Data (Parquet)
    file_path = rm.save(data=df, readable=True)

    # 3. Calculate N for Jaccard (Used for Summary & Metadata)
    # We do this early so we can save it to the JSON
    base_match_cols = [
        c
        for c in df.columns
        if c.startswith("_matchID_") and c.endswith(f"_{config.rec_dist_method}")
    ]
    crw_match_cols = [
        c
        for c in df.columns
        if c.startswith("CRW__matchID_") and c.endswith(f"_{config.rec_dist_method}")
    ]

    n_jaccard = _get_jaccard_n(
        df=df, config=config, base_cols=base_match_cols, crw_cols=crw_match_cols
    )

    # 4. Save Metadata (JSON)
    # This allows the CrossRunAnalyzer to know exactly how this run was configured
    if file_path is not None:
        _save_metadata(file_path, config, n_jaccard, important_params_list)

    # 5. Generate and Save Text Summary
    summary_text = _generate_stats(
        df=df,
        config=config,
        method=config.rec_dist_method,
        base_cols=base_match_cols,
        crw_cols=crw_match_cols,
        n_jaccard=n_jaccard,
    )

    if file_path is not None:
        txt_path = file_path.with_suffix(".txt")
        txt_path.write_text(summary_text, encoding="utf-8")
        print(f"  -> {txt_path}")

    _print_summary(text=summary_text)


def _save_metadata(
    file_path: Path, config, n_jaccard: int, important_params_list: list[str]
):
    """
    Saves a JSON file alongside the parquet file containing configuration metadata.
    This allows the Analyzer to automatically detect 'n' and other parameters.
    """
    # 1. Start with the explicit parameters the Analyzer strictly needs or that need formatting
    metadata = {
        "config_name": Path(config.__file__).stem,
        "overrides": config.overrides,  # Kept for historical context (CLI commands)
        "n_jaccard": int(n_jaccard),
    }

    # 2. Dynamically add everything deemed "important" by the main pipeline (ensures Analyzer matches engine hash logic)
    for param in important_params_list:
        if hasattr(config, param):
            val = getattr(config, param)
            # Basic type conversion for safety, though json.dump handles most primitives
            if isinstance(val, (np.integer, np.floating)):
                val = val.item()
            metadata[param] = val

    # # 3. Add explicit extras that might not be in important_params_list but are useful
    # # (Only adds them if they weren't already added by the loop above)
    # extras = ["subset_n", "E5_instruction", "use_OG_weights"]
    # for extra in extras:
    #     if extra not in metadata and hasattr(config, extra):
    #         val = getattr(config, extra)
    #         metadata[extra] = val

    json_path = file_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)


def _generate_stats(
    df, config, method, base_cols=None, crw_cols=None, n_jaccard=30
) -> str:
    # If cols weren't passed (legacy support), find them again
    if base_cols is None:
        base_cols = [
            c
            for c in df.columns
            if c.startswith("_matchID_") and c.endswith(f"_{method}")
        ]
    if crw_cols is None:
        crw_cols = [
            c
            for c in df.columns
            if c.startswith("CRW__matchID_") and c.endswith(f"_{method}")
        ]

    _safety_checks(df=df, base_cols=base_cols, crw_cols=crw_cols)

    total_candidates = len(base_cols)

    jaccard_scores, candidate_changes = _calculate_jaccard(
        config=config,
        df=df,
        base_cols=base_cols,
        crw_cols=crw_cols,
        n=n_jaccard,
    )
    rank_stats = _calculate_rank_metrics(df=df, base_cols=base_cols, crw_cols=crw_cols)

    summary_str = (
        f"\n--- Stats Summary {config.data_year}---\n"
        f"Scope: Evaluating top {n_jaccard} recommendations for Set Similarity, and all {total_candidates} slots for Rank Stability.\n\n"
        f"Set Similarity (Top {n_jaccard}):\n"
        f"  - Average Jaccard Similarity:       {np.mean(jaccard_scores):.4f} (1.0 = identical sets)\n"
        f"  - Minimum Jaccard Similarity:       {np.min(jaccard_scores):.4f}\n"
        f"  - Avg. Candidates Swapped In/Out:   {np.mean(candidate_changes):.2f} candidates per voter\n"
        f"  - Max Candidates Swapped In/Out:    {np.max(candidate_changes)} candidates\n\n"
        f"Rank Stability (All {total_candidates}):\n"
        f"  - Voters w/ at least 1 rank change: {rank_stats['voters_with_change']:.2f}%\n"
        f"  - Avg. candidates changed per voter:{rank_stats['avg_changed_cands']:>6.2f} candidates ({rank_stats['avg_pct_list_changed']:.2f}% of list)\n"
        f"  - Avg. positions a candidate moved: {rank_stats['avg_shift']:>6.2f} ranks\n"
        f"  - Max positions a candidate moved:  {rank_stats['max_shift']:>6} ranks\n"
        f"----------------------------\n"
    )
    return summary_str


def _safety_checks(df, base_cols, crw_cols):
    if len(base_cols) != len(crw_cols):
        raise ValueError(
            f"⚠️ Mismatch in number of match columns: {len(base_cols)} baseline vs {len(crw_cols)} CRW. Check column naming conventions."
        )
    if len(base_cols) == 0:
        raise ValueError(
            "⚠️ No match columns found! Check that your recommendation DataFrame has the expected column naming pattern."
        )
    if df[base_cols + crw_cols].isna().any().any():
        raise ValueError(
            "⚠️ Critical Error: NaN values detected in recommendation matches."
        )


def _get_jaccard_n(df, config, base_cols, crw_cols):
    n = 0
    if config.n_recommendations == "all":
        if config.district != "all":
            n = (
                config.SEATS_PER_CANTON.get(config.district)
                if config.data_year == 2023
                else config.SEATS_PER_CANTON19.get(config.district)
            )
        else:
            n = 30  # Default fallback if all candidates are ranked
    elif config.n_recommendations is not None:
        n = config.n_recommendations
    else:
        n = len(base_cols)
    return n


def _calculate_jaccard(config, df, base_cols, crw_cols, n=30):
    # Slice the matrix to top-n for Jaccard calculation
    base_matrix = df[base_cols].values[:, :n]
    crw_matrix = df[crw_cols].values[:, :n]

    jaccard_scores = []
    candidate_changes = []
    for b, c in zip(base_matrix, crw_matrix):
        s1 = set(b[~pd.isna(b)])
        s2 = set(c[~pd.isna(c)])
        intersection = len(s1 & s2)
        union = len(s1 | s2)

        jaccard_scores.append(intersection / union if union > 0 else 1.0)
        candidate_changes.append(n - intersection)

    return jaccard_scores, candidate_changes


def _calculate_rank_metrics(df, base_cols, crw_cols):
    base_matrix = df[base_cols].values
    crw_matrix = df[crw_cols].values
    n_total = len(base_cols)

    # 1. Broad list changes (How many candidates shifted at all?)
    matches = base_matrix == crw_matrix
    stable_candidates_per_voter = matches.sum(axis=1)
    changed_candidates_per_voter = n_total - stable_candidates_per_voter

    # 2. Specific rank shifts (How far did they move?)
    # Optimized loop with dict lookup for O(1) matching
    total_shifts = 0
    shift_count = 0
    max_shift = 0

    for b, c in zip(base_matrix, crw_matrix):
        # Strip NaNs for safety
        b_clean = b[~pd.isna(b)]
        c_clean = c[~pd.isna(c)]

        # Map Candidate ID -> Rank Index in the CRW list
        c_ranks = {val: idx for idx, val in enumerate(c_clean)}

        # Compare to their Rank Index in the baseline list
        for idx, val in enumerate(b_clean):
            if val in c_ranks:
                shift = abs(idx - c_ranks[val])
                total_shifts += shift
                shift_count += 1
                if shift > max_shift:
                    max_shift = shift

    avg_shift = (total_shifts / shift_count) if shift_count > 0 else 0

    # Package stats into a dictionary for clean extraction
    return {
        "voters_with_change": (changed_candidates_per_voter > 0).mean() * 100,
        "avg_changed_cands": changed_candidates_per_voter.mean(),
        "avg_pct_list_changed": (changed_candidates_per_voter / n_total * 100).mean(),
        "avg_shift": avg_shift,
        "max_shift": max_shift,
    }
