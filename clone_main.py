"""
Entry point for the clone generation pipeline.

Usage:
    python clone_main.py --config configs/create_clones/identical_q32214_n10.py
"""

import argparse
import importlib.util
from pathlib import Path

from configs.create_clones.base_clone_creator import (
    CLEANED_DIR,
    CLONED_DIR,
    DATA_DIR,
    CANDIDATES_PREFIX,
    CANDIDATES_19_PREFIX,
    VOTERS_PREFIX,
    VOTERS_19_PREFIX,
    QUESTIONS_2023_PATH,
    QUESTIONS_2019_PATH,
)
from clone_pipeline.loader import load_clean_data
from clone_pipeline.selector import build_selector
from clone_pipeline.spec import CloneSpec
from clone_pipeline.applicator import apply_specs
from clone_pipeline.writer import write_cloned_dataset
from clone_pipeline.loader import _find_parquet


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(config)  # type: ignore
    return config


def main():
    parser = argparse.ArgumentParser(description="Clone generation pipeline.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    # Resolve year-dependent paths
    if config.data_year == 2023:
        cand_prefix = CANDIDATES_PREFIX
        voters_prefix = VOTERS_PREFIX
        questions_path = QUESTIONS_2023_PATH
    else:
        cand_prefix = CANDIDATES_19_PREFIX
        voters_prefix = VOTERS_19_PREFIX
        questions_path = QUESTIONS_2019_PATH

    # 1. Load
    dataframes = load_clean_data(
        CLEANED_DIR, cand_prefix, voters_prefix, questions_path
    )

    # 2. Select questions
    selector = build_selector(config.selector_type, config.selector_params)
    q_ids = selector.select(
        dataframes["questions"],
        dataframes["candidates"],
        dataframes["voters"],
    )
    print(f"Selected question IDs: {q_ids}")

    # 3. Build CloneSpecs
    specs = [
        CloneSpec(
            source_q_id=q_id,
            clone_type=spec_def["clone_type"],
            n_clones=spec_def["n_clones"],
            flip_answers=spec_def["flip_answers"],
        )
        for q_id in q_ids
        for spec_def in config.clone_specs_config
    ]

    # 4. Apply
    modified_dfs = apply_specs(
        specs, dataframes
    )  # returns candidates, voters, questions

    # 5. Derive output folder name from config name
    config_name = Path(args.config).stem  # e.g. "identical_q32214_n10"
    out_dir = CLONED_DIR / config_name

    cand_filename = _find_parquet(CLEANED_DIR, cand_prefix).name
    voters_filename = _find_parquet(CLEANED_DIR, voters_prefix).name
    questions_filename = questions_path.name

    # 6. Write
    write_cloned_dataset(
        dataframes=modified_dfs,
        specs=specs,
        out_dir=out_dir,
        cand_filename=cand_filename,
        voters_filename=voters_filename,
        questions_filename=questions_filename,
        source_dir=CLEANED_DIR,
        data_year=config.data_year,
    )


if __name__ == "__main__":
    main()
